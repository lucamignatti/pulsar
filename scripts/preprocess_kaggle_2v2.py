#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _require_replay_stack() -> dict[str, Any]:
    try:
        from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
        from rlgym.rocket_league.obs_builders import DefaultObs
        from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
        from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
        from rlgym_tools.rocket_league.replays.pick_action import (
            get_best_action_options,
            get_weighted_action_options,
        )
    except ImportError as exc:
        raise SystemExit(
            "rlgym-tools, pandas, and pyarrow are required for replay preprocessing. "
            "Install the offline extras with `.venv/bin/pip install -e .[offline]`."
        ) from exc

    return {
        "BLUE_TEAM": BLUE_TEAM,
        "ORANGE_TEAM": ORANGE_TEAM,
        "DefaultObs": DefaultObs,
        "ParsedReplay": ParsedReplay,
        "replay_to_rlgym": replay_to_rlgym,
        "get_best_action_options": get_best_action_options,
        "get_weighted_action_options": get_weighted_action_options,
    }


def _make_rlgym_lookup_actions() -> np.ndarray:
    actions: list[list[float]] = []

    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [
                            float(throttle if throttle != 0 else boost),
                            float(steer),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(boost),
                            float(handbrake),
                        ]
                    )

    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == 0 and roll == 0 and jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [
                                float(boost),
                                float(yaw),
                                float(pitch),
                                float(yaw),
                                float(roll),
                                float(jump),
                                float(boost),
                                float(handbrake),
                            ]
                        )

    table = np.asarray(actions, dtype=np.float32)
    if table.shape != (90, 8):
        raise RuntimeError(f"Unexpected lookup table shape: {table.shape}")
    return table


ACTION_OPTIONS = _make_rlgym_lookup_actions()


@dataclass(slots=True)
class TrajectoryBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    action_probs: list[np.ndarray] = field(default_factory=list)
    next_goal: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.next_goal.clear()
        self.weights.clear()

    def __bool__(self) -> bool:
        return bool(self.obs)


def _find_replays(dataset_root: Path, split_subdir: str) -> list[Path]:
    roots = list(dataset_root.glob(f"**/{split_subdir}"))
    replay_paths: list[Path] = []
    for root in roots or [dataset_root]:
        replay_paths.extend(root.rglob("*.replay"))
    return sorted(set(replay_paths))


def _write_manifest(path: Path, obs_dim: int, action_dim: int, shards: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 3,
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "next_goal_classes": 3,
        "shards": shards,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _next_goal_label(team_num: int, next_scoring_team: int | None) -> int:
    if next_scoring_team is None:
        return 2
    return 0 if next_scoring_team == team_num else 1


def _sample_weight(action_probs: np.ndarray) -> float:
    # Weighted labels already encode ambiguity; keep sample weights close to 1 while
    # downweighting extremely diffuse targets a bit.
    max_prob = float(np.max(action_probs))
    return max(0.25, max_prob)


def _is_fresh_update(update_age: float, max_update_age: float) -> bool:
    return float(update_age) <= max_update_age + 1.0e-9


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the Kaggle high-level Rocket League 2v2 replay split into Pulsar offline trajectory shards."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split-subdir", default="2v2")
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--interpolation", choices=("none", "linear", "rocketsim"), default="rocketsim")
    parser.add_argument("--no-predict-pyr", action="store_true")
    parser.add_argument("--max-update-age", type=float, default=0.0)
    parser.add_argument("--action-target-mode", choices=("weighted", "best"), default="weighted")
    parser.add_argument("--dodge-deadzone", type=float, default=0.5)
    args = parser.parse_args()

    mods = _require_replay_stack()
    ParsedReplay = mods["ParsedReplay"]
    replay_to_rlgym = mods["replay_to_rlgym"]
    DefaultObs = mods["DefaultObs"]
    get_best_action_options = mods["get_best_action_options"]
    get_weighted_action_options = mods["get_weighted_action_options"]

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = _find_replays(args.dataset_root, args.split_subdir)
    if args.max_replays > 0:
      replay_paths = replay_paths[: args.max_replays]
    if not replay_paths:
        raise SystemExit("No replay files found for the requested split.")

    train_obs: list[np.ndarray] = []
    train_actions: list[int] = []
    train_action_probs: list[np.ndarray] = []
    train_next_goal: list[int] = []
    train_weights: list[float] = []
    train_episode_starts: list[float] = []
    val_obs: list[np.ndarray] = []
    val_actions: list[int] = []
    val_action_probs: list[np.ndarray] = []
    val_next_goal: list[int] = []
    val_weights: list[float] = []
    val_episode_starts: list[float] = []
    train_shards: list[dict[str, Any]] = []
    val_shards: list[dict[str, Any]] = []
    skipped = 0
    exact_samples = 0

    def flush(
        split: str,
        shard_index: int,
        obs_rows: list[np.ndarray],
        actions: list[int],
        action_probs: list[np.ndarray],
        labels: list[int],
        weights: list[float],
        episode_starts: list[float],
    ) -> None:
        if not obs_rows:
            return
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        obs_tensor = torch.from_numpy(np.stack(obs_rows).astype(np.float32))
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        action_probs_tensor = torch.from_numpy(np.stack(action_probs).astype(np.float32))
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        episode_starts_tensor = torch.tensor(episode_starts, dtype=torch.float32)
        base = f"shard_{shard_index:05d}"
        torch.save(obs_tensor, split_dir / f"{base}_obs.pt")
        torch.save(actions_tensor, split_dir / f"{base}_actions.pt")
        torch.save(action_probs_tensor, split_dir / f"{base}_action_probs.pt")
        torch.save(labels_tensor, split_dir / f"{base}_next_goal.pt")
        torch.save(weights_tensor, split_dir / f"{base}_weights.pt")
        torch.save(episode_starts_tensor, split_dir / f"{base}_episode_starts.pt")
        target = train_shards if split == "train" else val_shards
        target.append(
            {
                "obs_path": f"{split}/{base}_obs.pt",
                "actions_path": f"{split}/{base}_actions.pt",
                "action_probs_path": f"{split}/{base}_action_probs.pt",
                "next_goal_path": f"{split}/{base}_next_goal.pt",
                "weights_path": f"{split}/{base}_weights.pt",
                "episode_starts_path": f"{split}/{base}_episode_starts.pt",
                "samples": len(obs_rows),
            }
        )
        obs_rows.clear()
        actions.clear()
        action_probs.clear()
        labels.clear()
        weights.clear()
        episode_starts.clear()

    train_shard_index = 0
    val_shard_index = 0

    def append_trajectory(split: str, trajectory: TrajectoryBuffer) -> None:
        nonlocal train_shard_index, val_shard_index
        if not trajectory:
            return

        if split == "train":
            obs_rows = train_obs
            actions = train_actions
            action_probs = train_action_probs
            labels = train_next_goal
            weights = train_weights
            episode_starts = train_episode_starts
            shard_index = train_shard_index
        else:
            obs_rows = val_obs
            actions = val_actions
            action_probs = val_action_probs
            labels = val_next_goal
            weights = val_weights
            episode_starts = val_episode_starts
            shard_index = val_shard_index

        if obs_rows and len(obs_rows) + len(trajectory.obs) > args.shard_size:
            flush(split, shard_index, obs_rows, actions, action_probs, labels, weights, episode_starts)
            if split == "train":
                train_shard_index += 1
            else:
                val_shard_index += 1

        obs_rows.extend(trajectory.obs)
        actions.extend(trajectory.actions)
        action_probs.extend(trajectory.action_probs)
        labels.extend(trajectory.next_goal)
        weights.extend(trajectory.weights)
        episode_starts.extend([1.0] + [0.0] * (len(trajectory.obs) - 1))
        trajectory.clear()

    for replay_path in replay_paths:
        try:
            replay = ParsedReplay.load(replay_path)
            obs_builder = DefaultObs(zero_padding=2)
            trajectories: dict[int, TrajectoryBuffer] = {}
            split = "train" if rng.random() < args.train_fraction else "val"
            first_frame = True
            agent_order: list[int] = []

            for frame in replay_to_rlgym(
                replay,
                interpolation=args.interpolation,
                predict_pyr=not args.no_predict_pyr,
            ):
                if len(frame.state.cars) != 4:
                    continue
                if first_frame:
                    agent_order = sorted(frame.state.cars.keys())
                    obs_builder.reset(agent_order, frame.state, {})
                    trajectories = {agent_id: TrajectoryBuffer() for agent_id in agent_order}
                    first_frame = False

                obs_map = obs_builder.build_obs(agent_order, frame.state, {})

                for agent_id in agent_order:
                    car = frame.state.cars[agent_id]
                    replay_action = np.asarray(frame.actions.get(agent_id), dtype=np.float32)
                    update_age = float(frame.update_age.get(agent_id, 0.0))
                    trajectory = trajectories[agent_id]

                    if (
                        car.is_demoed
                        or not np.all(np.isfinite(replay_action))
                        or replay_action.shape != (8,)
                        or not _is_fresh_update(update_age, args.max_update_age)
                    ):
                        append_trajectory(split, trajectory)
                        continue

                    if args.action_target_mode == "weighted":
                        action_probs = get_weighted_action_options(
                            car,
                            replay_action,
                            ACTION_OPTIONS,
                            dodge_deadzone=args.dodge_deadzone,
                        )
                    else:
                        action_probs = get_best_action_options(
                            car,
                            replay_action,
                            ACTION_OPTIONS,
                            dodge_deadzone=args.dodge_deadzone,
                            greedy=True,
                        )
                    action_probs = np.asarray(action_probs, dtype=np.float32)
                    total_prob = float(action_probs.sum())
                    if total_prob <= 0.0 or not np.all(np.isfinite(action_probs)):
                        append_trajectory(split, trajectory)
                        continue
                    action_probs = action_probs / total_prob

                    trajectory.obs.append(np.asarray(obs_map[agent_id], dtype=np.float32))
                    trajectory.action_probs.append(action_probs)
                    trajectory.actions.append(int(np.argmax(action_probs)))
                    trajectory.next_goal.append(_next_goal_label(car.team_num, frame.next_scoring_team))
                    trajectory.weights.append(_sample_weight(action_probs))
                    exact_samples += 1

                if frame.scoreboard.go_to_kickoff or frame.scoreboard.is_over:
                    for agent_id in agent_order:
                        append_trajectory(split, trajectories[agent_id])

            for trajectory in trajectories.values():
                append_trajectory(split, trajectory)
        except Exception:
            skipped += 1
            continue

    flush(
        "train",
        train_shard_index,
        train_obs,
        train_actions,
        train_action_probs,
        train_next_goal,
        train_weights,
        train_episode_starts,
    )
    flush(
        "val",
        val_shard_index,
        val_obs,
        val_actions,
        val_action_probs,
        val_next_goal,
        val_weights,
        val_episode_starts,
    )

    if not train_shards:
        raise SystemExit("No training shards were produced. Check dataset path and replay parser dependencies.")

    obs_dim = 132
    _write_manifest(args.output_dir / "train_manifest.json", obs_dim, ACTION_OPTIONS.shape[0], train_shards)
    _write_manifest(args.output_dir / "val_manifest.json", obs_dim, ACTION_OPTIONS.shape[0], val_shards or train_shards)
    print(f"wrote {len(train_shards)} train shards and {len(val_shards)} val shards")
    print(f"exact_samples={exact_samples}")
    print(f"skipped_replays={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
