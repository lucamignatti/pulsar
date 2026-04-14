#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


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
    terminated: bool = False
    truncated: bool = False

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.next_goal.clear()
        self.weights.clear()
        self.terminated = False
        self.truncated = False

    def __bool__(self) -> bool:
        return bool(self.obs)


@dataclass(slots=True)
class WorkerResult:
    train_shards: list[dict[str, Any]] = field(default_factory=list)
    val_shards: list[dict[str, Any]] = field(default_factory=list)
    obs_dim: int | None = None
    exact_samples: int = 0
    skipped_replays: int = 0
    skipped_unbalanced_frames: int = 0
    skipped_bad_obs_frames: int = 0


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


def _has_two_cars_per_team(cars: dict[int, Any], blue_team: int, orange_team: int) -> bool:
    blue_count = 0
    orange_count = 0
    for car in cars.values():
        if car.team_num == blue_team:
            blue_count += 1
        elif car.team_num == orange_team:
            orange_count += 1
        else:
            return False
    return blue_count == 2 and orange_count == 2


def _stack_rows(rows: list[np.ndarray], label: str, split: str, shard_name: str) -> np.ndarray:
    shapes = sorted({tuple(np.asarray(row).shape) for row in rows})
    if len(shapes) != 1:
        raise RuntimeError(f"Inconsistent {label} shapes in {split} shard {shard_name}: {shapes}")
    return np.stack(rows).astype(np.float32)


def _assign_split(replay_path: str, seed: int, train_fraction: float) -> str:
    digest = hashlib.blake2b(f"{seed}:{replay_path}".encode("utf-8"), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / float(1 << 64)
    return "train" if draw < train_fraction else "val"


def _make_shard_name(prefix: str, shard_index: int) -> str:
    return f"{prefix}shard_{shard_index:05d}"


def _build_worker_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_dir": str(args.output_dir),
        "shard_size": int(args.shard_size),
        "seed": int(args.seed),
        "interpolation": str(args.interpolation),
        "no_predict_pyr": bool(args.no_predict_pyr),
        "max_update_age": float(args.max_update_age),
        "action_target_mode": str(args.action_target_mode),
        "dodge_deadzone": float(args.dodge_deadzone),
        "train_fraction": float(args.train_fraction),
    }


def _process_replay_subset(
    replay_paths: list[str],
    worker_config: dict[str, Any],
    shard_prefix: str,
) -> WorkerResult:
    mods = _require_replay_stack()
    blue_team = mods["BLUE_TEAM"]
    orange_team = mods["ORANGE_TEAM"]
    parsed_replay_cls = mods["ParsedReplay"]
    replay_to_rlgym = mods["replay_to_rlgym"]
    default_obs_cls = mods["DefaultObs"]
    get_best_action_options = mods["get_best_action_options"]
    get_weighted_action_options = mods["get_weighted_action_options"]

    output_dir = Path(worker_config["output_dir"])
    shard_size = int(worker_config["shard_size"])
    seed = int(worker_config["seed"])
    interpolation = str(worker_config["interpolation"])
    no_predict_pyr = bool(worker_config["no_predict_pyr"])
    max_update_age = float(worker_config["max_update_age"])
    action_target_mode = str(worker_config["action_target_mode"])
    dodge_deadzone = float(worker_config["dodge_deadzone"])
    train_fraction = float(worker_config["train_fraction"])

    train_obs: list[np.ndarray] = []
    train_actions: list[int] = []
    train_action_probs: list[np.ndarray] = []
    train_next_goal: list[int] = []
    train_weights: list[float] = []
    train_episode_starts: list[float] = []
    train_terminated: list[float] = []
    train_truncated: list[float] = []
    val_obs: list[np.ndarray] = []
    val_actions: list[int] = []
    val_action_probs: list[np.ndarray] = []
    val_next_goal: list[int] = []
    val_weights: list[float] = []
    val_episode_starts: list[float] = []
    val_terminated: list[float] = []
    val_truncated: list[float] = []
    result = WorkerResult()

    train_shard_index = 0
    val_shard_index = 0

    def flush(
        split: str,
        shard_index: int,
        obs_rows: list[np.ndarray],
        actions: list[int],
        action_probs: list[np.ndarray],
        labels: list[int],
        weights: list[float],
        episode_starts: list[float],
        terminated: list[float],
        truncated: list[float],
    ) -> None:
        if not obs_rows:
            return

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_name = _make_shard_name(shard_prefix, shard_index)
        obs_tensor = torch.from_numpy(_stack_rows(obs_rows, "observation", split, shard_name))
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        action_probs_tensor = torch.from_numpy(
            _stack_rows(action_probs, "action probability", split, shard_name)
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        episode_starts_tensor = torch.tensor(episode_starts, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminated, dtype=torch.float32)
        truncated_tensor = torch.tensor(truncated, dtype=torch.float32)

        torch.save(obs_tensor, split_dir / f"{shard_name}_obs.pt")
        torch.save(actions_tensor, split_dir / f"{shard_name}_actions.pt")
        torch.save(action_probs_tensor, split_dir / f"{shard_name}_action_probs.pt")
        torch.save(labels_tensor, split_dir / f"{shard_name}_next_goal.pt")
        torch.save(weights_tensor, split_dir / f"{shard_name}_weights.pt")
        torch.save(episode_starts_tensor, split_dir / f"{shard_name}_episode_starts.pt")
        torch.save(terminated_tensor, split_dir / f"{shard_name}_terminated.pt")
        torch.save(truncated_tensor, split_dir / f"{shard_name}_truncated.pt")

        target = result.train_shards if split == "train" else result.val_shards
        target.append(
            {
                "obs_path": f"{split}/{shard_name}_obs.pt",
                "actions_path": f"{split}/{shard_name}_actions.pt",
                "action_probs_path": f"{split}/{shard_name}_action_probs.pt",
                "next_goal_path": f"{split}/{shard_name}_next_goal.pt",
                "weights_path": f"{split}/{shard_name}_weights.pt",
                "episode_starts_path": f"{split}/{shard_name}_episode_starts.pt",
                "terminated_path": f"{split}/{shard_name}_terminated.pt",
                "truncated_path": f"{split}/{shard_name}_truncated.pt",
                "samples": len(obs_rows),
            }
        )
        obs_rows.clear()
        actions.clear()
        action_probs.clear()
        labels.clear()
        weights.clear()
        episode_starts.clear()
        terminated.clear()
        truncated.clear()

    def close_trajectory(split: str, trajectory: TrajectoryBuffer, *, terminated_end: bool, truncated_end: bool) -> None:
        nonlocal train_shard_index, val_shard_index
        if not trajectory:
            return

        trajectory.terminated = terminated_end
        trajectory.truncated = truncated_end

        if split == "train":
            obs_rows = train_obs
            actions = train_actions
            action_probs = train_action_probs
            labels = train_next_goal
            weights = train_weights
            episode_starts = train_episode_starts
            terminated = train_terminated
            truncated = train_truncated
            shard_index = train_shard_index
        else:
            obs_rows = val_obs
            actions = val_actions
            action_probs = val_action_probs
            labels = val_next_goal
            weights = val_weights
            episode_starts = val_episode_starts
            terminated = val_terminated
            truncated = val_truncated
            shard_index = val_shard_index

        if obs_rows and len(obs_rows) + len(trajectory.obs) > shard_size:
            flush(
                split,
                shard_index,
                obs_rows,
                actions,
                action_probs,
                labels,
                weights,
                episode_starts,
                terminated,
                truncated,
            )
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
        terminated.extend([0.0] * len(trajectory.obs))
        truncated.extend([0.0] * len(trajectory.obs))
        terminated[-1] = 1.0 if trajectory.terminated else 0.0
        truncated[-1] = 1.0 if trajectory.truncated else 0.0
        trajectory.clear()

    for replay_path_str in replay_paths:
        replay_path = Path(replay_path_str)
        try:
            replay = parsed_replay_cls.load(replay_path)
            obs_builder = default_obs_cls(zero_padding=2)
            trajectories: dict[int, TrajectoryBuffer] = {}
            split = _assign_split(replay_path.as_posix(), seed, train_fraction)
            first_frame = True
            agent_order: list[int] = []

            for frame in replay_to_rlgym(
                replay,
                interpolation=interpolation,
                predict_pyr=not no_predict_pyr,
            ):
                if len(frame.state.cars) != 4:
                    continue
                if not _has_two_cars_per_team(frame.state.cars, blue_team, orange_team):
                    if not first_frame:
                        for agent_id in agent_order:
                            close_trajectory(split, trajectories[agent_id], terminated_end=False, truncated_end=True)
                    result.skipped_unbalanced_frames += 1
                    continue
                if first_frame:
                    agent_order = sorted(frame.state.cars.keys())
                    obs_builder.reset(agent_order, frame.state, {})
                    trajectories = {agent_id: TrajectoryBuffer() for agent_id in agent_order}
                    first_frame = False

                obs_map = obs_builder.build_obs(agent_order, frame.state, {})
                obs_rows_by_agent: dict[int, np.ndarray] = {}
                bad_obs_frame = False

                for agent_id in agent_order:
                    obs_row = np.asarray(obs_map[agent_id], dtype=np.float32).reshape(-1)
                    if not np.all(np.isfinite(obs_row)):
                        bad_obs_frame = True
                        break
                    if result.obs_dim is None:
                        result.obs_dim = int(obs_row.shape[0])
                    if obs_row.shape != (result.obs_dim,):
                        bad_obs_frame = True
                        break
                    obs_rows_by_agent[agent_id] = obs_row

                if bad_obs_frame:
                    for agent_id in agent_order:
                        close_trajectory(split, trajectories[agent_id], terminated_end=False, truncated_end=True)
                    result.skipped_bad_obs_frames += 1
                    continue

                for agent_id in agent_order:
                    car = frame.state.cars[agent_id]
                    replay_action = np.asarray(frame.actions.get(agent_id), dtype=np.float32)
                    update_age = float(frame.update_age.get(agent_id, 0.0))
                    trajectory = trajectories[agent_id]

                    if (
                        car.is_demoed
                        or not np.all(np.isfinite(replay_action))
                        or replay_action.shape != (8,)
                        or not _is_fresh_update(update_age, max_update_age)
                    ):
                        close_trajectory(split, trajectory, terminated_end=False, truncated_end=True)
                        continue

                    if action_target_mode == "weighted":
                        action_probs = get_weighted_action_options(
                            car,
                            replay_action,
                            ACTION_OPTIONS,
                            dodge_deadzone=dodge_deadzone,
                        )
                    else:
                        action_probs = get_best_action_options(
                            car,
                            replay_action,
                            ACTION_OPTIONS,
                            dodge_deadzone=dodge_deadzone,
                            greedy=True,
                        )
                    action_probs = np.asarray(action_probs, dtype=np.float32)
                    total_prob = float(action_probs.sum())
                    if total_prob <= 0.0 or not np.all(np.isfinite(action_probs)):
                        close_trajectory(split, trajectory, terminated_end=False, truncated_end=True)
                        continue
                    action_probs = action_probs / total_prob

                    trajectory.obs.append(obs_rows_by_agent[agent_id])
                    trajectory.action_probs.append(action_probs)
                    trajectory.actions.append(int(np.argmax(action_probs)))
                    trajectory.next_goal.append(_next_goal_label(car.team_num, frame.next_scoring_team))
                    trajectory.weights.append(_sample_weight(action_probs))
                    result.exact_samples += 1

                if frame.scoreboard.go_to_kickoff or frame.scoreboard.is_over:
                    for agent_id in agent_order:
                        close_trajectory(split, trajectories[agent_id], terminated_end=True, truncated_end=False)

            for trajectory in trajectories.values():
                close_trajectory(split, trajectory, terminated_end=False, truncated_end=True)
        except Exception as exc:
            print(f"error while processing replay: {replay_path} ({exc!r})", flush=True)
            result.skipped_replays += 1
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
        train_terminated,
        train_truncated,
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
        val_terminated,
        val_truncated,
    )
    return result


def _process_replay_subset_job(job: tuple[list[str], dict[str, Any], str]) -> WorkerResult:
    replay_paths, worker_config, shard_prefix = job
    return _process_replay_subset(replay_paths, worker_config, shard_prefix)


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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--interpolation", choices=("none", "linear", "rocketsim"), default="rocketsim")
    parser.add_argument("--no-predict-pyr", action="store_true")
    parser.add_argument("--max-update-age", type=float, default=0.0)
    parser.add_argument("--action-target-mode", choices=("weighted", "best"), default="weighted")
    parser.add_argument("--dodge-deadzone", type=float, default=0.5)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = _find_replays(args.dataset_root, args.split_subdir)
    if args.max_replays > 0:
        replay_paths = replay_paths[: args.max_replays]
    if not replay_paths:
        raise SystemExit("No replay files found for the requested split.")

    replay_path_strs = [path.as_posix() for path in replay_paths]
    worker_count = min(args.workers, len(replay_path_strs))
    worker_config = _build_worker_config(args)
    job_width = max(2, len(str(max(len(replay_path_strs) - 1, 0))))
    jobs = [
        (
            [replay_path],
            worker_config,
            f"r{job_index:0{job_width}d}_",
        )
        for job_index, replay_path in enumerate(replay_path_strs)
    ]

    progress = tqdm(total=len(replay_path_strs), desc="Preprocess", unit="replay")
    obs_dim: int | None = None
    train_shards: list[dict[str, Any]] = []
    val_shards: list[dict[str, Any]] = []
    exact_samples = 0
    skipped_replays = 0
    skipped_unbalanced_frames = 0
    skipped_bad_obs_frames = 0
    try:
        if worker_count == 1:
            for replay_job, worker_config_chunk, shard_prefix in jobs:
                result = _process_replay_subset(replay_job, worker_config_chunk, shard_prefix)
                if result.obs_dim is not None:
                    if obs_dim is None:
                        obs_dim = result.obs_dim
                    elif obs_dim != result.obs_dim:
                        raise SystemExit(f"Inconsistent observation widths across workers: {[obs_dim, result.obs_dim]}")
                train_shards.extend(result.train_shards)
                val_shards.extend(result.val_shards)
                exact_samples += result.exact_samples
                skipped_replays += result.skipped_replays
                skipped_unbalanced_frames += result.skipped_unbalanced_frames
                skipped_bad_obs_frames += result.skipped_bad_obs_frames
                progress.update(1)
        else:
            try:
                with ProcessPoolExecutor(max_workers=worker_count, max_tasks_per_child=1) as executor:
                    future_to_chunk_size = {
                        executor.submit(_process_replay_subset_job, job): len(job[0])
                        for job in jobs
                    }
                    for future in as_completed(future_to_chunk_size):
                        result = future.result()
                        if result.obs_dim is not None:
                            if obs_dim is None:
                                obs_dim = result.obs_dim
                            elif obs_dim != result.obs_dim:
                                raise SystemExit(
                                    f"Inconsistent observation widths across workers: {[obs_dim, result.obs_dim]}"
                                )
                        train_shards.extend(result.train_shards)
                        val_shards.extend(result.val_shards)
                        exact_samples += result.exact_samples
                        skipped_replays += result.skipped_replays
                        skipped_unbalanced_frames += result.skipped_unbalanced_frames
                        skipped_bad_obs_frames += result.skipped_bad_obs_frames
                        progress.update(1)
            except PermissionError as exc:
                print(
                    f"worker_pool_unavailable={exc}; falling back to single-process preprocessing",
                    flush=True,
                )
                worker_count = 1
                for replay_job, worker_config_chunk, shard_prefix in jobs:
                    result = _process_replay_subset(replay_job, worker_config_chunk, shard_prefix)
                    if result.obs_dim is not None:
                        if obs_dim is None:
                            obs_dim = result.obs_dim
                        elif obs_dim != result.obs_dim:
                            raise SystemExit(
                                f"Inconsistent observation widths across workers: {[obs_dim, result.obs_dim]}"
                            )
                    train_shards.extend(result.train_shards)
                    val_shards.extend(result.val_shards)
                    exact_samples += result.exact_samples
                    skipped_replays += result.skipped_replays
                    skipped_unbalanced_frames += result.skipped_unbalanced_frames
                    skipped_bad_obs_frames += result.skipped_bad_obs_frames
                    progress.update(1)
    finally:
        progress.close()

    if obs_dim is None:
        raise SystemExit("No valid observations were produced. Check replay interpolation and parser output.")

    train_shards.sort(key=lambda shard: shard["obs_path"])
    val_shards.sort(key=lambda shard: shard["obs_path"])

    if not train_shards:
        raise SystemExit("No training shards were produced. Check dataset path and replay parser dependencies.")

    _write_manifest(args.output_dir / "train_manifest.json", obs_dim, ACTION_OPTIONS.shape[0], train_shards)
    _write_manifest(args.output_dir / "val_manifest.json", obs_dim, ACTION_OPTIONS.shape[0], val_shards or train_shards)
    print(f"workers={worker_count}")
    print(f"wrote {len(train_shards)} train shards and {len(val_shards)} val shards")
    print(f"exact_samples={exact_samples}")
    print(f"skipped_replays={skipped_replays}")
    print(f"skipped_unbalanced_frames={skipped_unbalanced_frames}")
    print(f"skipped_bad_obs_frames={skipped_bad_obs_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
