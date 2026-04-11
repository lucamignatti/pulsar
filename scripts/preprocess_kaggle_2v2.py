#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _require_subtr_actor() -> Any:
    try:
        import subtr_actor  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-driven
        raise SystemExit(
            "subtr_actor is required for replay preprocessing. "
            "Install the offline extras or run `.venv/bin/pip install subtr-actor-py`."
        ) from exc
    return subtr_actor


def _make_rlgym_lookup_actions() -> list[dict[str, float | bool]]:
    actions: list[dict[str, float | bool]] = []

    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        {
                            "throttle": float(throttle if throttle != 0 else boost),
                            "steer": float(steer),
                            "yaw": 0.0,
                            "pitch": float(steer),
                            "roll": 0.0,
                            "jump": False,
                            "boost": boost == 1,
                            "handbrake": handbrake == 1,
                        }
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
                            {
                                "throttle": float(boost),
                                "steer": float(yaw),
                                "yaw": float(yaw),
                                "pitch": float(pitch),
                                "roll": float(roll),
                                "jump": jump == 1,
                                "boost": boost == 1,
                                "handbrake": handbrake,
                            }
                        )
    return actions


ACTION_TABLE = _make_rlgym_lookup_actions()


def _invert_xy(vec: np.ndarray, inverted: bool) -> np.ndarray:
    if not inverted:
        return vec
    return np.asarray([-vec[0], -vec[1], vec[2]], dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _rotation_to_forward_up(rotation_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pitch, yaw, roll = [float(v) for v in rotation_xyz]
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.array(
        [
            cy * sp * sr - cr * sy,
            sy * sp * sr + cr * cy,
            cp * sr,
        ],
        dtype=np.float32,
    )
    return _normalize(forward), _normalize(up)


@dataclass
class PlayerFrame:
    name: str
    team_is_zero: bool
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    boost_raw: float
    any_jump_active: float
    dodge_active: float
    jump_active: float
    double_jump_active: float


def _extract_players(meta: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = meta.get("players")
    if isinstance(candidates, list):
        return [player for player in candidates if isinstance(player, dict)]
    raise ValueError("Replay metadata does not include a usable players list.")


def _player_team_flag(player_meta: dict[str, Any]) -> bool:
    for key in ("team_is_team_0", "team_zero", "is_team_0"):
        if key in player_meta:
            return bool(player_meta[key])
    raise ValueError("Replay metadata player does not expose team_is_team_0.")


def _extract_goal_events(frames_data: dict[str, Any]) -> list[tuple[float, bool]]:
    result: list[tuple[float, bool]] = []
    for event in frames_data.get("goal_events", []):
        if not isinstance(event, dict):
            continue
        seconds_remaining = event.get("seconds_remaining")
        if seconds_remaining is None:
            seconds_remaining = event.get("time_remaining")
        if seconds_remaining is None:
            continue
        team_is_zero = bool(event.get("scoring_team_is_team_0", event.get("team_is_team_0", True)))
        result.append((float(seconds_remaining), team_is_zero))
    result.sort(key=lambda item: item[0], reverse=True)
    return result


def _next_goal_label(seconds_remaining: float, team_is_zero: bool, goal_events: list[tuple[float, bool]]) -> int:
    for goal_time_remaining, scoring_team_is_zero in goal_events:
        if goal_time_remaining < seconds_remaining - 1.0e-4:
            return 0 if scoring_team_is_zero == team_is_zero else 1
    return 2


def _append_vec(out: list[float], vec: np.ndarray, scale: float) -> None:
    out.extend((vec * scale).astype(np.float32).tolist())


def _encode_car(player: PlayerFrame, inverted: bool, boost_active: bool) -> list[float]:
    forward, up = _rotation_to_forward_up(player.rotation)
    pos = _invert_xy(player.position, inverted)
    vel = _invert_xy(player.linear_velocity, inverted)
    ang = _invert_xy(player.angular_velocity, inverted)
    forward = _invert_xy(forward, inverted)
    up = _invert_xy(up, inverted)

    values: list[float] = []
    _append_vec(values, pos, 1.0 / 2300.0)
    _append_vec(values, forward, 1.0)
    _append_vec(values, up, 1.0)
    _append_vec(values, vel, 1.0 / 2300.0)
    _append_vec(values, ang, 1.0 / math.pi)
    values.append(float(player.boost_raw / 255.0))
    values.append(0.0)
    values.append(0.0 if player.any_jump_active > 0.5 else 1.0)
    values.append(1.0 if boost_active else 0.0)
    values.append(1.0 if np.linalg.norm(player.linear_velocity[:2]) > 2200.0 else 0.0)
    return values


def _build_obs(players: list[PlayerFrame], ball: np.ndarray, ball_vel: np.ndarray, ball_ang: np.ndarray, focal: int) -> np.ndarray:
    self_player = players[focal]
    inverted = not self_player.team_is_zero
    obs: list[float] = []
    _append_vec(obs, _invert_xy(ball, inverted), 1.0 / 2300.0)
    _append_vec(obs, _invert_xy(ball_vel, inverted), 1.0 / 2300.0)
    _append_vec(obs, _invert_xy(ball_ang, inverted), 1.0 / math.pi)
    obs.extend([0.0] * 34)
    obs.extend(
        [
            1.0 if self_player.jump_active > 0.5 else 0.0,
            0.0,
            1.0 if self_player.any_jump_active > 0.5 else 0.0,
            1.0 if self_player.jump_active > 0.5 else 0.0,
            1.0 if self_player.dodge_active > 0.5 else 0.0,
            1.0 if self_player.dodge_active > 0.5 else 0.0,
            1.0 if self_player.double_jump_active > 0.5 else 0.0,
            0.0 if self_player.double_jump_active > 0.5 else 1.0,
            0.0,
        ]
    )
    obs.extend(_encode_car(self_player, inverted, False))

    allies = [idx for idx, player in enumerate(players) if idx != focal and player.team_is_zero == self_player.team_is_zero]
    opponents = [idx for idx, player in enumerate(players) if idx != focal and player.team_is_zero != self_player.team_is_zero]

    for idx in allies:
        obs.extend(_encode_car(players[idx], inverted, False))
    while len(allies) < 1:
        obs.extend([0.0] * 20)
        allies.append(-1)

    for idx in opponents:
        obs.extend(_encode_car(players[idx], inverted, False))
    while len(opponents) < 2:
        obs.extend([0.0] * 20)
        opponents.append(-1)

    return np.asarray(obs, dtype=np.float32)


def _infer_action_index(current: PlayerFrame, nxt: PlayerFrame) -> int:
    forward, _ = _rotation_to_forward_up(current.rotation)
    right = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
    delta_vel = nxt.linear_velocity - current.linear_velocity
    local_forward_accel = float(np.dot(delta_vel, forward))
    local_lateral_accel = float(np.dot(delta_vel, right))
    speed_forward = float(np.dot(nxt.linear_velocity, forward))

    throttle = 0.0
    if local_forward_accel > 25.0 or speed_forward > 400.0:
        throttle = 1.0
    elif local_forward_accel < -25.0 or speed_forward < -250.0:
        throttle = -1.0

    steer = 0.0
    if local_lateral_accel > 15.0:
        steer = 1.0
    elif local_lateral_accel < -15.0:
        steer = -1.0

    jump = nxt.jump_active > current.jump_active + 0.5 or nxt.any_jump_active > current.any_jump_active + 0.5
    boost = nxt.boost_raw < current.boost_raw - 0.5 and throttle > 0.0
    handbrake = abs(steer) > 0.0 and np.linalg.norm(current.linear_velocity[:2]) > 1200.0
    pitch = steer if jump and nxt.dodge_active > current.dodge_active + 0.5 else 0.0
    yaw = 0.0
    roll = 0.0

    target = {
        "throttle": throttle,
        "steer": steer,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "jump": jump,
        "boost": boost,
        "handbrake": handbrake,
    }

    best_idx = 0
    best_score = float("inf")
    for idx, action in enumerate(ACTION_TABLE):
        score = 0.0
        score += abs(float(action["throttle"]) - target["throttle"]) * 1.5
        score += abs(float(action["steer"]) - target["steer"]) * 1.5
        score += abs(float(action["yaw"]) - target["yaw"]) * 1.0
        score += abs(float(action["pitch"]) - target["pitch"]) * 1.0
        score += abs(float(action["roll"]) - target["roll"]) * 0.75
        score += (0.0 if bool(action["jump"]) == target["jump"] else 4.0)
        score += (0.0 if bool(action["boost"]) == target["boost"] else 3.0)
        score += (0.0 if bool(action["handbrake"]) == target["handbrake"] else 1.0)
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _frame_to_players(frame: np.ndarray, player_meta: list[dict[str, Any]], global_headers: list[str], player_headers: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[PlayerFrame]]:
    global_count = len(global_headers)
    player_count = len(player_headers)
    globals_block = frame[:global_count]
    players_block = frame[global_count:]

    ball_pos = globals_block[0:3].astype(np.float32)
    ball_vel = globals_block[6:9].astype(np.float32)
    ball_ang = globals_block[9:12].astype(np.float32)
    seconds_remaining = float(globals_block[14]) if len(global_headers) >= 15 else 0.0

    players: list[PlayerFrame] = []
    for player_index, meta in enumerate(player_meta):
        start = player_index * player_count
        values = players_block[start : start + player_count]
        if values.shape[0] != player_count:
            raise ValueError("Player block width mismatch while parsing replay frame.")
        players.append(
            PlayerFrame(
                name=str(meta.get("name", f"player_{player_index}")),
                team_is_zero=_player_team_flag(meta),
                position=values[0:3].astype(np.float32),
                rotation=values[3:6].astype(np.float32),
                linear_velocity=values[6:9].astype(np.float32),
                angular_velocity=values[9:12].astype(np.float32),
                boost_raw=float(values[12]),
                any_jump_active=float(values[13]),
                dodge_active=float(values[14]),
                jump_active=float(values[15]),
                double_jump_active=float(values[16]),
            )
        )

    return ball_pos, ball_vel, ball_ang, seconds_remaining, players


def _find_replays(dataset_root: Path, split_subdir: str) -> list[Path]:
    roots = list(dataset_root.glob(f"**/{split_subdir}"))
    replay_paths: list[Path] = []
    for root in roots or [dataset_root]:
        replay_paths.extend(root.rglob("*.replay"))
    return sorted(set(replay_paths))


def _write_manifest(path: Path, obs_dim: int, action_dim: int, shards: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "next_goal_classes": 3,
        "shards": shards,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the Kaggle high-level Rocket League 2v2 replay split into Pulsar offline tensor shards.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split-subdir", default="2v2")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    subtr_actor = _require_subtr_actor()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = _find_replays(args.dataset_root, args.split_subdir)
    if args.max_replays > 0:
        replay_paths = replay_paths[: args.max_replays]
    if not replay_paths:
        raise SystemExit("No replay files found for the requested split.")

    global_feature_adders = ["BallRigidBody", "GameTime"]
    player_feature_adders = ["PlayerRigidBody", "PlayerBoost", "PlayerAnyJump", "PlayerJump"]
    headers = subtr_actor.get_column_headers(
        global_feature_adders=global_feature_adders,
        player_feature_adders=player_feature_adders,
    )
    global_headers = list(headers["global_headers"])
    player_headers = list(headers["player_headers"])

    train_obs: list[np.ndarray] = []
    train_actions: list[int] = []
    train_next_goal: list[int] = []
    train_weights: list[float] = []
    val_obs: list[np.ndarray] = []
    val_actions: list[int] = []
    val_next_goal: list[int] = []
    val_weights: list[float] = []
    train_shards: list[dict[str, Any]] = []
    val_shards: list[dict[str, Any]] = []
    skipped = 0

    def flush(split: str, shard_index: int, obs_rows: list[np.ndarray], actions: list[int], labels: list[int], weights: list[float]) -> None:
        if not obs_rows:
            return
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        obs_tensor = torch.from_numpy(np.stack(obs_rows).astype(np.float32))
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        base = f"shard_{shard_index:05d}"
        torch.save(obs_tensor, split_dir / f"{base}_obs.pt")
        torch.save(actions_tensor, split_dir / f"{base}_actions.pt")
        torch.save(labels_tensor, split_dir / f"{base}_next_goal.pt")
        torch.save(weights_tensor, split_dir / f"{base}_weights.pt")
        target = train_shards if split == "train" else val_shards
        target.append(
            {
                "obs_path": f"{split}/{base}_obs.pt",
                "actions_path": f"{split}/{base}_actions.pt",
                "next_goal_path": f"{split}/{base}_next_goal.pt",
                "weights_path": f"{split}/{base}_weights.pt",
                "samples": len(obs_rows),
            }
        )
        obs_rows.clear()
        actions.clear()
        labels.clear()
        weights.clear()

    train_shard_index = 0
    val_shard_index = 0

    for replay_path in replay_paths:
        try:
            meta, ndarray = subtr_actor.get_ndarray_with_info_from_replay_filepath(
                str(replay_path),
                global_feature_adders=global_feature_adders,
                player_feature_adders=player_feature_adders,
                fps=args.fps,
                dtype="float32",
            )
            frames_data = subtr_actor.get_replay_frames_data(str(replay_path))
            players_meta = _extract_players(meta)
            if len(players_meta) != 4:
                skipped += 1
                continue
            goal_events = _extract_goal_events(frames_data)
            if ndarray.shape[0] < 2:
                skipped += 1
                continue

            is_train = rng.random() < args.train_fraction
            obs_rows = train_obs if is_train else val_obs
            actions = train_actions if is_train else val_actions
            labels = train_next_goal if is_train else val_next_goal
            weights = train_weights if is_train else val_weights

            for frame_index in range(ndarray.shape[0] - 1):
                ball, ball_vel, ball_ang, seconds_remaining, players = _frame_to_players(
                    ndarray[frame_index],
                    players_meta,
                    global_headers,
                    player_headers,
                )
                _, _, _, _, next_players = _frame_to_players(
                    ndarray[frame_index + 1],
                    players_meta,
                    global_headers,
                    player_headers,
                )
                for player_index, player in enumerate(players):
                    obs_rows.append(_build_obs(players, ball, ball_vel, ball_ang, player_index))
                    actions.append(_infer_action_index(player, next_players[player_index]))
                    labels.append(_next_goal_label(seconds_remaining, player.team_is_zero, goal_events))
                    weights.append(1.0)

                if len(obs_rows) >= args.shard_size:
                    if is_train:
                        flush("train", train_shard_index, train_obs, train_actions, train_next_goal, train_weights)
                        train_shard_index += 1
                    else:
                        flush("val", val_shard_index, val_obs, val_actions, val_next_goal, val_weights)
                        val_shard_index += 1
        except Exception:
            skipped += 1
            continue

    flush("train", train_shard_index, train_obs, train_actions, train_next_goal, train_weights)
    flush("val", val_shard_index, val_obs, val_actions, val_next_goal, val_weights)

    if not train_shards:
        raise SystemExit("No training shards were produced. Check the dataset path and parser assumptions.")

    obs_dim = 132
    _write_manifest(args.output_dir / "train_manifest.json", obs_dim, len(ACTION_TABLE), train_shards)
    _write_manifest(args.output_dir / "val_manifest.json", obs_dim, len(ACTION_TABLE), val_shards or train_shards)
    print(f"wrote {len(train_shards)} train shards and {len(val_shards)} val shards")
    print(f"skipped_replays={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
