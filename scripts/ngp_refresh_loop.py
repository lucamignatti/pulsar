#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _phase_name(index: int) -> str:
    return f"cycle_{index:04d}"


def _normalize_shard_paths(manifest_path: Path, shard: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(shard)
    base = manifest_path.parent
    for key in [
        "obs_path",
        "actions_path",
        "action_probs_path",
        "next_goal_path",
        "weights_path",
        "episode_starts_path",
        "terminated_path",
        "truncated_path",
    ]:
        value = normalized.get(key)
        if value:
            normalized[key] = str((base / value).resolve())
    return normalized


def _load_manifest_bundle(manifest_paths: list[Path]) -> tuple[int, int, int, list[dict[str, Any]], int]:
    if not manifest_paths:
        raise RuntimeError("at least one manifest is required")

    observation_dim: int | None = None
    action_dim: int | None = None
    next_goal_classes: int | None = None
    shards: list[dict[str, Any]] = []
    sample_count = 0

    for manifest_path in manifest_paths:
        manifest = _load_json(manifest_path)
        current_obs_dim = int(manifest["observation_dim"])
        current_action_dim = int(manifest["action_dim"])
        current_next_goal_classes = int(manifest.get("next_goal_classes", 3))
        if observation_dim is None:
            observation_dim = current_obs_dim
            action_dim = current_action_dim
            next_goal_classes = current_next_goal_classes
        else:
            if current_obs_dim != observation_dim:
                raise RuntimeError(f"observation_dim mismatch in {manifest_path}")
            if current_action_dim != action_dim:
                raise RuntimeError(f"action_dim mismatch in {manifest_path}")
            if current_next_goal_classes != next_goal_classes:
                raise RuntimeError(f"next_goal_classes mismatch in {manifest_path}")

        for shard in manifest["shards"]:
            normalized = _normalize_shard_paths(manifest_path, shard)
            sample_count += int(normalized.get("samples", 0))
            shards.append(normalized)

    return observation_dim or 0, action_dim or 0, next_goal_classes or 3, shards, sample_count


def _write_manifest(
    output_path: Path,
    observation_dim: int,
    action_dim: int,
    next_goal_classes: int,
    shards: list[dict[str, Any]],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "next_goal_classes": next_goal_classes,
        "shards": shards,
    }
    _write_json(output_path, payload)
    return sum(int(shard.get("samples", 0)) for shard in shards)


def _select_anchor_subset(
    anchor_manifest_paths: list[Path],
    target_samples: int,
    seed: int,
) -> tuple[int, int, int, list[dict[str, Any]], int]:
    observation_dim, action_dim, next_goal_classes, shards, total_samples = _load_manifest_bundle(anchor_manifest_paths)
    if target_samples <= 0 or not shards:
        return observation_dim, action_dim, next_goal_classes, [], 0
    if target_samples >= total_samples:
        return observation_dim, action_dim, next_goal_classes, shards, total_samples

    rng = random.Random(seed)
    shuffled = list(shards)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    selected_samples = 0
    for shard in shuffled:
        selected.append(shard)
        selected_samples += int(shard.get("samples", 0))
        if selected_samples >= target_samples:
            break
    return observation_dim, action_dim, next_goal_classes, selected, selected_samples


def _merge_all_manifests(manifest_paths: list[Path], output_path: Path) -> int:
    observation_dim, action_dim, next_goal_classes, shards, _ = _load_manifest_bundle(manifest_paths)
    return _write_manifest(output_path, observation_dim, action_dim, next_goal_classes, shards)


def _build_mixed_train_manifest(
    output_path: Path,
    anchor_manifest_paths: list[Path],
    online_manifest_paths: list[Path],
    old_data_fraction: float,
    seed: int,
) -> tuple[int, int]:
    online_obs_dim, online_action_dim, next_goal_classes, online_shards, online_samples = _load_manifest_bundle(
        online_manifest_paths
    )
    if online_samples <= 0:
        raise RuntimeError("cannot build a mixed training manifest without online samples")

    if not (0.0 <= old_data_fraction < 1.0):
        raise RuntimeError("old_data_fraction must be in [0, 1)")

    if old_data_fraction == 0.0:
        anchor_samples = 0
        selected_anchor: list[dict[str, Any]] = []
        anchor_obs_dim = online_obs_dim
        anchor_action_dim = online_action_dim
    else:
        target_anchor_samples = int(math.ceil(online_samples * old_data_fraction / (1.0 - old_data_fraction)))
        anchor_obs_dim, anchor_action_dim, _, selected_anchor, anchor_samples = _select_anchor_subset(
            anchor_manifest_paths,
            target_anchor_samples,
            seed,
        )
        if anchor_obs_dim != online_obs_dim or anchor_action_dim != online_action_dim:
            raise RuntimeError("anchor and online manifests disagree on dimensions")

    _write_manifest(
        output_path,
        online_obs_dim,
        online_action_dim,
        next_goal_classes,
        [*selected_anchor, *online_shards],
    )
    return online_samples, anchor_samples


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def _read_latest_metrics_line(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        raise RuntimeError(f"missing metrics file: {metrics_path}")
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"metrics file is empty: {metrics_path}")
    return json.loads(lines[-1])


def _make_eval_config(
    base_offline: dict[str, Any],
    checkpoint_path: str,
    val_manifest: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_offline))
    config["offline_dataset"]["train_manifest"] = str(val_manifest.resolve())
    config["offline_dataset"]["val_manifest"] = str(val_manifest.resolve())
    config["behavior_cloning"]["enabled"] = False
    config["behavior_cloning"]["epochs"] = 0
    config["next_goal_predictor"]["enabled"] = True
    config["next_goal_predictor"]["init_checkpoint"] = checkpoint_path
    config["next_goal_predictor"]["epochs"] = 0
    config["next_goal_predictor"]["reuse_normalizer"] = True
    config["value_pretraining"]["enabled"] = False
    config["value_pretraining"]["epochs"] = 0
    config["wandb"]["enabled"] = False
    config["wandb"]["job_type"] = "ngp_eval"
    return config


def _make_candidate_config(
    base_offline: dict[str, Any],
    init_checkpoint: str,
    train_manifest: Path,
    val_manifest: Path,
    epochs: int,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_offline))
    config["offline_dataset"]["train_manifest"] = str(train_manifest.resolve())
    config["offline_dataset"]["val_manifest"] = str(val_manifest.resolve())
    config["behavior_cloning"]["enabled"] = False
    config["behavior_cloning"]["epochs"] = 0
    config["next_goal_predictor"]["enabled"] = True
    config["next_goal_predictor"]["init_checkpoint"] = init_checkpoint
    config["next_goal_predictor"]["epochs"] = epochs
    config["next_goal_predictor"]["reuse_normalizer"] = True
    config["value_pretraining"]["enabled"] = False
    config["value_pretraining"]["epochs"] = 0
    config["wandb"]["job_type"] = "ngp_refresh_candidate"
    return config


def _relative_improvement(active_loss: float, candidate_loss: float) -> float:
    scale = max(abs(active_loss), 1.0e-8)
    return (active_loss - candidate_loss) / scale


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PPO in short chunks with a frozen active reward NGP, continuously fine-tune a "
            "candidate NGP on mixed online+anchor data, and promote the candidate when it improves "
            "recent validation without regressing anchor validation past the configured threshold."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--train-binary", type=Path, required=True, help="Path to pulsar_train")
    parser.add_argument("--offline-binary", type=Path, required=True, help="Path to pulsar_offline_train")
    parser.add_argument("--ppo-config", type=Path, required=True, help="Base PPO config")
    parser.add_argument("--offline-config", type=Path, required=True, help="Base offline config for NGP updates")
    parser.add_argument("--run-root", type=Path, required=True, help="Output root for the dynamic refresh run")
    parser.add_argument("--total-updates", type=int, required=True, help="Total PPO updates to run")
    parser.add_argument(
        "--ppo-updates-per-cycle",
        type=int,
        default=1,
        help="PPO updates per controller cycle. 1 means evaluate promotion after every PPO update.",
    )
    parser.add_argument(
        "--candidate-epochs-per-cycle",
        type=int,
        default=1,
        help="Additional NGP fine-tuning epochs to run on the candidate each cycle.",
    )
    parser.add_argument(
        "--old-data-fraction",
        type=float,
        default=0.30,
        help="Target fraction of anchor replay data in candidate NGP training. 0.30 means 70/30 new/old.",
    )
    parser.add_argument(
        "--online-buffer-cycles",
        type=int,
        default=0,
        help="How many recent online-export cycles to retain after the last promotion. 0 means all cycles since promotion.",
    )
    parser.add_argument(
        "--min-online-train-samples",
        type=int,
        default=32768,
        help="Minimum number of online training samples required before updating the candidate NGP.",
    )
    parser.add_argument(
        "--min-recent-loss-improvement",
        type=float,
        default=0.02,
        help="Required relative reduction in recent validation NGP loss for promotion.",
    )
    parser.add_argument(
        "--max-anchor-loss-regression",
        type=float,
        default=0.01,
        help="Maximum allowed relative increase in anchor validation NGP loss for promotion.",
    )
    parser.add_argument(
        "--promotion-cooldown-cycles",
        type=int,
        default=1,
        help="Minimum number of completed cycles between promotions.",
    )
    args = parser.parse_args()

    if args.total_updates <= 0:
        raise RuntimeError("--total-updates must be positive")
    if args.ppo_updates_per_cycle <= 0:
        raise RuntimeError("--ppo-updates-per-cycle must be positive")
    if args.candidate_epochs_per_cycle <= 0:
        raise RuntimeError("--candidate-epochs-per-cycle must be positive")

    repo_root = args.repo_root.resolve()
    train_binary = args.train_binary.resolve()
    offline_binary = args.offline_binary.resolve()
    ppo_base = _load_json(args.ppo_config.resolve())
    offline_base = _load_json(args.offline_config.resolve())
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    cycles_root = run_root / "cycles"
    cycles_root.mkdir(parents=True, exist_ok=True)

    active_policy_checkpoint = ppo_base["ppo"].get("init_checkpoint", "")
    active_ngp_checkpoint = ppo_base["reward"].get("ngp_checkpoint", "")
    if not active_ngp_checkpoint:
        raise RuntimeError("base PPO config must provide reward.ngp_checkpoint")

    anchor_train_manifest = Path(offline_base["offline_dataset"].get("train_manifest", "")).resolve()
    anchor_val_manifest = Path(
        offline_base["offline_dataset"].get("val_manifest", "") or offline_base["offline_dataset"].get("train_manifest", "")
    ).resolve()
    if not anchor_train_manifest.exists() or not anchor_val_manifest.exists():
        raise RuntimeError("offline config must provide existing anchor train/val manifests")

    promotion_index = 0
    cycles_since_promotion = 0
    online_train_history: list[Path] = []
    online_val_history: list[Path] = []
    candidate_checkpoint: str | None = None
    cycle_records: list[dict[str, Any]] = []
    completed_updates = 0

    while completed_updates < args.total_updates:
        cycle_index = len(cycle_records)
        cycle_dir = cycles_root / _phase_name(cycle_index)
        cycle_dir.mkdir(parents=True, exist_ok=True)
        ppo_run_dir = cycle_dir / "ppo"
        online_export_dir = cycle_dir / "online_ngp"
        ppo_config_path = cycle_dir / "ppo_config.json"

        cycle_updates = min(args.ppo_updates_per_cycle, args.total_updates - completed_updates)
        ppo_config = json.loads(json.dumps(ppo_base))
        ppo_config["ppo"]["init_checkpoint"] = active_policy_checkpoint
        ppo_config["reward"]["ngp_checkpoint"] = str(Path(active_ngp_checkpoint).resolve())
        ppo_config["reward"]["ngp_label"] = f"active_ngp_{promotion_index:03d}"
        ppo_config["reward"]["online_dataset"] = {
            "enabled": True,
            "output_dir": str(online_export_dir.resolve()),
            "shard_size": ppo_config["reward"].get("online_dataset", {}).get("shard_size", 65536),
            "train_fraction": ppo_config["reward"].get("online_dataset", {}).get("train_fraction", 0.9),
            "seed": ppo_config["reward"].get("online_dataset", {}).get("seed", cycle_index),
        }
        ppo_config["reward"]["refresh"] = {"enabled": False, "candidate_checkpoint": "", "check_interval_updates": 1}
        _write_json(ppo_config_path, ppo_config)

        _run([str(train_binary), str(ppo_config_path), str(ppo_run_dir), str(cycle_updates)], repo_root)
        completed_updates += cycle_updates
        cycles_since_promotion += 1
        active_policy_checkpoint = str((ppo_run_dir / "final").resolve())

        train_manifest = (online_export_dir / "train_manifest.json").resolve()
        val_manifest = (online_export_dir / "val_manifest.json").resolve()
        if not train_manifest.exists() or not val_manifest.exists():
            raise RuntimeError(f"missing exported online manifests in {online_export_dir}")
        online_train_history.append(train_manifest)
        online_val_history.append(val_manifest)

        if args.online_buffer_cycles > 0:
            buffered_train = online_train_history[-args.online_buffer_cycles :]
            buffered_val = online_val_history[-args.online_buffer_cycles :]
        else:
            buffered_train = list(online_train_history)
            buffered_val = list(online_val_history)

        _, _, _, _, online_train_samples = _load_manifest_bundle(buffered_train)
        _, _, _, _, online_val_samples = _load_manifest_bundle(buffered_val)

        cycle_record: dict[str, Any] = {
            "cycle_index": cycle_index,
            "completed_updates": completed_updates,
            "cycle_updates": cycle_updates,
            "ppo_run_dir": str(ppo_run_dir.resolve()),
            "online_export_dir": str(online_export_dir.resolve()),
            "active_policy_checkpoint": active_policy_checkpoint,
            "active_ngp_checkpoint_before_cycle": active_ngp_checkpoint,
            "promotion_index_before_cycle": promotion_index,
            "online_train_samples": online_train_samples,
            "online_val_samples": online_val_samples,
            "candidate_checkpoint_before_cycle": candidate_checkpoint,
        }

        if online_train_samples < args.min_online_train_samples or online_val_samples <= 0:
            cycle_record["candidate_update_skipped"] = True
            cycle_record["skip_reason"] = "insufficient_online_data"
            cycle_records.append(cycle_record)
            _write_json(run_root / "refresh_loop.json", {"cycles": cycle_records})
            continue

        mixed_dir = cycle_dir / "mixed_manifests"
        mixed_train_manifest = mixed_dir / "train_manifest.json"
        mixed_val_manifest = mixed_dir / "val_manifest.json"
        used_online_train_samples, used_anchor_train_samples = _build_mixed_train_manifest(
            mixed_train_manifest,
            [anchor_train_manifest],
            buffered_train,
            args.old_data_fraction,
            seed=cycle_index + promotion_index * 100003,
        )
        _merge_all_manifests([anchor_val_manifest, *buffered_val], mixed_val_manifest)
        cycle_record["mixed_train_manifest"] = str(mixed_train_manifest.resolve())
        cycle_record["mixed_val_manifest"] = str(mixed_val_manifest.resolve())
        cycle_record["used_online_train_samples"] = used_online_train_samples
        cycle_record["used_anchor_train_samples"] = used_anchor_train_samples

        candidate_base_checkpoint = candidate_checkpoint or active_ngp_checkpoint
        candidate_dir = cycle_dir / "candidate_ngp"
        candidate_config_path = cycle_dir / "candidate_config.json"
        candidate_config = _make_candidate_config(
            offline_base,
            candidate_base_checkpoint,
            mixed_train_manifest,
            mixed_val_manifest,
            args.candidate_epochs_per_cycle,
        )
        _write_json(candidate_config_path, candidate_config)
        _run([str(offline_binary), str(candidate_config_path), str(candidate_dir)], repo_root)
        candidate_checkpoint = str(candidate_dir.resolve())
        cycle_record["candidate_checkpoint_after_cycle"] = candidate_checkpoint

        recent_val_manifest = cycle_dir / "recent_val_manifest.json"
        _merge_all_manifests(buffered_val, recent_val_manifest)

        eval_specs = [
            ("active_anchor", active_ngp_checkpoint, anchor_val_manifest),
            ("active_recent", active_ngp_checkpoint, recent_val_manifest),
            ("candidate_anchor", candidate_checkpoint, anchor_val_manifest),
            ("candidate_recent", candidate_checkpoint, recent_val_manifest),
        ]
        eval_results: dict[str, dict[str, Any]] = {}
        for name, checkpoint_path, val_path in eval_specs:
            eval_dir = cycle_dir / name
            eval_config_path = cycle_dir / f"{name}_config.json"
            eval_config = _make_eval_config(offline_base, checkpoint_path, val_path)
            _write_json(eval_config_path, eval_config)
            _run([str(offline_binary), str(eval_config_path), str(eval_dir)], repo_root)
            eval_results[name] = _read_latest_metrics_line(eval_dir / "offline_metrics.jsonl")

        active_anchor_loss = float(eval_results["active_anchor"]["val_ngp_loss"])
        active_recent_loss = float(eval_results["active_recent"]["val_ngp_loss"])
        candidate_anchor_loss = float(eval_results["candidate_anchor"]["val_ngp_loss"])
        candidate_recent_loss = float(eval_results["candidate_recent"]["val_ngp_loss"])
        recent_improvement = _relative_improvement(active_recent_loss, candidate_recent_loss)
        anchor_regression = _relative_improvement(active_anchor_loss, candidate_anchor_loss) * -1.0

        promote = (
            cycles_since_promotion >= args.promotion_cooldown_cycles
            and recent_improvement >= args.min_recent_loss_improvement
            and anchor_regression <= args.max_anchor_loss_regression
        )

        cycle_record["candidate_update_skipped"] = False
        cycle_record["active_anchor_val"] = eval_results["active_anchor"]
        cycle_record["active_recent_val"] = eval_results["active_recent"]
        cycle_record["candidate_anchor_val"] = eval_results["candidate_anchor"]
        cycle_record["candidate_recent_val"] = eval_results["candidate_recent"]
        cycle_record["recent_loss_improvement"] = recent_improvement
        cycle_record["anchor_loss_regression"] = anchor_regression
        cycle_record["promoted"] = promote

        if promote:
            active_ngp_checkpoint = candidate_checkpoint
            promotion_index += 1
            cycles_since_promotion = 0
            candidate_checkpoint = None
            online_train_history.clear()
            online_val_history.clear()
            cycle_record["active_ngp_checkpoint_after_cycle"] = active_ngp_checkpoint
            cycle_record["promotion_index_after_cycle"] = promotion_index
        else:
            cycle_record["active_ngp_checkpoint_after_cycle"] = active_ngp_checkpoint
            cycle_record["promotion_index_after_cycle"] = promotion_index

        cycle_records.append(cycle_record)
        _write_json(run_root / "refresh_loop.json", {"cycles": cycle_records})

    latest_policy = run_root / "latest_policy"
    latest_ngp = run_root / "latest_ngp"
    if latest_policy.exists():
        shutil.rmtree(latest_policy)
    if latest_ngp.exists():
        shutil.rmtree(latest_ngp)
    shutil.copytree(Path(active_policy_checkpoint), latest_policy)
    shutil.copytree(Path(active_ngp_checkpoint), latest_ngp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
