#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_shard_paths(manifest_path: Path, shard: dict[str, Any]) -> dict[str, Any]:
    base = manifest_path.parent
    result = dict(shard)
    for key in [
        "obs_path",
        "actions_path",
        "action_probs_path",
        "outcome_path",
        "outcome_known_path",
        "weights_path",
        "episode_starts_path",
        "terminated_path",
        "truncated_path",
    ]:
        value = result.get(key)
        if value:
            result[key] = str((base / value).resolve())
    return result


def _merge(paths: list[Path], output_path: Path) -> None:
    if not paths:
        raise ValueError("at least one manifest must be provided")

    merged_shards: list[dict[str, Any]] = []
    observation_dim: int | None = None
    action_dim: int | None = None
    outcome_classes: int | None = None

    for path in paths:
        manifest = _load_manifest(path)
        current_obs_dim = int(manifest["observation_dim"])
        current_action_dim = int(manifest["action_dim"])
        current_outcome_classes = int(manifest.get("outcome_classes", 3))

        if observation_dim is None:
            observation_dim = current_obs_dim
            action_dim = current_action_dim
            outcome_classes = current_outcome_classes
        else:
            if current_obs_dim != observation_dim:
                raise RuntimeError(f"observation_dim mismatch in {path}")
            if current_action_dim != action_dim:
                raise RuntimeError(f"action_dim mismatch in {path}")
            if current_outcome_classes != outcome_classes:
                raise RuntimeError(f"outcome_classes mismatch in {path}")

        for shard in manifest["shards"]:
            merged_shards.append(_normalize_shard_paths(path, shard))

    output = {
        "schema_version": 4,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "outcome_classes": outcome_classes,
        "shards": merged_shards,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple Pulsar offline manifests into one manifest with absolute shard paths."
    )
    parser.add_argument("output", type=Path, help="Path to the merged manifest to write")
    parser.add_argument("manifests", type=Path, nargs="+", help="Input manifest paths")
    args = parser.parse_args()

    _merge(args.manifests, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
