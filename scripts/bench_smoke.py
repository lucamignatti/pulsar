#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "2v2_appo.json").exists():
            return candidate
    raise FileNotFoundError(
        "could not locate repo root containing configs/2v2_appo.json"
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: bench_smoke.py <pulsar_bench>")

    bench_binary = Path(sys.argv[1]).resolve()
    repo_root = find_repo_root(bench_binary.parent)
    base_config = json.loads(
        (repo_root / "configs" / "2v2_appo.json").read_text(encoding="utf-8")
    )
    base_config["model"].update(
        {
            "encoder_dim": 64,
            "num_encoder_blocks": 1,
            "sequence_length": 4,
            "max_forward_samples": 256,
            "value_hidden_dim": 64,
        }
    )
    base_config["ppo"]["rollout_length"] = 8
    base_config["ppo"]["minibatch_size"] = 32
    base_config["ppo"]["num_envs"] = 2
    base_config["ppo"]["collection_shards"] = 1
    base_config["ppo"]["collection_workers"] = 0
    base_config["ppo"]["device"] = "cpu"
    base_config["wandb"]["enabled"] = False
    base_config["env"]["collision_meshes_path"] = str(
        (repo_root / "collision_meshes").resolve()
    )
    base_config["goal_critic"]["hidden_dim"] = 64
    base_config["goal_critic"]["contrastive_batch_size"] = 16
    with tempfile.TemporaryDirectory(prefix="pulsar_bench_smoke_") as tmp:
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps(base_config), encoding="utf-8")
        result = subprocess.run(
            [str(bench_binary), "1", str(config_path), "cpu"],
            check=True,
            capture_output=True,
            text=True,
        )
    metrics = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()

    required = {
        "collection_agent_steps_per_second",
        "ppo_update_agent_steps_per_second",
        "overall_agent_steps_per_second",
        "forward_backward_seconds",
        "optimizer_step_seconds",
        "es_updates",
        "es_seconds",
        "es_eval_seconds",
        "es_agent_steps_per_second",
        "es_effective_population",
        "es_virtual_population_waves",
        "es_eval_shards",
        "es_policy_update_norm",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise RuntimeError(f"benchmark output missing fields: {missing}")

    for key in required:
        if float(metrics[key]) < 0.0:
            raise RuntimeError(f"benchmark metric should be non-negative: {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
