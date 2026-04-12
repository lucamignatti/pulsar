#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def skip(message: str) -> int:
    print(f"SKIP: {message}")
    return 77


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: rocm_bf16_smoke.py <repo_root> <pulsar_train> <base_config>")

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    base_config_path = Path(sys.argv[3]).resolve()

    if not torch.cuda.is_available():
        return skip("ROCm smoke requires an available CUDA/ROCm device")
    if not getattr(torch.version, "hip", None):
        return skip("ROCm smoke requires a HIP-enabled PyTorch build")
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        return skip("current ROCm GPU does not report BF16 support")

    with tempfile.TemporaryDirectory(prefix="pulsar_rocm_smoke_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["model"]["hidden_sizes"] = [64, 64]
        config["model"]["encoder_dim"] = 64
        config["model"]["workspace_dim"] = 64
        config["model"]["stm_slots"] = 8
        config["model"]["stm_key_dim"] = 16
        config["model"]["stm_value_dim"] = 16
        config["model"]["ltm_slots"] = 8
        config["model"]["ltm_dim"] = 16
        config["model"]["controller_dim"] = 64
        config["ppo"]["num_envs"] = 2
        config["ppo"]["rollout_length"] = 4
        config["ppo"]["minibatch_size"] = 8
        config["ppo"]["epochs"] = 1
        config["ppo"]["checkpoint_interval"] = 1
        config["ppo"]["sequence_length"] = 2
        config["ppo"]["burn_in"] = 1
        config["ppo"]["collection_workers"] = 0
        config["ppo"]["device"] = "cuda"
        config["ppo"]["precision"] = {"mode": "amp_bf16"}
        config["ppo"].setdefault("self_play", {})
        config["ppo"]["self_play"]["enabled"] = False
        config["reward"]["mode"] = "shaped"
        config["reward"]["ngp_checkpoint"] = ""
        config["reward"]["shaped_scale"] = 1.0
        config["reward"]["ngp_scale"] = 0.0
        config["wandb"]["enabled"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [str(train_binary), str(config_path), str(checkpoint_dir), "1"],
            check=True,
            cwd=repo_root,
        )

        metrics_lines = [
            json.loads(line)
            for line in (checkpoint_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not metrics_lines:
            raise RuntimeError("ROCm smoke did not emit metrics")
        for field in ["policy_loss", "value_loss", "entropy"]:
            value = float(metrics_lines[-1][field])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite metric in ROCm smoke: {field}={value}")
        if not (checkpoint_dir / "final" / "model.pt").exists():
            raise RuntimeError("ROCm smoke did not write final checkpoint")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
