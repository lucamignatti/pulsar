#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from two_stage_smoke_common import run_offline_pretrain


def skip(message: str) -> int:
    print(f"SKIP: {message}")
    return 77


def rocm_arch_name() -> str:
    props = torch.cuda.get_device_properties(0)
    for attr in ("gcnArchName", "gcn_arch_name"):
        value = getattr(props, attr, "")
        if value:
            return str(value).split(":", 1)[0]
    name = str(getattr(props, "name", ""))
    if "gfx" in name:
        suffix = name.split("gfx", 1)[1].split()[0].strip()
        return f"gfx{suffix}"
    return ""


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: rocm_smoke.py <repo_root> <pulsar_train> <pulsar_offline_train> "
            "<ppo_base_config> <offline_base_config>"
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    offline_binary = Path(sys.argv[3]).resolve()
    ppo_base_config_path = Path(sys.argv[4]).resolve()
    offline_base_config_path = Path(sys.argv[5]).resolve()

    if not torch.cuda.is_available():
        return skip("ROCm smoke requires an available CUDA/ROCm device")
    if not getattr(torch.version, "hip", None):
        return skip("ROCm smoke requires a HIP-enabled PyTorch build")
    arch = rocm_arch_name()
    if not arch:
        return skip("ROCm smoke could not determine the current GPU architecture")

    tmp_dir = Path(tempfile.mkdtemp(prefix="pulsar_rocm_smoke_"))
    keep_tmp_dir = True
    try:
        model_overrides = {
            "encoder_dim": 64,
            "workspace_dim": 64,
            "stm_slots": 8,
            "stm_key_dim": 16,
            "stm_value_dim": 16,
            "ltm_slots": 8,
            "ltm_dim": 16,
            "controller_dim": 64,
        }
        offline_output_dir = run_offline_pretrain(
            repo_root=repo_root,
            offline_binary=offline_binary,
            offline_base_config_path=offline_base_config_path,
            work_dir=tmp_dir,
            device="cuda:0",
            model_overrides=model_overrides,
        )
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(ppo_base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["model"].update(model_overrides)
        config["ppo"]["num_envs"] = 2
        config["ppo"]["rollout_length"] = 4
        config["ppo"]["minibatch_size"] = 8
        config["ppo"]["epochs"] = 1
        config["ppo"]["checkpoint_interval"] = 1
        config["ppo"]["sequence_length"] = 2
        config["ppo"]["burn_in"] = 1
        config["ppo"]["collection_workers"] = 0
        config["ppo"]["device"] = "cuda:0"
        config["ppo"]["init_checkpoint"] = str(offline_output_dir.resolve())
        config["reward"]["ngp_checkpoint"] = str(offline_output_dir.resolve())
        config["reward"]["ngp_scale"] = 1.0
        config["wandb"]["enabled"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        command = [str(train_binary), str(config_path), str(checkpoint_dir), "1"]
        print(f"ROCm smoke temp dir: {tmp_dir}")
        print("ROCm smoke command:", " ".join(command))
        try:
            subprocess.run(
                command,
                check=True,
                cwd=repo_root,
            )
        except subprocess.CalledProcessError:
            print(f"ROCm smoke failed; preserving temp dir: {tmp_dir}", file=sys.stderr)
            raise

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
        keep_tmp_dir = False
    finally:
        if keep_tmp_dir:
            print(f"ROCm smoke preserved temp dir: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
