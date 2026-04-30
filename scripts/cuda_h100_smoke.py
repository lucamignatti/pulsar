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


def cuda_device_name() -> str:
    props = torch.cuda.get_device_properties(0)
    return str(getattr(props, "name", ""))


def is_h100() -> bool:
    props = torch.cuda.get_device_properties(0)
    name = cuda_device_name().lower()
    return "h100" in name or (int(getattr(props, "major", 0)) == 9 and int(getattr(props, "minor", 0)) == 0)


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: cuda_h100_smoke.py <repo_root> <pulsar_lfpo_train> <pulsar_lfpo_pretrain> "
            "<lfpo_base_config> <offline_base_config>"
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    pretrain_binary = Path(sys.argv[3]).resolve()
    lfpo_base_config_path = Path(sys.argv[4]).resolve()
    offline_base_config_path = Path(sys.argv[5]).resolve()

    if not torch.cuda.is_available():
        return skip("CUDA/H100 smoke requires an available CUDA device")
    if not getattr(torch.version, "cuda", None):
        return skip("CUDA/H100 smoke requires a CUDA-enabled PyTorch build")
    if not is_h100():
        return skip(f"CUDA/H100 smoke requires H100 or sm_90; found {cuda_device_name()}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tmp_dir = Path(tempfile.mkdtemp(prefix="pulsar_cuda_h100_smoke_"))
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
            "action_embedding_dim": 16,
        }
        offline_output_dir = run_offline_pretrain(
            repo_root=repo_root,
            offline_binary=pretrain_binary,
            offline_base_config_path=offline_base_config_path,
            work_dir=tmp_dir,
            device="cuda:0",
            model_overrides=model_overrides,
        )
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(lfpo_base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["model"].update(model_overrides)
        config["future_evaluator"].update(
            {
                "horizons": [1, 2, 3],
                "latent_dim": 8,
                "model_dim": 16,
                "layers": 1,
                "heads": 4,
                "feedforward_dim": 32,
            }
        )
        config["lfpo"]["num_envs"] = 2
        config["lfpo"]["rollout_length"] = 4
        config["lfpo"]["minibatch_size"] = 8
        config["lfpo"]["update_epochs"] = 1
        config["lfpo"]["checkpoint_interval"] = 1
        config["lfpo"]["sequence_length"] = 2
        config["lfpo"]["burn_in"] = 0
        config["lfpo"]["collection_workers"] = 0
        config["lfpo"]["device"] = "cuda:0"
        config["lfpo"]["init_checkpoint"] = str(offline_output_dir.resolve())
        config["lfpo"]["candidate_count"] = 4
        config["lfpo"]["evaluator_update_interval"] = 1
        config["lfpo"]["evaluator_target_update_interval"] = 1
        config["wandb"]["enabled"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        command = [str(train_binary), str(config_path), str(checkpoint_dir), "1"]
        print(f"CUDA/H100 smoke temp dir: {tmp_dir}")
        print("CUDA/H100 smoke command:", " ".join(command))
        try:
            subprocess.run(
                command,
                check=True,
                cwd=repo_root,
            )
        except subprocess.CalledProcessError:
            print(f"CUDA/H100 smoke failed; preserving temp dir: {tmp_dir}", file=sys.stderr)
            raise

        metrics_lines = [
            json.loads(line)
            for line in (checkpoint_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not metrics_lines:
            raise RuntimeError("CUDA/H100 smoke did not emit metrics")
        for field in ["policy_loss", "latent_loss", "entropy"]:
            value = float(metrics_lines[-1][field])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite metric in CUDA/H100 smoke: {field}={value}")
        if not (checkpoint_dir / "final" / "model.pt").exists():
            raise RuntimeError("CUDA/H100 smoke did not write final checkpoint")
        keep_tmp_dir = False
    finally:
        if keep_tmp_dir:
            print(f"CUDA/H100 smoke preserved temp dir: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
