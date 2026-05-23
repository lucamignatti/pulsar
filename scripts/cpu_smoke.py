#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: cpu_smoke.py <repo_root> <pulsar_appo_train> <appo_base_config>"
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    appo_base_config_path = Path(sys.argv[3]).resolve()

    tmp_dir = Path(tempfile.mkdtemp(prefix="pulsar_cpu_smoke_"))
    keep_tmp_dir = True
    try:
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(appo_base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["model"].update({
            "encoder_dim": 64,
            "num_encoder_blocks": 1,
            "value_hidden_dim": 64,
        })
        config["ppo"]["num_envs"] = 2
        config["ppo"]["collection_shards"] = 1
        config["ppo"]["rollout_length"] = 4
        config["ppo"]["minibatch_size"] = 8
        config["ppo"]["update_epochs"] = 1
        config["ppo"]["checkpoint_interval"] = 1
        config["ppo"]["collection_workers"] = 0
        config["ppo"]["device"] = "cpu"
        config["wandb"]["enabled"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        command = [str(train_binary), str(config_path), str(checkpoint_dir), "2"]
        print(f"CPU smoke temp dir: {tmp_dir}")
        print("CPU smoke command:", " ".join(command))
        try:
            subprocess.run(
                command,
                check=True,
                cwd=repo_root,
            )
        except subprocess.CalledProcessError:
            print(f"CPU smoke failed; preserving temp dir: {tmp_dir}", file=sys.stderr)
            raise

        metrics_lines = [
            json.loads(line)
            for line in (checkpoint_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not metrics_lines:
            raise RuntimeError("CPU smoke did not emit metrics")
        for field in ["policy_loss", "value_loss", "entropy"]:
            value = float(metrics_lines[-1][field])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite metric in CPU smoke: {field}={value}")
        print("CPU smoke test successfully completed 2 updates and verified loss metrics!")
        keep_tmp_dir = False
    finally:
        if keep_tmp_dir:
            print(f"CPU smoke preserved temp dir: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
