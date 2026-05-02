#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from two_stage_smoke_common import run_bc_pretrain


def _small_model() -> dict[str, int | bool]:
    return {
        "use_layer_norm": False,
        "encoder_dim": 32,
        "workspace_dim": 32,
        "stm_slots": 4,
        "stm_key_dim": 8,
        "stm_value_dim": 8,
        "ltm_slots": 4,
        "ltm_dim": 8,
        "controller_dim": 32,
    }


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: train_binary_smoke.py <repo_root> <pulsar_appo_train> <pulsar_bc_pretrain> "
            "<appo_base_config> <bc_base_config>"
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    pretrain_binary = Path(sys.argv[3]).resolve()
    appo_base_config_path = Path(sys.argv[4]).resolve()
    bc_base_config_path = Path(sys.argv[5]).resolve()

    with tempfile.TemporaryDirectory(prefix="pulsar_appo_train_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        model_overrides = _small_model()
        bc_output_dir = run_bc_pretrain(
            repo_root=repo_root,
            pretrain_binary=pretrain_binary,
            bc_base_config_path=bc_base_config_path,
            work_dir=tmp_dir,
            device="cpu",
            model_overrides=model_overrides,
        )

        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"
        config = json.loads(appo_base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["env"]["max_episode_ticks"] = 24
        config["model"].update(model_overrides)
        config["model"]["value_hidden_dim"] = 64
        config["model"]["value_num_atoms"] = 51
        config["model"]["value_v_min"] = -10.0
        config["model"]["value_v_max"] = 10.0
        config["ppo"]["num_envs"] = 2
        config["ppo"]["rollout_length"] = 4
        config["ppo"]["minibatch_size"] = 8
        config["ppo"]["update_epochs"] = 1
        config["ppo"]["checkpoint_interval"] = 1
        config["ppo"]["sequence_length"] = 2
        config["ppo"]["burn_in"] = 0
        config["ppo"]["collection_workers"] = 0
        config["ppo"]["device"] = "cpu"
        config["ppo"]["init_checkpoint"] = str(bc_output_dir.resolve())
        config["wandb"]["enabled"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [str(train_binary), str(config_path), str(checkpoint_dir), "1"],
            check=True,
            cwd=repo_root,
        )

        metrics_path = checkpoint_dir / "metrics.jsonl"
        metrics_lines = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not metrics_lines:
            raise RuntimeError("metrics.jsonl was not written")
        required_fields = {
            "collection_agent_steps_per_second",
            "update_agent_steps_per_second",
            "overall_agent_steps_per_second",
            "obs_build_seconds",
            "mask_build_seconds",
            "policy_forward_seconds",
            "forward_backward_seconds",
            "optimizer_step_seconds",
        }
        if missing := sorted(required_fields - metrics_lines[-1].keys()):
            raise RuntimeError(f"missing APPO trainer metrics: {missing}")

        for rel in [
            "update_1/model.pt",
            "update_1/config.json",
            "final/model.pt",
        ]:
            if not (checkpoint_dir / rel).exists():
                raise RuntimeError(f"missing checkpoint artifact: {rel}")

        resume_dir = tmp_dir / "resume_checkpoints"
        resume_config = json.loads(json.dumps(config))
        resume_config["ppo"]["init_checkpoint"] = str((checkpoint_dir / "update_1").resolve())
        resume_config_path = tmp_dir / "resume_config.json"
        resume_config_path.write_text(json.dumps(resume_config, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [str(train_binary), str(resume_config_path), str(resume_dir), "1"],
            check=True,
            cwd=repo_root,
        )
        if not (resume_dir / "update_2" / "model.pt").exists():
            raise RuntimeError("resume run did not continue APPO update numbering")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
