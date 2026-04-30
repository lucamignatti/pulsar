#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from two_stage_smoke_common import run_offline_pretrain


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
        "action_embedding_dim": 8,
    }


def _small_evaluator() -> dict[str, int | list[int]]:
    return {
        "horizons": [1, 2, 3],
        "latent_dim": 8,
        "model_dim": 16,
        "layers": 1,
        "heads": 4,
        "feedforward_dim": 32,
    }


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: train_binary_smoke.py <repo_root> <pulsar_lfpo_train> <pulsar_lfpo_pretrain> "
            "<lfpo_base_config> <offline_base_config>"
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    pretrain_binary = Path(sys.argv[3]).resolve()
    lfpo_base_config_path = Path(sys.argv[4]).resolve()
    offline_base_config_path = Path(sys.argv[5]).resolve()

    with tempfile.TemporaryDirectory(prefix="pulsar_lfpo_train_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        model_overrides = _small_model()
        offline_output_dir = run_offline_pretrain(
            repo_root=repo_root,
            offline_binary=pretrain_binary,
            offline_base_config_path=offline_base_config_path,
            work_dir=tmp_dir,
            device="cpu",
            model_overrides=model_overrides,
        )

        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"
        config = json.loads(lfpo_base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["env"]["max_episode_ticks"] = 24
        config["model"].update(model_overrides)
        config["future_evaluator"].update(_small_evaluator())
        config["lfpo"]["num_envs"] = 2
        config["lfpo"]["rollout_length"] = 4
        config["lfpo"]["minibatch_size"] = 8
        config["lfpo"]["update_epochs"] = 1
        config["lfpo"]["checkpoint_interval"] = 1
        config["lfpo"]["sequence_length"] = 2
        config["lfpo"]["burn_in"] = 0
        config["lfpo"]["collection_workers"] = 0
        config["lfpo"]["device"] = "cpu"
        config["lfpo"]["init_checkpoint"] = str(offline_output_dir.resolve())
        config["lfpo"]["candidate_count"] = 4
        config["lfpo"]["evaluator_update_interval"] = 1
        config["lfpo"]["evaluator_target_update_interval"] = 1
        config["lfpo"]["online_window_capacity"] = 8
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
            "lfpo_forward_backward_seconds",
            "optimizer_step_seconds",
            "online_outcome_samples",
            "online_outcome_trajectories",
            "evaluator_target_update_index",
        }
        if missing := sorted(required_fields - metrics_lines[-1].keys()):
            raise RuntimeError(f"missing LFPO trainer metrics: {missing}")

        for rel in [
            "update_1/model.pt",
            "update_1/future_evaluator/model.pt",
            "update_1/future_evaluator/online_model.pt",
            "final/model.pt",
        ]:
            if not (checkpoint_dir / rel).exists():
                raise RuntimeError(f"missing checkpoint artifact: {rel}")

        resume_dir = tmp_dir / "resume_checkpoints"
        resume_config = json.loads(json.dumps(config))
        resume_config["lfpo"]["init_checkpoint"] = str((checkpoint_dir / "update_1").resolve())
        resume_config_path = tmp_dir / "resume_config.json"
        resume_config_path.write_text(json.dumps(resume_config, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [str(train_binary), str(resume_config_path), str(resume_dir), "1"],
            check=True,
            cwd=repo_root,
        )
        if not (resume_dir / "update_2" / "model.pt").exists():
            raise RuntimeError("resume run did not continue LFPO update numbering")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
