#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from two_stage_smoke_common import run_offline_pretrain


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: train_binary_smoke.py <repo_root> <pulsar_train> <pulsar_offline_train> "
            "<ppo_base_config> <offline_base_config>"
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    offline_binary = Path(sys.argv[3]).resolve()
    ppo_base_config_path = Path(sys.argv[4]).resolve()
    offline_base_config_path = Path(sys.argv[5]).resolve()

    with tempfile.TemporaryDirectory(prefix="pulsar_train_smoke_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
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
            device="cpu",
            model_overrides=model_overrides,
        )
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(ppo_base_config_path.read_text(encoding="utf-8"))
        config["env"]["collision_meshes_path"] = str((repo_root / "collision_meshes").resolve())
        config["env"]["max_episode_ticks"] = 16
        config["model"].update(model_overrides)
        config["ppo"]["num_envs"] = 4
        config["ppo"]["rollout_length"] = 4
        config["ppo"]["minibatch_size"] = 16
        config["ppo"]["epochs"] = 1
        config["ppo"]["checkpoint_interval"] = 1
        config["ppo"]["sequence_length"] = 2
        config["ppo"]["burn_in"] = 1
        config["ppo"]["collection_workers"] = 0
        config["ppo"]["device"] = "cpu"
        config["ppo"]["init_checkpoint"] = str(offline_output_dir.resolve())
        config["reward"]["ngp_checkpoint"] = str(offline_output_dir.resolve())
        config["reward"]["ngp_label"] = "bootstrap"
        config["reward"]["ngp_scale"] = 1.0
        promoted_ngp_dir = tmp_dir / "promoted_next_goal"
        promoted_ngp_dir.mkdir(parents=True)
        for child in offline_output_dir.iterdir():
            if child.is_file():
                target = promoted_ngp_dir / child.name
                target.write_bytes(child.read_bytes())
        config["reward"]["online_dataset"] = {
            "enabled": True,
            "output_dir": str((tmp_dir / "online_ngp_data").resolve()),
            "shard_size": 8,
            "train_fraction": 1.0,
            "seed": 7,
        }
        config["reward"]["refresh"] = {
            "enabled": True,
            "candidate_checkpoint": str(promoted_ngp_dir.resolve()),
            "check_interval_updates": 1,
        }
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
            "ppo_forward_backward_seconds",
            "optimizer_step_seconds",
            "ngp_promotion_index",
            "ngp_label",
            "ngp_checkpoint",
            "ngp_online_samples_written",
        }
        if missing := sorted(required_fields - metrics_lines[-1].keys()):
            raise RuntimeError(f"missing trainer metrics: {missing}")
        if metrics_lines[-1]["ngp_label"] != "bootstrap":
            raise RuntimeError("trainer metrics should report the active bootstrap NGP label")
        if metrics_lines[-1]["ngp_online_samples_written"] <= 0:
            raise RuntimeError("online NGP dataset export did not record any samples")

        promotion_lines = [
            json.loads(line)
            for line in (checkpoint_dir / "ngp_promotions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not promotion_lines:
            raise RuntimeError("ngp_promotions.jsonl was not written")

        for rel in ["update_1/model.pt", "best/model.pt", "final/model.pt"]:
            if not (checkpoint_dir / rel).exists():
                raise RuntimeError(f"missing checkpoint artifact: {rel}")

        for rel in ["train_manifest.json", "val_manifest.json"]:
            if not (tmp_dir / "online_ngp_data" / rel).exists():
                raise RuntimeError(f"missing online NGP dataset manifest: {rel}")

        integrated_checkpoint_dir = tmp_dir / "integrated_checkpoints"
        integrated_config_path = tmp_dir / "integrated_config.json"
        integrated_online_dir = tmp_dir / "online_ngp_integrated"
        integrated_config = json.loads(json.dumps(config))
        integrated_config["reward"]["online_dataset"] = {
            "enabled": True,
            "output_dir": str(integrated_online_dir.resolve()),
            "shard_size": 8,
            "train_fraction": 1.0,
            "seed": 17,
        }
        anchor_manifest = (tmp_dir / "offline_data" / "manifest.json").resolve()
        integrated_config["reward"]["refresh"] = {
            "enabled": True,
            "candidate_checkpoint": "",
            "check_interval_updates": 1,
            "train_candidate_in_process": True,
            "online_train_fraction": 0.7,
            "anchor_train_manifest": str(anchor_manifest),
            "anchor_val_manifest": str(anchor_manifest),
            "candidate_epochs": 1,
            "old_data_fraction": 0.3,
            "min_online_train_samples": 8,
            "min_recent_loss_improvement": -1.0,
            "max_anchor_loss_regression": 1.0,
            "promotion_cooldown_updates": 1,
            "train_trunk": False,
        }
        integrated_config_path.write_text(json.dumps(integrated_config, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [str(train_binary), str(integrated_config_path), str(integrated_checkpoint_dir), "2"],
            check=True,
            cwd=repo_root,
        )

        if not (integrated_checkpoint_dir / "ngp_promotions.jsonl").exists():
            raise RuntimeError("integrated refresh did not write ngp_promotions.jsonl")
        if not any((integrated_checkpoint_dir / "ngp_versions").glob("promotion_*/model.pt")):
            raise RuntimeError("integrated refresh did not save a promoted NGP checkpoint")

        resumed_checkpoint_dir = tmp_dir / "integrated_resume_checkpoints"
        resumed_stage1_config = json.loads(json.dumps(integrated_config))
        resumed_stage1_config["ppo"]["init_checkpoint"] = str(offline_output_dir.resolve())
        resumed_stage1_config_path = tmp_dir / "integrated_resume_stage1.json"
        resumed_stage1_config_path.write_text(
            json.dumps(resumed_stage1_config, indent=2) + "\n",
            encoding="utf-8",
        )

        subprocess.run(
            [str(train_binary), str(resumed_stage1_config_path), str(resumed_checkpoint_dir), "1"],
            check=True,
            cwd=repo_root,
        )

        update_1_metadata = json.loads(
            (resumed_checkpoint_dir / "update_1" / "metadata.json").read_text(encoding="utf-8")
        )
        if update_1_metadata["reward_ngp_promotion_index"] < 1:
            raise RuntimeError("integrated update_1 checkpoint did not persist the promoted NGP for resume")

        resumed_stage2_config = json.loads(json.dumps(integrated_config))
        resumed_stage2_config["ppo"]["init_checkpoint"] = str((resumed_checkpoint_dir / "update_1").resolve())
        resumed_stage2_config_path = tmp_dir / "integrated_resume_stage2.json"
        resumed_stage2_config_path.write_text(
            json.dumps(resumed_stage2_config, indent=2) + "\n",
            encoding="utf-8",
        )

        subprocess.run(
            [str(train_binary), str(resumed_stage2_config_path), str(resumed_checkpoint_dir), "1"],
            check=True,
            cwd=repo_root,
        )

        resumed_metrics = [
            json.loads(line)
            for line in (resumed_checkpoint_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [line["update"] for line in resumed_metrics] != [1, 2]:
            raise RuntimeError("integrated resume did not preserve PPO update numbering")

        resumed_promotions = [
            json.loads(line)
            for line in (resumed_checkpoint_dir / "ngp_promotions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [line["update"] for line in resumed_promotions] != [1, 2]:
            raise RuntimeError("integrated resume did not preserve NGP promotion history")
        if len({line["new_ngp_checkpoint"] for line in resumed_promotions}) != 2:
            raise RuntimeError("integrated resume reused an existing promoted NGP checkpoint path")

        resumed_final_metadata = json.loads(
            (resumed_checkpoint_dir / "final" / "metadata.json").read_text(encoding="utf-8")
        )
        if resumed_final_metadata["update_index"] != 2:
            raise RuntimeError("integrated final checkpoint did not preserve resumed update index")
        if resumed_final_metadata["reward_ngp_promotion_index"] < 2:
            raise RuntimeError("integrated resume did not advance the promoted NGP index after resuming")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
