#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: offline_binary_smoke.py <repo_root> <pulsar_offline_train>")

    repo_root = Path(sys.argv[1]).resolve()
    offline_binary = Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory(prefix="pulsar_offline_binary_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        data_dir.mkdir(parents=True)

        rows = 32
        obs = torch.randn(rows, 132)
        actions = torch.randint(90, (rows,), dtype=torch.long)
        action_probs = torch.nn.functional.one_hot(actions, 90).to(torch.float32)
        next_goal = torch.randint(3, (rows,), dtype=torch.long)
        weights = torch.ones(rows)
        episode_starts = torch.zeros(rows)
        terminated = torch.zeros(rows)
        truncated = torch.zeros(rows)
        episode_starts[0] = 1.0
        episode_starts[16] = 1.0
        terminated[15] = 1.0
        terminated[31] = 1.0

        torch.save(obs, data_dir / "obs.pt")
        torch.save(actions, data_dir / "actions.pt")
        torch.save(action_probs, data_dir / "action_probs.pt")
        torch.save(next_goal, data_dir / "next_goal.pt")
        torch.save(weights, data_dir / "weights.pt")
        torch.save(episode_starts, data_dir / "episode_starts.pt")
        torch.save(terminated, data_dir / "terminated.pt")
        torch.save(truncated, data_dir / "truncated.pt")

        (data_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "observation_dim": 132,
                    "action_dim": 90,
                    "next_goal_classes": 3,
                    "shards": [
                        {
                            "obs_path": "obs.pt",
                            "actions_path": "actions.pt",
                            "action_probs_path": "action_probs.pt",
                            "next_goal_path": "next_goal.pt",
                            "weights_path": "weights.pt",
                            "episode_starts_path": "episode_starts.pt",
                            "terminated_path": "terminated.pt",
                            "truncated_path": "truncated.pt",
                            "samples": rows,
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        config = json.loads((repo_root / "configs/2v2_offline.json").read_text(encoding="utf-8"))
        config["offline_dataset"]["train_manifest"] = str((data_dir / "manifest.json").resolve())
        config["offline_dataset"]["val_manifest"] = str((data_dir / "manifest.json").resolve())
        config["offline_dataset"]["batch_size"] = 8
        config["behavior_cloning"]["epochs"] = 1
        config["next_goal_predictor"]["epochs"] = 1
        config["value_pretraining"]["epochs"] = 1
        config["ppo"]["device"] = "cpu"
        config["wandb"]["enabled"] = False
        config_path = tmp_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [str(offline_binary), str(config_path), str(output_dir)],
            check=True,
            cwd=repo_root,
        )

        for rel in [
            "model.pt",
            "trunk_optimizer.pt",
            "policy_head_optimizer.pt",
            "value_head_optimizer.pt",
            "ngp_head_optimizer.pt",
            "offline_metrics.jsonl",
        ]:
            if not (output_dir / rel).exists():
                raise RuntimeError(f"missing offline artifact: {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
