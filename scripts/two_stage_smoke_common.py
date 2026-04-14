from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def write_synthetic_offline_dataset(
    data_dir: Path,
    rows: int = 32,
    observation_dim: int = 132,
    action_dim: int = 90,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)

    obs = torch.randn(rows, observation_dim)
    actions = torch.randint(action_dim, (rows,), dtype=torch.long)
    action_probs = torch.nn.functional.one_hot(actions, action_dim).to(torch.float32)
    next_goal = torch.randint(3, (rows,), dtype=torch.long)
    weights = torch.ones(rows)
    episode_starts = torch.zeros(rows)
    terminated = torch.zeros(rows)
    truncated = torch.zeros(rows)
    episode_starts[0] = 1.0
    if rows > 16:
        episode_starts[rows // 2] = 1.0
        terminated[(rows // 2) - 1] = 1.0
    terminated[rows - 1] = 1.0

    torch.save(obs, data_dir / "obs.pt")
    torch.save(actions, data_dir / "actions.pt")
    torch.save(action_probs, data_dir / "action_probs.pt")
    torch.save(next_goal, data_dir / "next_goal.pt")
    torch.save(weights, data_dir / "weights.pt")
    torch.save(episode_starts, data_dir / "episode_starts.pt")
    torch.save(terminated, data_dir / "terminated.pt")
    torch.save(truncated, data_dir / "truncated.pt")

    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "observation_dim": observation_dim,
                "action_dim": action_dim,
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
    return manifest_path


def run_offline_pretrain(
    repo_root: Path,
    offline_binary: Path,
    offline_base_config_path: Path,
    work_dir: Path,
    device: str,
    model_overrides: dict[str, Any] | None = None,
) -> Path:
    data_dir = work_dir / "offline_data"
    output_dir = work_dir / "offline_output"
    config_path = work_dir / "offline_config.json"
    manifest_path = write_synthetic_offline_dataset(data_dir)

    config: dict[str, Any] = json.loads(offline_base_config_path.read_text(encoding="utf-8"))
    config["offline_dataset"]["train_manifest"] = str(manifest_path.resolve())
    config["offline_dataset"]["val_manifest"] = str(manifest_path.resolve())
    config["offline_dataset"]["batch_size"] = 8
    config["behavior_cloning"]["epochs"] = 1
    config["next_goal_predictor"]["epochs"] = 1
    config["value_pretraining"]["epochs"] = 1
    config["ppo"]["device"] = device
    config["wandb"]["enabled"] = False
    if model_overrides:
        config["model"].update(model_overrides)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [str(offline_binary), str(config_path), str(output_dir)],
        check=True,
        cwd=repo_root,
    )
    return output_dir
