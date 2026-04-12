#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: e2e_smoke.py <repo_root> <pulsar_train> <base_config>")
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "rlgym is not currently compatible with Python 3.14 in this test path. "
            "Recreate the virtualenv with Python 3.12 or 3.13."
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    base_config_path = Path(sys.argv[3]).resolve()

    sys.path.insert(0, str(repo_root / "python"))

    from pulsar_viz.api import ensure_checkpoint_config_matches, load_config, load_shared_model, make_eval_env

    with tempfile.TemporaryDirectory(prefix="pulsar_e2e_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(base_config_path.read_text(encoding="utf-8"))
        config["model"]["hidden_sizes"] = [64, 64]
        config["model"]["use_layer_norm"] = False
        config["ppo"]["num_envs"] = 2
        config["ppo"]["rollout_length"] = 4
        config["ppo"]["minibatch_size"] = 8
        config["ppo"]["epochs"] = 1
        config["ppo"]["checkpoint_interval"] = 1
        config["ppo"]["device"] = "cpu"
        config["ppo"]["precision"] = {"mode": "fp32"}
        config["reward"]["mode"] = "shaped"
        config["reward"]["ngp_checkpoint"] = ""
        config["reward"]["shaped_scale"] = 1.0
        config["reward"]["ngp_scale"] = 0.0
        config["env"]["collision_meshes_path"] = str(repo_root / "collision_meshes")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [str(train_binary), str(config_path), str(checkpoint_dir), "1"],
            check=True,
            cwd=repo_root,
        )

        update_dir = checkpoint_dir / "update_1"
        ensure_checkpoint_config_matches(load_config(update_dir / "config.json"), update_dir)
        model = load_shared_model(update_dir, "cpu")
        bundle = make_eval_env(load_config(update_dir / "config.json"))
        observations = bundle.env.reset()
        first_obs = next(iter(observations.values()))
        logits = model.forward(first_obs.tolist())
        if len(logits) != config["model"]["action_dim"]:
            raise RuntimeError("Shared model output width does not match action_dim.")
        bundle.env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
