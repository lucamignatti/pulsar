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
            "usage: e2e_smoke.py <repo_root> <pulsar_lfpo_train> <pulsar_lfpo_pretrain> "
            "<lfpo_base_config> <offline_base_config>"
        )
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "rlgym is not currently compatible with Python 3.14 in this test path. "
            "Recreate the virtualenv with Python 3.12 or 3.13."
        )

    repo_root = Path(sys.argv[1]).resolve()
    train_binary = Path(sys.argv[2]).resolve()
    offline_binary = Path(sys.argv[3]).resolve()
    lfpo_base_config_path = Path(sys.argv[4]).resolve()
    offline_base_config_path = Path(sys.argv[5]).resolve()

    sys.path.insert(0, str(repo_root / "python"))

    from pulsar_viz.api import ensure_checkpoint_config_matches, load_config, load_latent_future_actor, make_eval_env

    with tempfile.TemporaryDirectory(prefix="pulsar_e2e_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        model_overrides = {
            "use_layer_norm": False,
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
            offline_binary=offline_binary,
            offline_base_config_path=offline_base_config_path,
            work_dir=tmp_dir,
            device="cpu",
            model_overrides=model_overrides,
        )
        checkpoint_dir = tmp_dir / "checkpoints"
        config_path = tmp_dir / "config.json"

        config = json.loads(lfpo_base_config_path.read_text(encoding="utf-8"))
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
        config["lfpo"]["device"] = "cpu"
        config["lfpo"]["init_checkpoint"] = str(offline_output_dir.resolve())
        config["lfpo"]["candidate_count"] = 4
        config["lfpo"]["evaluator_update_interval"] = 1
        config["lfpo"]["evaluator_target_update_interval"] = 1
        config["env"]["collision_meshes_path"] = str(repo_root / "collision_meshes")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        subprocess.run(
            [str(train_binary), str(config_path), str(checkpoint_dir), "1"],
            check=True,
            cwd=repo_root,
        )

        update_dir = checkpoint_dir / "update_1"
        ensure_checkpoint_config_matches(load_config(update_dir / "config.json"), update_dir)
        model = load_latent_future_actor(update_dir, "cpu")
        bundle = make_eval_env(load_config(update_dir / "config.json"))
        observations = bundle.env.reset()
        first_obs = next(iter(observations.values()))
        logits = model.forward(first_obs.tolist())
        if len(logits) != config["model"]["action_dim"]:
            raise RuntimeError("LFPO actor output width does not match action_dim.")
        bundle.env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
