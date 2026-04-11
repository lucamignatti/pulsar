from __future__ import annotations

import argparse

from .api import (
    ensure_checkpoint_config_matches,
    load_config,
    load_shared_model,
    make_eval_env,
    run_viz_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Pulsar visualization episode")
    parser.add_argument("--config", required=True, help="Path to the shared experiment JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device string")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_checkpoint_config_matches(config, args.checkpoint)
    model = load_shared_model(args.checkpoint, args.device)
    bundle = make_eval_env(config)
    run_viz_episode(model, bundle.env, bundle.renderer, config, args.seed)


if __name__ == "__main__":
    main()
