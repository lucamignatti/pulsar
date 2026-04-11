#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream JSON metrics from Pulsar C++ loops into Weights & Biases.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--mode", default="online")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required for Pulsar W&B logging. Install it with `pip install wandb` or `pip install -e .[offline]`."
        ) from exc

    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        name=args.run_name,
        group=args.group or None,
        job_type=args.job_type,
        dir=args.wandb_dir,
        mode=args.mode,
        tags=args.tag or None,
        config=_load_config(args.config_path) if args.config_path else None,
        reinit=True,
    )
    if run is None:
        return 0

    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            wandb.log(payload)
    finally:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
