#!/usr/bin/env python3
"""Read JSON-lines from stdin and stream metrics to W&B.

Each line is a JSON object. "_step" sets the W&B step; all other keys are
logged as metrics. Metrics are organized into sections by prefix:
  train/*     — PPO losses and win rate
  curriculum/* — reachability grid progress
  perf/*      — throughput

Usage (spawned by pulsar_train):
    python3 scripts/wandb_stream.py <project> [run_name] [entity]
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    project  = sys.argv[1] if len(sys.argv) > 1 else "pulsar"
    run_name = sys.argv[2] if len(sys.argv) > 2 else None
    entity   = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        import wandb
    except ImportError:
        sys.stderr.write("[wandb_stream] wandb not installed — logging disabled\n")
        for _ in sys.stdin:  # drain so caller doesn't get SIGPIPE
            pass
        return

    wandb.init(project=project, entity=entity or None,
               name=run_name or None, resume="allow")

    wandb.define_metric("_step")
    for prefix in ("train", "curriculum", "perf"):
        wandb.define_metric(f"{prefix}/*", step_metric="_step")

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            step = payload.pop("_step", None)
            wandb.log(payload, step=step)
        except Exception as exc:
            sys.stderr.write(f"[wandb_stream] {exc}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
