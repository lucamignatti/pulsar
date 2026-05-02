#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: bench_smoke.py <pulsar_bench>")

    bench_binary = Path(sys.argv[1]).resolve()
    result = subprocess.run(
        [str(bench_binary), "1", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()

    required = {
        "collection_agent_steps_per_second",
        "ppo_update_agent_steps_per_second",
        "offline_pretrain_samples_per_second",
        "offline_pretrain_epoch_seconds",
        "policy_forward_seconds",
        "ppo_forward_backward_seconds",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise RuntimeError(f"benchmark output missing fields: {missing}")

    for key in required:
        if float(metrics[key]) < 0.0:
            raise RuntimeError(f"benchmark metric should be non-negative: {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
