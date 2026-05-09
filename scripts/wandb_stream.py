#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB — warn below this threshold


def _check_disk_space(dir_path: str) -> None:
    """Verify at least _MIN_FREE_BYTES are available on the filesystem containing dir_path.

    Exits gracefully with a clear message when disk is too full, avoiding the cascade of
    nested OSErrors that happens when logging / wandb operations fail mid-flight.
    """
    usage = shutil.disk_usage(os.path.dirname(os.path.abspath(dir_path)) or dir_path)
    free_gb = usage.free / (1024**3)
    if usage.free < _MIN_FREE_BYTES:
        print(
            f"FATAL: Disk is full on {dir_path}. "
            f"Free: {free_gb:.1f} GiB, required: {_MIN_FREE_BYTES / (1024**3):.0f} GiB. "
            f"Free up space and retry.",
            file=sys.stderr,
        )
        sys.exit(1)


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

    _check_disk_space(args.wandb_dir)

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
        reinit="finish_previous",
    )
    if run is None:
        return 0

    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)

            if "_position_heatmap" in payload:
                hm = payload.pop("_position_heatmap")
                x_labels = [str(round(x, 1)) for x in hm["x_labels"]]
                y_labels = [str(round(y, 1)) for y in hm["y_labels"]]
                blue_grid = hm["blue_grid"]
                orange_grid = hm["orange_grid"]
                step_count = hm.get("step_count", 0)

                rows = []
                for j, y_label in enumerate(y_labels):
                    for i, x_label in enumerate(x_labels):
                        rows.append([
                            x_label,
                            y_label,
                            blue_grid[j][i] if blue_grid else 0,
                            orange_grid[j][i] if orange_grid else 0,
                        ])

                table = wandb.Table(
                    data=rows,
                    columns=["x", "y", "blue_count", "orange_count"],
                )
                wandb.log({
                    "car_position_heatmap": table,
                    "car_position_heatmap_step_count": step_count,
                })

            if payload:
                wandb.log(payload)
    finally:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
