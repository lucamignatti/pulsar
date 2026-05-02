#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from two_stage_smoke_common import run_bc_pretrain


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: offline_binary_smoke.py <repo_root> <pulsar_bc_pretrain>")

    repo_root = Path(sys.argv[1]).resolve()
    pretrain_binary = Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory(prefix="pulsar_bc_pretrain_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_dir = run_bc_pretrain(
            repo_root,
            pretrain_binary,
            repo_root / "configs/2v2_bc.json",
            tmp_dir,
            "cpu",
            model_overrides={
                "encoder_dim": 16,
                "workspace_dim": 16,
                "stm_slots": 4,
                "stm_key_dim": 8,
                "stm_value_dim": 8,
                "ltm_slots": 4,
                "ltm_dim": 8,
                "controller_dim": 16,
                "value_hidden_dim": 32,
                "value_num_atoms": 51,
                "value_v_min": -10.0,
                "value_v_max": 10.0,
            },
        )

        for rel in [
            "model.pt",
            "actor_optimizer.pt",
            "bc_metrics.jsonl",
            "metadata.json",
            "config.json",
        ]:
            if not (output_dir / rel).exists():
                raise RuntimeError(f"missing BC pretraining artifact: {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
