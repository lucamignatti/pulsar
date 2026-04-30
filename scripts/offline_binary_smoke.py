#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from two_stage_smoke_common import run_offline_pretrain


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: offline_binary_smoke.py <repo_root> <pulsar_lfpo_pretrain>")

    repo_root = Path(sys.argv[1]).resolve()
    pretrain_binary = Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory(prefix="pulsar_lfpo_pretrain_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_dir = run_offline_pretrain(
            repo_root,
            pretrain_binary,
            repo_root / "configs/2v2_offline.json",
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
                "action_embedding_dim": 8,
            },
        )

        for rel in [
            "model.pt",
            "actor_optimizer.pt",
            "future_evaluator/model.pt",
            "future_evaluator/optimizer.pt",
            "offline_metrics.jsonl",
            "metadata.json",
            "config.json",
        ]:
            if not (output_dir / rel).exists():
                raise RuntimeError(f"missing LFPO pretraining artifact: {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
