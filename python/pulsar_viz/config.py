from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_checkpoint_config_matches(config: dict[str, Any], checkpoint_path: str | Path) -> None:
    checkpoint_config = load_config(Path(checkpoint_path) / "config.json")
    if checkpoint_config != config:
        raise RuntimeError(
            "The requested evaluation config does not match the checkpoint config. "
            "Use the checkpoint's config.json or regenerate the checkpoint with the active config."
        )
