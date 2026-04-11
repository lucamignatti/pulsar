#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_DATASET = "rolvarild/high-level-rocket-league-replay-dataset"


def _require_kagglehub():
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-driven
        raise SystemExit(
            "kagglehub is required to download the Kaggle dataset. "
            "Install the offline extras or run `.venv/bin/pip install kagglehub`."
        ) from exc
    return kagglehub


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _materialize_output(source: Path, output: Path, mode: str, force: bool) -> None:
    if output.exists() or output.is_symlink():
        if not force:
            raise SystemExit(
                f"Refusing to overwrite existing path: {output}. "
                "Pass --force to replace it."
            )
        _remove_existing(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        output.symlink_to(source, target_is_directory=True)
        return

    if mode == "copy":
        shutil.copytree(source, output)
        return

    raise SystemExit(f"Unsupported output mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kaggle high-level Rocket League replay dataset through kagglehub "
            "and optionally materialize it at a stable local path."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug. Defaults to {DEFAULT_DATASET!r}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional stable output path. By default the script only prints the kagglehub cache path. "
            "If provided, use --mode symlink or --mode copy to materialize it."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize --output. Defaults to symlink.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing --output path if present.",
    )
    args = parser.parse_args()

    kagglehub = _require_kagglehub()
    source = Path(kagglehub.dataset_download(args.dataset)).resolve()

    print(f"kagglehub_cache_path={source}")

    if args.output is not None:
        output = args.output.expanduser().resolve()
        _materialize_output(source, output, args.mode, args.force)
        print(f"materialized_path={output}")
        print(f"materialized_mode={args.mode}")

    print("dataset_download_complete=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
