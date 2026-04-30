#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
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


def _cleanup_partial_archives(cache_dir: Path | None) -> None:
    if cache_dir is None or not cache_dir.exists():
        return
    for partial in cache_dir.rglob("*.archive"):
        try:
            partial.unlink()
            print(f"removed_partial_archive={partial}", file=sys.stderr)
        except OSError as exc:
            print(f"failed_to_remove_partial_archive={partial} error={exc}", file=sys.stderr)


def _download_with_retries(kagglehub, dataset: str, retries: int, retry_wait_seconds: float, cache_dir: Path | None) -> Path:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"dataset_download_attempt={attempt}/{retries}", file=sys.stderr)
            return Path(kagglehub.dataset_download(dataset)).resolve()
        except Exception as exc:  # pragma: no cover - network and kagglehub dependent
            last_error = exc
            print(f"dataset_download_failed_attempt={attempt} error={exc}", file=sys.stderr)
            if attempt >= retries:
                break
            _cleanup_partial_archives(cache_dir)
            time.sleep(retry_wait_seconds)
    raise RuntimeError(f"dataset download failed after {retries} attempts") from last_error


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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for kagglehub's download cache. Defaults to <output-parent>/.kagglehub "
            "when --output is provided, otherwise kagglehub's default cache is used."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of full dataset download attempts. Defaults to 5.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=30.0,
        help="Seconds to sleep between failed download attempts. Defaults to 30.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None and args.output is not None:
        cache_dir = args.output.expanduser().resolve().parent / ".kagglehub"
    if cache_dir is not None:
        cache_dir = cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
        print(f"kagglehub_cache_root={cache_dir}")

    kagglehub = _require_kagglehub()
    source = _download_with_retries(
        kagglehub,
        args.dataset,
        max(1, args.retries),
        max(0.0, args.retry_wait_seconds),
        cache_dir,
    )

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
