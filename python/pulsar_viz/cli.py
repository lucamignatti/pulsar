from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .api import (
    ensure_checkpoint_config_matches,
    load_config,
    load_shared_model,
    make_eval_env,
    run_viz_episode,
)


def _resolve_collision_meshes(config: dict, base_dir: Path) -> None:
    collision_path = config.get("env", {}).get("collision_meshes_path")
    if not collision_path:
        return
    collision_path = Path(collision_path).expanduser()
    if collision_path.is_absolute():
        return
    config["env"]["collision_meshes_path"] = str((base_dir / collision_path).resolve())


def _find_pyviser() -> Path | None:
    candidates = []
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name and argv0.name != "-":
        candidates.append(argv0.resolve().parent / "pyviser")
    candidates.append(Path(sys.executable).resolve().parent / "pyviser")
    which_pyviser = shutil.which("pyviser")
    if which_pyviser:
        candidates.append(Path(which_pyviser).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@contextmanager
def _rlviser_launch_dir() -> object:
    original_cwd = Path.cwd()
    pyviser = _find_pyviser()
    if pyviser is None:
        yield
        return

    with tempfile.TemporaryDirectory(prefix="pulsar-rlviser-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "rlviser").symlink_to(pyviser)
        os.chdir(tmp_path)
        try:
            yield
        finally:
            os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Pulsar visualization episode")
    parser.add_argument("--config", required=True, help="Path to the shared experiment JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device string")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument(
        "--renderer",
        choices=("rlviser", "rocketsimvis"),
        default="rlviser",
        help="Visualization backend",
    )
    parser.add_argument("--udp-ip", default="127.0.0.1", help="RocketSimVis host")
    parser.add_argument("--udp-port", type=int, default=9273, help="RocketSimVis UDP port")
    args = parser.parse_args()

    original_cwd = Path.cwd()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config = load_config(config_path)
    ensure_checkpoint_config_matches(config, checkpoint_path)
    _resolve_collision_meshes(config, original_cwd)
    model = load_shared_model(checkpoint_path, args.device)
    if args.renderer == "rocketsimvis":
        print(
            "RocketSimVis backend selected. Start the external RocketSimVis viewer separately; "
            "pulsar-viz only sends UDP state packets and will not open a window by itself.",
            file=sys.stderr,
        )
    else:
        pyviser = _find_pyviser()
        if pyviser is not None:
            print(
                f"Using packaged RLViser binary at {pyviser}.",
                file=sys.stderr,
            )
        else:
            print(
                "No packaged `pyviser` binary found. RLViser will be resolved from the current working directory.",
                file=sys.stderr,
            )

    with _rlviser_launch_dir() if args.renderer == "rlviser" else nullcontext():
        bundle = make_eval_env(config, renderer_backend=args.renderer, udp_ip=args.udp_ip, udp_port=args.udp_port)
        try:
            run_viz_episode(model, bundle.env, bundle.renderer, config, args.seed)
        finally:
            bundle.renderer.close()
            bundle.env.close()


if __name__ == "__main__":
    main()
