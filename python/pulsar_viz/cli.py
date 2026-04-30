from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import signal
import shutil
import sys
import tempfile

from .api import (
    ensure_checkpoint_config_matches,
    load_config,
    load_latent_future_actor,
    make_eval_env,
    run_viz_episode,
)
from .video import RLViserVideoRecorder


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


@contextmanager
def _ignore_sigint_during_cleanup() -> object:
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Pulsar visualization episode")
    parser.add_argument("--config", required=True, help="Path to the LFPO experiment JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint directory")
    parser.add_argument("--device", default="cpu", help="Torch device string")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument(
        "--renderer",
        choices=("rlviser", "rocketsimvis"),
        default="rlviser",
        help="Visualization backend",
    )
    parser.add_argument(
        "--policy",
        choices=("deterministic", "stochastic"),
        help="Policy sampling mode. Defaults to self_play_league.eval_policy when present.",
    )
    parser.add_argument("--udp-ip", default="127.0.0.1", help="RocketSimVis host")
    parser.add_argument("--udp-port", type=int, default=9273, help="RocketSimVis UDP port")
    parser.add_argument(
        "--video-out",
        help="Optional output video path. Captures the actual RLViser window on macOS.",
    )
    args = parser.parse_args()

    original_cwd = Path.cwd()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config = load_config(config_path)
    ensure_checkpoint_config_matches(config, checkpoint_path)
    if args.video_out and args.renderer != "rlviser":
        parser.error("--video-out currently requires --renderer rlviser")

    policy_mode = args.policy or config.get("self_play_league", {}).get("eval_policy", "deterministic")
    if policy_mode not in {"deterministic", "stochastic"}:
        parser.error(f"Unsupported policy mode: {policy_mode}")

    _resolve_collision_meshes(config, original_cwd)
    model = load_latent_future_actor(checkpoint_path, args.device)
    if args.renderer == "rocketsimvis":
        print(
            "RocketSimVis backend selected. Start the external RocketSimVis viewer separately; "
            "pulsar-viz only sends UDP state packets and will not open a window by itself.",
            file=sys.stderr,
        )
    elif args.renderer == "rlviser":
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
    print(f"Using {policy_mode} policy selection.", file=sys.stderr)
    if args.video_out:
        print(
            f"Recording RLViser window to {Path(args.video_out).expanduser().resolve()}.",
            file=sys.stderr,
        )

    with _rlviser_launch_dir() if args.renderer == "rlviser" else nullcontext():
        bundle = make_eval_env(config, renderer_backend=args.renderer, udp_ip=args.udp_ip, udp_port=args.udp_port)
        video_recorder: RLViserVideoRecorder | None = None
        interrupted = False
        if args.video_out:
            video_recorder = RLViserVideoRecorder(
                output_path=args.video_out,
                fps=config["env"]["tick_rate"] / config["env"]["tick_skip"],
            )
        try:
            try:
                run_viz_episode(
                    model,
                    bundle.env,
                    bundle.renderer,
                    config,
                    args.seed,
                    realtime=True,
                    policy_mode=policy_mode,
                    startup_hook=video_recorder.start if video_recorder is not None else None,
                )
            except KeyboardInterrupt:
                interrupted = True
                print("Interrupted. Stopping visualization and finalizing cleanup.", file=sys.stderr)
        finally:
            recorder_error: Exception | None = None
            renderer_error: Exception | None = None
            env_error: Exception | None = None
            with _ignore_sigint_during_cleanup():
                try:
                    if video_recorder is not None:
                        video_recorder.close()
                except Exception as exc:
                    recorder_error = exc
                try:
                    bundle.renderer.close()
                except Exception as exc:
                    renderer_error = exc
                finally:
                    try:
                        bundle.env.close()
                    except Exception as exc:
                        env_error = exc

            if interrupted:
                if recorder_error is not None:
                    print(f"Warning: failed to finalize recording cleanly: {recorder_error}", file=sys.stderr)
                if renderer_error is not None:
                    print(f"Warning: failed to close renderer cleanly: {renderer_error}", file=sys.stderr)
                if env_error is not None:
                    print(f"Warning: failed to close environment cleanly: {env_error}", file=sys.stderr)
                return

            if recorder_error is not None:
                raise recorder_error
            if renderer_error is not None:
                raise renderer_error
            if env_error is not None:
                raise env_error


if __name__ == "__main__":
    main()
