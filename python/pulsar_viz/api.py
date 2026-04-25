from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import ensure_checkpoint_config_matches as _ensure_checkpoint_config_matches
from .config import load_config as _load_config
from .env import EvalBundle, action_index_to_env_action, make_eval_env as _make_eval_env
from .viz import run_viz_episode as _run_viz_episode


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_config(path)


def load_shared_model(checkpoint_path: str | Path, device: str = "cpu") -> Any:
    try:
        import pulsar_native  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The pulsar_native extension is not available. Build the C++ Python bindings first."
        ) from exc

    return pulsar_native.load_shared_model(str(checkpoint_path), device)


def make_eval_env(
    config: dict[str, Any],
    renderer_backend: str = "rlviser",
    udp_ip: str = "127.0.0.1",
    udp_port: int = 9273,
) -> EvalBundle:
    return _make_eval_env(config, renderer_backend=renderer_backend, udp_ip=udp_ip, udp_port=udp_port)


def run_viz_episode(
    model: Any,
    env: Any,
    renderer: Any,
    config: dict[str, Any],
    seed: int = 0,
    realtime: bool = True,
    policy_mode: str = "deterministic",
    startup_hook: Callable[[], None] | None = None,
) -> None:
    _run_viz_episode(
        model,
        env,
        renderer,
        config,
        seed,
        realtime=realtime,
        policy_mode=policy_mode,
        startup_hook=startup_hook,
    )


def ensure_checkpoint_config_matches(config: dict[str, Any], checkpoint_path: str | Path) -> None:
    _ensure_checkpoint_config_matches(config, checkpoint_path)


__all__ = [
    "load_config",
    "load_shared_model",
    "make_eval_env",
    "run_viz_episode",
    "action_index_to_env_action",
    "ensure_checkpoint_config_matches",
]
