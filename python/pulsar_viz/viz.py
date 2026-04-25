from __future__ import annotations

from functools import lru_cache
import random
import time
from typing import Any, Callable

import numpy as np

from .env import action_index_to_env_action


@lru_cache(maxsize=8)
def _cached_builtin_action_mask_fields(builtin_name: str) -> tuple[tuple[bool, bool], ...]:
    if builtin_name != "rlgym_lookup_v1":
        raise ValueError(f"Unsupported builtin action table: {builtin_name}")

    mask_fields: list[tuple[bool, bool]] = []

    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    _ = handbrake
                    if boost == 1 and throttle != 1:
                        continue
                    mask_fields.append((boost == 1, False))

    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == 0 and roll == 0 and jump == 0:
                            continue
                        mask_fields.append((boost == 1, jump == 1))

    return tuple(mask_fields)


def _action_mask_fields(config: dict[str, Any]) -> tuple[tuple[bool, bool], ...]:
    action_table_cfg = config.get("action_table", {})
    builtin_name = action_table_cfg.get("builtin", "")
    if builtin_name:
        return _cached_builtin_action_mask_fields(builtin_name)

    actions = action_table_cfg.get("actions", [])
    if not actions:
        raise ValueError("Action table configuration is empty.")
    return tuple((bool(action.get("boost", False)), bool(action.get("jump", False))) for action in actions)


def _ordered_agent_ids(observations: dict[Any, np.ndarray]) -> list[Any]:
    try:
        return sorted(observations)
    except TypeError:
        return sorted(observations, key=lambda agent_id: (type(agent_id).__module__, type(agent_id).__qualname__, repr(agent_id)))


def _has_flip(car: Any) -> bool:
    if hasattr(car, "has_flip"):
        return bool(car.has_flip)
    return bool(not car.has_double_jumped and not car.has_flipped)


def _masked_action_index(
    logits: np.ndarray,
    mask: np.ndarray,
    policy_mode: str,
) -> int:
    masked_logits = np.asarray(logits, dtype=np.float32).copy()
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.any():
        masked_logits[~mask_array] = -np.inf
    if policy_mode == "stochastic":
        masked_logits = masked_logits + np.random.gumbel(size=masked_logits.shape).astype(np.float32)
    return int(np.argmax(masked_logits))


def _select_actions(
    model: Any,
    observations: dict[Any, np.ndarray],
    state: Any,
    config: dict[str, Any],
    policy_mode: str,
) -> dict[Any, np.ndarray]:
    agent_ids = _ordered_agent_ids(observations)
    batch_obs = [observations[agent_id].tolist() for agent_id in agent_ids]
    batch_logits = model.forward_batch(batch_obs)
    action_mask_fields = _action_mask_fields(config)
    action_map: dict[Any, np.ndarray] = {}
    for agent_id, logits in zip(agent_ids, batch_logits, strict=False):
        car = state.cars[agent_id]
        can_boost = float(car.boost_amount) > 0.5
        can_jump = bool(car.on_ground or _has_flip(car))
        mask = np.asarray(
            [
                (not requires_boost or can_boost) and (not requires_jump or can_jump)
                for requires_boost, requires_jump in action_mask_fields
            ],
            dtype=np.bool_,
        )
        action_index = _masked_action_index(np.asarray(logits, dtype=np.float32), mask, policy_mode)
        action_map[agent_id] = action_index_to_env_action(config, action_index)
    return action_map


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
    if policy_mode not in {"deterministic", "stochastic"}:
        raise ValueError(f"Unsupported policy mode: {policy_mode}")

    np.random.seed(seed)
    random.seed(seed)
    try:
        observations = env.reset(seed=seed)
    except TypeError:
        observations = env.reset()
    model.reset(len(observations))
    renderer.render(env.state, {})
    if startup_hook is not None:
        startup_hook()
    done = False
    step_seconds = config["env"]["tick_skip"] / config["env"]["tick_rate"]

    while not done:
        step_start = time.perf_counter()
        actions = _select_actions(model, observations, env.state, config, policy_mode)
        observations, _, terminated, truncated = env.step(actions)
        renderer.render(env.state, {})
        done = all(terminated.values()) or all(truncated.values())
        if realtime:
            remaining = step_seconds - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
