from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from .env import action_index_to_env_action


def _select_actions(model: Any, observations: dict[Any, np.ndarray], config: dict[str, Any]) -> dict[Any, np.ndarray]:
    agent_ids = list(observations.keys())
    batch_obs = [observations[agent_id].tolist() for agent_id in agent_ids]
    batch_logits = model.forward_batch(batch_obs)
    action_map: dict[Any, np.ndarray] = {}
    for agent_id, logits in zip(agent_ids, batch_logits, strict=False):
        action_index = int(np.argmax(np.asarray(logits, dtype=np.float32)))
        action_map[agent_id] = action_index_to_env_action(config, action_index)
    return action_map


def run_viz_episode(model: Any, env: Any, renderer: Any, config: dict[str, Any], seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        observations = env.reset(seed=seed)
    except TypeError:
        observations = env.reset()
    model.reset(len(observations))
    renderer.render(env.state, {})
    done = False
    step_seconds = config["env"]["tick_skip"] / config["env"]["tick_rate"]

    while not done:
        step_start = time.perf_counter()
        actions = _select_actions(model, observations, config)
        observations, _, terminated, truncated = env.step(actions)
        renderer.render(env.state, {})
        done = all(terminated.values()) or all(truncated.values())
        remaining = step_seconds - (time.perf_counter() - step_start)
        if remaining > 0.0:
            time.sleep(remaining)
