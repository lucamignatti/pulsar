from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


def _require_rlgym() -> Any:
    try:
        from rlgym.api import RLGym
        from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
        from rlgym.rocket_league.obs_builders import DefaultObs
        from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
        from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence
    except ImportError as exc:
        raise RuntimeError(
            "rlgym is required for visualization. Install the project with `pip install .[viz]`."
        ) from exc

    return {
        "RLGym": RLGym,
        "LookupTableAction": LookupTableAction,
        "RepeatAction": RepeatAction,
        "DefaultObs": DefaultObs,
        "RocketSimEngine": RocketSimEngine,
        "FixedTeamSizeMutator": FixedTeamSizeMutator,
        "MutatorSequence": MutatorSequence,
    }


def _require_rlgym_tools() -> Any:
    try:
        from rlgym_tools.rocket_league.done_conditions.game_condition import GameCondition
        from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardProvider
        from rlgym_tools.rocket_league.state_mutators.game_mutator import GameMutator
    except ImportError as exc:
        raise RuntimeError(
            "rlgym-tools is required for full-match visualization. Install the project with `pip install .[viz]`."
        ) from exc

    return {
        "GameCondition": GameCondition,
        "ScoreboardProvider": ScoreboardProvider,
        "GameMutator": GameMutator,
    }


class ZeroReward:
    def reset(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_rewards(self, agents: list[Any], *args: Any, **kwargs: Any) -> dict[Any, float]:
        return {agent: 0.0 for agent in agents}


@dataclass(slots=True)
class EvalBundle:
    env: Any
    renderer: Any


def _make_renderer(
    renderer_backend: str,
    env_cfg: dict[str, Any],
    udp_ip: str,
    udp_port: int,
) -> Any:
    if renderer_backend == "rlviser":
        try:
            from rlgym.rocket_league.rlviser import RLViserRenderer
        except ImportError as exc:
            raise RuntimeError(
                "The RLViser backend requires `rlgym[rl-rlviser]` and a working RLViser runtime."
            ) from exc
        return RLViserRenderer(tick_rate=env_cfg["tick_rate"] / env_cfg["tick_skip"])

    if renderer_backend == "rocketsimvis":
        try:
            from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
        except ImportError as exc:
            raise RuntimeError(
                "The RocketSimVis backend requires `rlgym-tools`. Install the project with "
                "`pip install .[viz,offline]` or add `rlgym-tools` to the environment."
            ) from exc
        return RocketSimVisRenderer(udp_ip=udp_ip, udp_port=udp_port)

    raise ValueError(f"Unsupported renderer backend: {renderer_backend}")


def make_eval_env(
    config: dict[str, Any],
    renderer_backend: str = "rlviser",
    udp_ip: str = "127.0.0.1",
    udp_port: int = 9273,
) -> EvalBundle:
    mods = _require_rlgym()
    tool_mods = _require_rlgym_tools()
    env_cfg = config["env"]
    os.environ.setdefault("RS_COLLISION_MESHES", env_cfg.get("collision_meshes_path", "collision_meshes"))

    renderer = _make_renderer(renderer_backend, env_cfg, udp_ip, udp_port)
    game_length_seconds = float(env_cfg.get("game_length_seconds", 300.0))
    kickoff_timer_seconds = float(env_cfg.get("kickoff_timer_seconds", 5.0))
    state_mutator = mods["MutatorSequence"](
        mods["FixedTeamSizeMutator"](blue_size=env_cfg["team_size"], orange_size=env_cfg["team_size"]),
        tool_mods["GameMutator"](
            game_length_seconds=game_length_seconds,
            kickoff_timer_seconds=kickoff_timer_seconds,
        ),
    )

    env = mods["RLGym"](
        state_mutator=state_mutator,
        obs_builder=mods["DefaultObs"](zero_padding=env_cfg["team_size"]),
        action_parser=mods["RepeatAction"](mods["LookupTableAction"](), repeats=env_cfg["tick_skip"]),
        reward_fn=ZeroReward(),
        transition_engine=mods["RocketSimEngine"](),
        termination_cond=tool_mods["GameCondition"](),
        shared_info_provider=tool_mods["ScoreboardProvider"](),
        renderer=renderer,
    )
    return EvalBundle(env=env, renderer=renderer)


def action_index_to_env_action(config: dict[str, Any], action_index: int) -> np.ndarray:
    _ = config
    return np.asarray([action_index], dtype=np.int64)
