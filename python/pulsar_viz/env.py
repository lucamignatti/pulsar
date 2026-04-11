from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


def _require_rlgym() -> Any:
    try:
        from rlgym.api import RLGym
        from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
        from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
        from rlgym.rocket_league.obs_builders import DefaultObs
        from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
        from rlgym.rocket_league.rlviser import RLViserRenderer
        from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
        from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    except ImportError as exc:
        raise RuntimeError(
            "rlgym and rlviser are required for visualization. Install the project with "
            "`pip install .[viz]` and ensure the RLViser runtime is available."
        ) from exc

    return {
        "RLGym": RLGym,
        "LookupTableAction": LookupTableAction,
        "RepeatAction": RepeatAction,
        "GoalCondition": GoalCondition,
        "NoTouchTimeoutCondition": NoTouchTimeoutCondition,
        "TimeoutCondition": TimeoutCondition,
        "DefaultObs": DefaultObs,
        "CombinedReward": CombinedReward,
        "GoalReward": GoalReward,
        "TouchReward": TouchReward,
        "RLViserRenderer": RLViserRenderer,
        "RocketSimEngine": RocketSimEngine,
        "FixedTeamSizeMutator": FixedTeamSizeMutator,
        "KickoffMutator": KickoffMutator,
        "MutatorSequence": MutatorSequence,
    }


@dataclass(slots=True)
class EvalBundle:
    env: Any
    renderer: Any


def make_eval_env(config: dict[str, Any]) -> EvalBundle:
    mods = _require_rlgym()
    env_cfg = config["env"]
    os.environ.setdefault("RS_COLLISION_MESHES", env_cfg.get("collision_meshes_path", "collision_meshes"))

    renderer = mods["RLViserRenderer"](tick_rate=env_cfg["tick_rate"] / env_cfg["tick_skip"])
    state_mutator = mods["MutatorSequence"](
        mods["FixedTeamSizeMutator"](blue_size=env_cfg["team_size"], orange_size=env_cfg["team_size"]),
        mods["KickoffMutator"](),
    )

    reward = mods["CombinedReward"](
        (mods["GoalReward"](), 1.0),
        (mods["TouchReward"](), 0.1),
    )

    env = mods["RLGym"](
        state_mutator=state_mutator,
        obs_builder=mods["DefaultObs"](zero_padding=env_cfg["team_size"]),
        action_parser=mods["RepeatAction"](mods["LookupTableAction"](), repeats=env_cfg["tick_skip"]),
        reward_fn=reward,
        transition_engine=mods["RocketSimEngine"](),
        termination_cond=mods["GoalCondition"](),
        truncation_cond=mods["NoTouchTimeoutCondition"](env_cfg["no_touch_timeout_seconds"]),
        renderer=renderer,
    )
    return EvalBundle(env=env, renderer=renderer)


def action_index_to_env_action(config: dict[str, Any], action_index: int) -> np.ndarray:
    _ = config
    return np.asarray([action_index], dtype=np.int64)
