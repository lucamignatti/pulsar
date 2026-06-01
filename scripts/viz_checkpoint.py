#!/usr/bin/env python3
"""Visualize a Pulsar checkpoint by running self-play in RocketSim + RLViser.

Usage:
    python scripts/viz_checkpoint.py <checkpoint_dir> [--episodes N] [--deterministic]

    checkpoint_dir: path containing actor.pt (e.g. ~/Documents/pulsar_out/ckpt_500)

Controls (RLViser window):
    Space  — pause/resume
    +/-    — speed up/slow down

The obs builder and action parser are loaded from the compiled pulsar_py module
(build/release/pulsar_py.so) so C++ is always the single source of truth.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── pulsar_py binding ─────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_BUILD_DIR  = _SCRIPT_DIR.parent / "build" / "release"
sys.path.insert(0, str(_BUILD_DIR))
try:
    import pulsar_py as pulsar
except ModuleNotFoundError:
    sys.exit(
        "pulsar_py.so not found. Build with:\n"
        "  cmake -S . -B build/release \\\n"
        "    -DCMAKE_PREFIX_PATH='...' \\\n"
        "    -DROCM_PATH=/usr/lib64/rocm \\\n"
        "    -DPULSAR_ENABLE_PYTHON=ON\n"
        "  cmake --build build/release --parallel $(nproc)"
    )

# ── RocketSim + RLViser ───────────────────────────────────────────────────────
try:
    import RocketSim as rsim
except ModuleNotFoundError:
    sys.exit("RocketSim not installed. Run: pip install rocketsim")
try:
    import rlviser_py as rlviser
except ModuleNotFoundError:
    sys.exit("rlviser_py not installed. Run: pip install 'rlgym[rl-rlviser]'")

# ── RocketSim initialisation ─────────────────────────────────────────────────
_MESHES = str(_SCRIPT_DIR.parent / "collision_meshes")
try:
    rsim.init(_MESHES)
except Exception as e:
    sys.exit(f"RocketSim init failed: {e}\nCollision meshes expected at {_MESHES}")

# ── Boost pad locations for RLViser ──────────────────────────────────────────
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
rlviser.set_boost_pad_locations(BOOST_LOCATIONS)

# ============================================================================
# Python model — mirrors PulsarActorImpl exactly
# ============================================================================
OBS_DIM    = 47
ACTION_DIM = 8
GOAL_DIM   = 6
HIDDEN_DIM = 128


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2),
                bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PulsarActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(
            _layer_init(nn.Linear(OBS_DIM, HIDDEN_DIM)),
            nn.Tanh(),
            _layer_init(nn.Linear(HIDDEN_DIM, HIDDEN_DIM)),
            nn.Tanh(),
        )
        self.actor_mean  = _layer_init(nn.Linear(HIDDEN_DIM, ACTION_DIM), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, ACTION_DIM))
        self.goal_proj = nn.Linear(GOAL_DIM, HIDDEN_DIM)
        nn.init.normal_(self.goal_proj.weight, 0.0, 0.1)
        nn.init.zeros_(self.goal_proj.bias)

    def forward(self, obs: torch.Tensor,
                goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.base(obs)
        feat = feat + self.goal_proj(goal.to(feat.dtype))
        mean = self.actor_mean(feat)
        std  = torch.exp(self.actor_logstd.expand_as(mean))
        return mean, std

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray,
            deterministic: bool = False) -> np.ndarray:
        obs_t  = torch.from_numpy(obs).float().unsqueeze(0)
        goal_t = torch.from_numpy(goal).float().unsqueeze(0)
        mean, std = self.forward(obs_t, goal_t)
        if deterministic:
            return mean.squeeze(0).numpy()
        return (mean + std * torch.randn_like(mean)).squeeze(0).numpy()


# ============================================================================
# Obs/action helpers — delegated to pulsar_py (C++ source of truth)
# ============================================================================
GOAL_SCORE  = pulsar.goal_score()    # Blue scorer
GOAL_DEFEND = pulsar.goal_defend()   # Orange defender


def get_obs(car_states: list, ball_state) -> list[np.ndarray]:
    """Build obs for both agents from RocketSim states."""
    obs_list = []
    for agent_id in range(2):
        invert = (agent_id == 1)
        ego = car_states[agent_id]
        opp = car_states[1 - agent_id]

        def vec3(v):  # rsim.Vec → float[3]
            return np.array([v.x, v.y, v.z], dtype=np.float32)

        fwd = ego.rot_mat.forward
        up  = ego.rot_mat.up
        o_fwd = opp.rot_mat.forward
        o_up  = opp.rot_mat.up

        obs = pulsar.build_obs(
            invert,
            vec3(ego.pos),  vec3(ego.vel),
            vec3(fwd),      vec3(up),
            vec3(ego.ang_vel), ego.boost,
            vec3(ball_state.pos), vec3(ball_state.vel), vec3(ball_state.ang_vel),
            vec3(opp.pos),  vec3(opp.vel),
            vec3(o_fwd),    vec3(o_up),
            vec3(opp.ang_vel), opp.boost,
        )
        obs_list.append(obs)
    return obs_list


def apply_action(car, action_arr: np.ndarray, agent_id: int) -> None:
    """Parse action array and set RocketSim car controls."""
    ctrl_dict = pulsar.parse_action(action_arr.astype(np.float32), agent_id)
    ctrl = rsim.CarControls()
    ctrl.throttle  = float(ctrl_dict["throttle"])
    ctrl.steer     = float(ctrl_dict["steer"])
    ctrl.pitch     = float(ctrl_dict["pitch"])
    ctrl.yaw       = float(ctrl_dict["yaw"])
    ctrl.roll      = float(ctrl_dict["roll"])
    ctrl.jump      = bool(ctrl_dict["jump"])
    ctrl.boost     = bool(ctrl_dict["boost"])
    ctrl.handbrake = bool(ctrl_dict["handbrake"])
    car.set_controls(ctrl)


# ============================================================================
# RLViser rendering
# ============================================================================
def render(arena: rsim.Arena, cars: list, tick_count: int,
           tick_rate: float = 120 / 8) -> None:
    ball_state = arena.ball.get_state()

    ball = rsim.BallState()
    ball.pos     = ball_state.pos
    ball.vel     = ball_state.vel
    ball.ang_vel = ball_state.ang_vel
    ball.rot_mat = ball_state.rot_mat

    pad_states = [bool(p.get_state().is_active) for p in arena.get_boost_pads()]

    car_data = []
    for idx, car in enumerate(cars):
        team   = rsim.Team.BLUE if idx == 0 else rsim.Team.ORANGE
        cs     = car.get_state()
        car_data.append((idx + 1, team, rsim.CarConfig(rsim.CarConfig.octane()), cs))

    rlviser.render(
        tick_count=tick_count,
        tick_rate=tick_rate,
        game_mode=rsim.GameMode.SOCCAR,
        boost_pad_states=pad_states,
        ball=ball,
        cars=car_data,
    )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Pulsar checkpoint")
    parser.add_argument("checkpoint_dir", help="Directory containing actor.pt")
    parser.add_argument("--episodes",     type=int, default=0,
                        help="Number of episodes to run (0 = unlimited)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use mean action instead of sampling")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_dir) / "actor.pt"
    if not ckpt_path.exists():
        sys.exit(f"actor.pt not found at {ckpt_path}")

    # ---- Load model ---------------------------------------------------------
    actor = PulsarActor()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()
    print(f"Loaded actor from {ckpt_path}")

    # ---- Arena setup --------------------------------------------------------
    arena = rsim.Arena(rsim.GameMode.SOCCAR)
    blue_car   = arena.add_car(rsim.Team.BLUE,   rsim.CarConfig(rsim.CarConfig.octane()))
    orange_car = arena.add_car(rsim.Team.ORANGE, rsim.CarConfig(rsim.CarConfig.octane()))
    cars = [blue_car, orange_car]

    TICK_SKIP  = 8
    MAX_STEPS  = 200
    episode    = 0
    tick_count = 0

    rlviser.launch()
    time.sleep(0.5)  # give RLViser window time to open

    try:
        while args.episodes == 0 or episode < args.episodes:
            arena.reset_kickoff()
            steps = 0
            done  = False

            while not done and steps < MAX_STEPS:
                car_states  = [c.get_state() for c in cars]
                ball_state  = arena.ball.get_state()
                obs_list    = get_obs(car_states, ball_state)

                for agent_id, car in enumerate(cars):
                    goal = GOAL_SCORE if agent_id == 0 else GOAL_DEFEND
                    act  = actor.act(obs_list[agent_id], goal, args.deterministic)
                    apply_action(car, act, agent_id)

                arena.step(TICK_SKIP)
                tick_count += TICK_SKIP
                steps      += 1

                render(arena, cars, tick_count)

                # Check done: goal scored or ball OOB
                bs = arena.ball.get_state()
                if abs(bs.pos.y) > 5120 or abs(bs.pos.x) > 4096:
                    done = True

            episode += 1
            outcome  = "goal" if done else "timeout"
            print(f"Episode {episode} ended ({outcome})")

    except KeyboardInterrupt:
        pass
    finally:
        rlviser.quit()


if __name__ == "__main__":
    main()
