#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_stub_stack(root: Path) -> None:
    write(
        root / "rlgym/rocket_league/common_values.py",
        "BLUE_TEAM = 0\nORANGE_TEAM = 1\n",
    )
    write(root / "rlgym/__init__.py", "")
    write(root / "rlgym/rocket_league/__init__.py", "")
    write(
        root / "rlgym/rocket_league/obs_builders.py",
        """
import numpy as np

class DefaultObs:
    def __init__(self, zero_padding=2):
        self.zero_padding = zero_padding

    def reset(self, agent_order, state, shared_info):
        self.agent_order = list(agent_order)

    def build_obs(self, agent_order, state, shared_info):
        obs = {}
        for idx, agent_id in enumerate(agent_order):
            vec = np.zeros(132, dtype=np.float32)
            vec[0] = float(agent_id)
            vec[1] = float(state.frame_index)
            vec[2] = float(idx)
            obs[agent_id] = vec
        return obs
""",
    )
    write(root / "rlgym_tools/__init__.py", "")
    write(root / "rlgym_tools/rocket_league/__init__.py", "")
    write(root / "rlgym_tools/rocket_league/replays/__init__.py", "")
    write(
        root / "rlgym_tools/rocket_league/replays/parsed_replay.py",
        """
class ParsedReplay:
    @staticmethod
    def load(path):
        return ParsedReplay()
""",
    )
    write(
        root / "rlgym_tools/rocket_league/replays/pick_action.py",
        """
import numpy as np

def get_best_action_options(car, replay_action, action_table, dodge_deadzone=0.5, greedy=True):
    probs = np.zeros(action_table.shape[0], dtype=np.float32)
    probs[5] = 1.0
    return probs

def get_weighted_action_options(car, replay_action, action_table, dodge_deadzone=0.5):
    probs = np.zeros(action_table.shape[0], dtype=np.float32)
    probs[5] = 0.75
    probs[9] = 0.25
    return probs
""",
    )
    write(
        root / "rlgym_tools/rocket_league/replays/convert.py",
        """
class Scoreboard:
    def __init__(self, go_to_kickoff=False, is_over=False):
        self.go_to_kickoff = go_to_kickoff
        self.is_over = is_over

class Car:
    def __init__(self, team_num):
        self.team_num = team_num
        self.is_demoed = False

class State:
    def __init__(self, frame_index):
        self.frame_index = frame_index
        self.cars = {
            0: Car(0),
            1: Car(0),
            2: Car(1),
            3: Car(1),
        }

class Frame:
    def __init__(self, frame_index, is_last=False):
        self.state = State(frame_index)
        self.actions = {agent_id: [0.0] * 8 for agent_id in self.state.cars}
        self.update_age = {agent_id: 0.0 for agent_id in self.state.cars}
        self.next_scoring_team = 0 if not is_last else None
        self.scoreboard = Scoreboard(go_to_kickoff=is_last, is_over=is_last)

def replay_to_rlgym(replay, interpolation="rocketsim", predict_pyr=True):
    yield Frame(0, is_last=False)
    yield Frame(1, is_last=True)
""",
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: preprocess_smoke.py <repo_root>")

    repo_root = Path(sys.argv[1]).resolve()
    script = repo_root / "scripts/preprocess_kaggle_2v2.py"

    with tempfile.TemporaryDirectory(prefix="pulsar_preprocess_smoke_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stubs = tmp_dir / "stubs"
        make_stub_stack(stubs)
        dataset_root = tmp_dir / "dataset/2v2"
        dataset_root.mkdir(parents=True)
        (dataset_root / "fake.replay").write_bytes(b"stub")
        output_dir = tmp_dir / "out"

        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{stubs}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(stubs)
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                str(tmp_dir / "dataset"),
                str(output_dir),
                "--max-replays",
                "1",
                "--shard-size",
                "8",
            ],
            check=True,
            cwd=repo_root,
            env=env,
        )

        train_manifest = json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))
        if train_manifest["observation_dim"] != 132 or train_manifest["action_dim"] != 90:
            raise RuntimeError("preprocess manifest dimensions mismatch")
        shard = train_manifest["shards"][0]
        obs = torch.load(output_dir / shard["obs_path"])
        action_probs = torch.load(output_dir / shard["action_probs_path"])
        episode_starts = torch.load(output_dir / shard["episode_starts_path"])
        if obs.size(1) != 132:
            raise RuntimeError("preprocess obs tensor width mismatch")
        if action_probs.size(1) != 90:
            raise RuntimeError("preprocess action_probs width mismatch")
        if episode_starts[0].item() != 1.0:
            raise RuntimeError("preprocess episode_starts should mark trajectory boundaries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
