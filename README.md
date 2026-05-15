# Pulsar

Sparse-reward Rocket League bot training with **APPO**, **contrastive goal-conditioned auxiliary learning**, **all-mode self-play**, **PCGrad**, and **slow ES-LoRA**.

The training loop combines three complementary signals:

1. **Sparse APPO** — the main local optimizer, trained from terminal win/loss/touch outcomes combined with dense gameplay and mechanic rewards.
2. **Contrastive goal-conditioned auxiliary** — self-supervised future-goal encoders (state-action and goal) trained with symmetric InfoNCE. Provides a representation-learning signal that never shapes the environment reward.
3. **Slow Rank-4 ES-LoRA** — periodic global parameter-space search over a LoRA adapter on the final policy layer, driven by true sparse winrate with KL penalty.

The production setup trains 1v1, 2v2, and 3v3 from the start using one stable reward function. The curriculum layer is still available, but the default config uses it only as a mode-allocation wrapper rather than a staged reward schedule.

## Architecture

- **Choice of encoder**: SWA Transformer, **Mamba-2 (state space model)**, or MLP — configured via `model.encoder_type`
- **Contrastive goal-conditioned encoder pair** — state-action encoder and goal encoder trained with symmetric InfoNCE
- **Rank-4 LoRA adapter** on the final policy layer — mutated by ES, also trainable by APPO
- **ES-LoRA** periodic global optimizer: population sampling, antithetic evaluation, sparse-outcome fitness
- **Self-play league** with ELO ratings, snapshot management, and periodic evaluation against past policies
- **All-mode training** with PCGrad over per-mode minibatch groups
- **Comprehensive reward engine**: terminal outcome rewards, 18 mechanic detectors (speed flip, wavedash, flip reset, ceiling shot, air dribble, redirect, pinch, etc.), and dense gameplay rewards
- Optional **PCGrad** gradient projection for multi-task gradients
- **Perfetto-compatible tracing** for performance analysis

## Repository Layout

- `cpp/`: C++20 runtime, neural network models, APPO training loop, batched collector, reward engine, curriculum, tests, benchmarks
- `configs/`: experiment config files (JSON)
- `scripts/`: smoke tests, collision mesh downloader, W&B streaming, development setup
- `docs/`: platform notes (CUDA setup)
- `external/RocketSim/`: vendored RocketSim submodule
- `python/pulsar_viz/`: Python visualization and evaluation package
- `cmake/`: CMake dependency finders

## Requirements

- CMake 3.25+
- C++20 compiler
- Python 3.10 – 3.13
- `torch` and `pybind11` for the trainer and Python bindings
- `.[viz]` extras for visualization
- `.[offline]` extras for offline dataset tools

## Setup

```bash
git clone --recurse-submodules https://github.com/lucamignatti/pulsar.git
cd pulsar
python3 -m venv .venv
. .venv/bin/activate
pip install torch pybind11
pip install -e '.[viz]'
python3 scripts/collision_mesh_downloader.py
```

Build:

```bash
cmake -S . -B build/release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build build/release --parallel
```

For CUDA setup, see [docs/cuda_linux.md](docs/cuda_linux.md).

## Validation

```bash
ctest --test-dir build/release --output-on-failure
./build/release/pulsar_bench 20 configs/2v2_appo.json cuda:0
```

## Core Binary

- `pulsar_appo_train`: online APPO + goal-conditioned auxiliary + ES-LoRA + curriculum + self-play training

## Training

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs
```

To run a bounded number of updates:

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs 100
```

### Full Config Structure

```json
{
  "schema_version": 6,
  "obs_schema_version": 2,

  "env": {
    "mode": "soccar",
    "team_size": 2,
    "tick_skip": 8,
    "tick_rate": 120,
    "max_episode_ticks": 2250,
    "no_touch_timeout_seconds": 10.0,
    "randomize_kickoffs": true
  },

  "outcome": {
    "score": 0.0, "concede": 0.0,
    "neutral": 1.0, "neutral_no_touch": -1.0
  },

  "mechanic_rewards": { /* speed_flip, wavedash, flip_reset, ... */ },
  "dense_rewards": { /* ball_touch_vel, speed_toward_ball, boost_efficiency, ... */ },

  "curriculum": {
    "enabled": true,
    "stages": [
      { "name": "touch-1v1", "mode": "1v1", "min_agent_steps": 2500000, ... },
      { "name": "direction-1v1", "mode": "1v1", "unlocked_mechanics": ["speed_flip", "wavedash"], ... },
      { "name": "scoring-1v1", "mode": "1v1", ... },
      { "name": "aerials-1v1", "mode": "1v1", "unlocked_mechanics": ["air_dribble", "flip_reset", ...], ... },
      { "name": "teamplay", "mode_allocation": { "1v1": 0.5, "2v2": 0.5 }, ... },
      { "name": "aggression", "mode_allocation": { "1v1": 0.34, "2v2": 0.33, "3v3": 0.33 }, ... }
    ]
  },

  "action_table": { "builtin": "rlgym_lookup_v1" },

  "model": {
    "observation_dim": 172,
    "action_dim": 90,
    "encoder_type": "mamba2",
    "encoder_dim": 640,
    "num_encoder_blocks": 5,
    "value_hidden_dim": 384,
    "policy_hidden_dim": 0
  },

  "ppo": {
    "num_envs": 384,
    "collection_workers": 16,
    "collection_shards": 4,
    "rollout_length": 64,
    "minibatch_size": 8192,
    "update_epochs": 2,
    "optimizer_accumulation_steps": 2,
    "entropy_coef": 0.03,
    "entropy_floor": 0.08,
    "learning_rate": 0.0003,
    "max_grad_norm": 1.0,
    "device": "cuda",
    "pcgrad": true,
    "checkpoint_interval": 50,
    "max_rolling_checkpoints": 3
  },

  "goal_mapping": { "arena_max_distance": 8192.0 },
  "goal_critic": {
    "goal_dim": 3, "hidden_dim": 384, "embedding_dim": 64,
    "logsumexp_penalty_coeff": 0.01, "contrastive_batch_size": 2048,
    "max_future_horizon": 256
  },

  "es_lora": {
    "rank": 4, "lora_alpha": 4.0, "population_size": 16,
    "sigma_ES": 0.05, "eta_ES": 0.003, "es_interval": 25,
    "eval_rollout_length": 450, "beta_KL": 0.01
  },

  "self_play_league": {
    "enabled": true,
    "opponent_probability": 0.5,
    "snapshot_interval_updates": 50,
    "max_snapshots": 12
  },

  "wandb": { "enabled": true }
}
```

Self-play policy snapshots are written under `policy_versions/` when `self_play_league.enabled` is true.

## Observation Schema

The `PulsarObsBuilder` produces a fixed-size observation vector of dimension `52 + 40 * N` where `N` is the maximum team size (typically 3 for multi-mode training, zero-padded for smaller modes):

- Ball position, velocity, angular velocity (3 × 3 = 9 floats)
- Boost pad timers (34 floats)
- Self-car state (9 floats: jump status, handbrake, air time, etc.)
- Self-car physics (20 floats: position, forward, up, velocity, angular velocity, boost, on_ground, boosting, supersonic)
- Ally cars (up to N-1, each 20 floats, zero-padded)
- Enemy cars (up to N, each 20 floats, zero-padded)

## Action Space

The `rlgym_lookup_v1` builtin action table provides **90 discrete actions** combining:
- Ground actions: throttle (-1/0/1) × steer (-1/0/1) × boost (0/1) × handbrake (0/1) with boost-throttle constraints
- Aerial actions: pitch (-1/0/1) × yaw (-1/0/1) × roll (-1/0/1) × jump (0/1) × boost (0/1) with jump-yaw constraints

Per-agent **action masking** disables boost actions when out of boost and jump actions when the car has no flip/jump available, with a guaranteed no-op fallback.

## Encoder Options

Set `model.encoder_type` to one of:

| Type | Description |
|------|-------------|
| `"transformer"` | Sliding-window self-attention transformer with configurable window size, heads, and FFN multiplier. Uses a CLS token for pooled representation. |
| `"mamba2"` | State space model (Mamba-2) with selective scan. Supports recurrent inference with `initial_recurrent_state` / `forward_step` for efficient autoregressive deployment. |
| `"mlp"` | Simple multi-layer perceptron with ReLU activations. |

## Mode Allocation

The production config uses a single stable reward stage with a 1v1 / 2v2 / 3v3 allocation from the first rollout. The trainer dynamically rebuilds collectors before collection so each mode gets its own environment group and PCGrad can project per-mode gradients.

The curriculum system still supports staged reward schedules, promotion gates, demotion, mechanic unlocks, and per-stage overrides for experiments that need them.

## Self-Play League

When `self_play_league.enabled` is true:
- Every `snapshot_interval_updates`, the current policy is snapshotted (model + normalizer + ELO)
- During training, opponents are sampled from past snapshots with configurable probability
- Opponent actions are inferred using the snapshot's policy with the configured sampling mode (deterministic/stochastic)
- Periodic evaluation matches update ELO ratings (initial 1000, K-factor 32)
- Snapshot count is capped at `max_snapshots` (oldest trimmed)

## Reward Engine

The reward system has three components, each with per-episode caps defined in the config:

### Terminal Outcomes
- `score`: agent's team scored
- `concede`: opponent scored
- `neutral`: episode ended without a goal
- `neutral_no_touch`: ended without the agent ever touching the ball

### Dense Gameplay Rewards (configurable weights)
- Ball touch velocity, touch direction, speed toward ball, face ball
- Velocity ball-to-goal, air reward, air touch (height × air-time quality)
- Save boost, boost efficiency, boost used (toward goal), defensive positioning
- Shot accuracy, big/small boost pickup

### Mechanic Detectors (gated by curriculum unlocks)
- Speed flip, wavedash, chain dash, half flip, wall dash
- Air dribble (consecutive touches scaling), flip reset, ceiling shot
- Double tap, preflip, redirect, pogo, pinch, team pinch
- Kickoff first touch

## Distributed Collection

Environments are sharded across `collection_shards`, each with its own `BatchedRocketSimCollector`. Collection can be further parallelized with `collection_workers` using the `ParallelExecutor` (lock-free work-stealing thread pool). Observations and action masks are built in parallel, then transferred to the GPU via pinned host buffers when CUDA is the runtime device.

## Visualizing a Checkpoint

```bash
pulsar-viz --config /path/to/checkpoint/config.json --checkpoint /path/to/checkpoint --device cpu
```

Additional options:

```bash
pulsar-viz \
  --config /path/to/checkpoint/config.json \
  --checkpoint /path/to/checkpoint \
  --device cuda \
  --seed 42 \
  --renderer rlviser        # or rocketsimvis (needs external viewer)
  --policy deterministic     # or stochastic
  --udp-ip 127.0.0.1 \
  --udp-port 9273 \
  --video-out ./recording.mp4  # macOS screen capture of RLViser window
```
