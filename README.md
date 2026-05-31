# Pulsar

Rocket League bot trained from sparse scoring rewards only — no hand-crafted reward shaping.

The training system uses **PPO with a quasimetric contrastive critic** to learn goal-conditioned policies entirely from ±1 terminal rewards, combined with a **reachability-grid curriculum** that automatically expands the bot's competence outward from easy positions to full-field play.

## How it works

Three components work together:

**Quasimetric contrastive critic** — A two-tower network (`φ` over state-action, `ψ` over goal) trained with symmetric InfoNCE on hindsight-relabeled goals. The asymmetric softplus distance `softplus(φ − ψ).sum(−1)` gives a directed reachability signal: how hard is it to get from this state to that goal? This critic's output drives the PPO advantage rather than the environment reward, providing a dense learning signal even when goals are rarely reached.

**PPO update** — Standard clipped surrogate with GAE baseline, separate offense/defense weighting (1.0 / 0.85), and advantage sign-flip for the defending Orange agent. Three Adam optimisers (actor, value, critic) updated jointly each rollout.

**Reachability-grid curriculum** — The field is partitioned into a 12 × 16 grid of ball positions. Training starts from the center cell (scorer behind ball, stationary). When the policy's per-cell KL divergence EWMA drops below a mastery threshold, neighboring cells open up. Four velocity tiers progressively add ball and car speed. Once the full field is mastered at the highest tier, resets switch to standard kickoff.

## Architecture

- **Policy**: 128 × 128 tanh MLP with a Gaussian head and a goal residual connection — the goal vector (opponent-net position) is projected and added to the hidden state
- **Value**: 128 × 128 × 1 MLP over `[obs | goal]`
- **Critic**: Quasimetric two-tower MLP (`φ`: 128 → 32, `ψ`: 128 → 32), L2-normalised embeddings
- **Observation**: 47-dim symmetric team-relative — ego car (19) | ball (9) | opponent (19). Orange perspective mirrors X/Y. No reward shaping, no hand-crafted features.
- **Actions**: Continuous 8-dim Gaussian (throttle, steer, pitch, yaw, roll, jump, boost, handbrake). Orange steering/yaw/roll mirrored automatically.
- **Environment**: RocketSim 1v1 self-play, `tick_skip = 8`, up to 200 steps per episode

## Repository layout

```
cpp/
  pulsar/          # Training system (models, collector, update loop, curriculum)
  src/             # Core infrastructure (RocketSim engine, config, checkpoint, W&B logging)
  include/pulsar/  # Core headers (types, interfaces, env, training utilities)
configs/
  desktop.json     # Tuned for 7900X + 6800 XT
external/
  RocketSim/       # Physics simulation
```

## Requirements

- CMake 3.25+
- C++20 compiler (GCC 12+ or Clang 15+)
- PyTorch (CPU, CUDA, or ROCm build)
- RocketSim collision meshes (included as a submodule)

## Build

```bash
git clone --recurse-submodules https://github.com/lucamignatti/pulsar.git
cd pulsar

cmake -S . -B build/pulsar \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DPULSAR_ENABLE_PYTHON=OFF \
  -DPULSAR_ENABLE_BENCHMARKS=OFF \
  -DPULSAR_ENABLE_ROCKETSIM=ON \
  -DPULSAR_ENABLE_TRAINING=ON

cmake --build build/pulsar --parallel
```

For AMD ROCm/HIP setup see [docs/hip_linux.md](docs/hip_linux.md). For CUDA see [docs/cuda_linux.md](docs/cuda_linux.md).

## Train

```bash
# Run indefinitely (Ctrl+C to stop)
./build/pulsar/pulsar_train configs/desktop.json

# Bounded run
./build/pulsar/pulsar_train configs/desktop.json /path/to/output 1000
```

The device is selected automatically: ROCm/CUDA if available, MPS on Apple Silicon, otherwise CPU.

## Config

Key parameters in `configs/desktop.json` (tuned for 7900X + 6800 XT):

| Key | Default | Notes |
|---|---|---|
| `num_envs` | 32 | Parallel 1v1 arenas |
| `num_workers` | 11 | CPU threads for env stepping (= physical cores − 1) |
| `num_steps` | 256 | Rollout length per update |
| `minibatch_size` | 4096 | PPO minibatch |
| `lr` | 3e-4 | Adam learning rate (all three optimisers) |
| `tick_skip` | 8 | Physics ticks per action |
| `max_steps` | 200 | Max steps per episode before truncation |
| `collision_meshes_path` | `collision_meshes` | Path to RocketSim mesh files |
