# Pulsar

Sparse-reward Rocket League bot training with **Continuous Goal-Conditioned Distributional APPO** and **Slow ES-LoRA**.

Three mechanisms work together:

1. **Sparse Distributional APPO** — the main local optimizer, trained from terminal win/loss outcomes only.
2. **Continuous goal-conditioned distributional critic** — an auxiliary 51-bin categorical critic that predicts future discounted car-to-ball proximity. Provides an immediate controllability gradient via a small goal actor loss.
3. **Slow Rank-4 ES-LoRA** — periodic global parameter-space search over a LoRA adapter on the final policy layer, driven by true sparse winrate plus goal pressure.

The real environment reward remains strictly sparse terminal win/loss. The goal-distance machinery is an auxiliary critic target, never the environment reward.

## Architecture

- **Continuum** recurrent actor with policy and distributional value heads
- **51-bin categorical distributional sparse value critic** for terminal win/loss
- **51-bin goal-conditioned distributional critic** `Z_g(h_t, a_t, g)` predicting car-to-ball proximity
- **Rank-4 LoRA adapter** on the final policy layer — mutated by ES, also trainable by APPO
- **ES-LoRA** periodic global optimizer: population sampling, antithetic evaluation, sparse-outcome fitness

## Repository Layout

- `cpp/`: C++20 runtime, Continuum model, training loop, collector, tests, benchmarks
- `configs/`: experiment config files
- `scripts/`: smoke tests, W&B streaming, utility scripts
- `docs/`: platform notes (CUDA setup)
- `external/RocketSim/`: vendored RocketSim submodule
- `python/pulsar_viz/`: visualization package

## Requirements

- CMake 3.25+
- C++20 compiler
- Python 3.10+
- `torch` and `pybind11` for the trainer and Python bindings
- `.[viz]` extras for visualization

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
./build/release/pulsar_bench
```

## Core Binary

- `pulsar_appo_train`: online APPO + GCRL + ES-LoRA self-play training

## Training

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs
```

To run a bounded number of updates:

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs 100
```

### Key Config Sections

```json
{
  "outcome":       { "score": 1.0, "concede": -1.0, "neutral": 0.0 },
  "goal_mapping":  { "goal": 0.0, "kernel_sigma": 0.08, "arena_max_distance": 8192.0 },
  "goal_critic":   { "horizon_H": 256, "gamma_g": 0.99, "num_atoms": 51, "lambda_Zg": 1.0 },
  "actor_goal":    { "lambda_g": 0.1 },
  "es_lora":       { "rank": 4, "population_size": 16, "es_interval": 100 },
  "self_play_league": { "enabled": false, ... }
}
```

Self-play policy snapshots are written under `policy_versions/` when `self_play_league.enabled` is true.

## Visualizing a Checkpoint

```bash
pulsar-viz --config /path/to/checkpoint/config.json --checkpoint /path/to/checkpoint --device cpu
```
