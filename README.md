# Pulsar

`Pulsar` is a DAPPO-native Rocket League bot training stack built around a high-throughput C++ runtime and a thin Python visualization layer.

The training workflow has two stages:

1. `pulsar_bc_pretrain` behavior-clones the Continuum actor on offline replay data using cross-entropy on action targets, with optional distributional value pretraining on terminal outcome labels.
2. `pulsar_appo_train` runs self-play Distributional APPO. It samples actions from the policy, estimates advantages via GAE on quantile-sampled distributional values, applies adaptive clipping based on critic variance, confidence-weights the PPO objective by inverse distribution entropy, and uses a distributional value loss.

The actor is the Continuum recurrent memory architecture with `pi(a|s)` and categorical distributional `V(s)` heads. Only sparse terminal outcomes (goal scored/conceded) are used as the reward signal.

## Design Goals

- DAPPO (Distributional APPO) with BC pretraining
- Adaptive clipping driven by critic distribution variance
- Confidence-weighted PPO objectives (inverse entropy)
- Quantile-sampled distributional values for GAE
- Sparse terminal outcomes as the only ground-truth reward signal
- Continuum actor with policy and distributional value heads
- CUDA production build path, optimized for H100-class throughput

## Repository Layout

- `cpp/`: runtime, DAPPO models, training code, tests, and benchmarks
- `python/pulsar_viz/`: visualization and evaluation package
- `configs/`: APPO + BC experiment configs
- `scripts/`: setup, preprocessing, smoke tests, and utility scripts
- `docs/`: platform-specific notes such as CUDA setup
- `external/RocketSim/`: vendored RocketSim submodule

## Requirements

- CMake 3.25+
- A C++20 compiler
- Python 3.10-3.13
- The `RocketSim` submodule
- `torch` and `pybind11` for the trainer and Python bindings
- `.[viz]` extras for visualization
- `.[offline]` extras for replay preprocessing and offline pretraining

## Setup

```bash
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch pybind11
pip install -e '.[viz,offline]'
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

For CUDA deployment notes, see [docs/cuda_linux.md](docs/cuda_linux.md).

## Validation

```bash
ctest --test-dir build/release --output-on-failure
ctest --test-dir build/release -L smoke --output-on-failure
ctest --test-dir build/release -L cuda --output-on-failure
./build/release/pulsar_bench
```

## Core Binaries

- `pulsar_bc_pretrain`: offline BC pretraining
- `pulsar_appo_train`: online DAPPO self-play training
- `pulsar_bench`: DAPPO model throughput benchmark
- `pulsar_native`: Python extension used by visualization
- `pulsar-viz`: Python CLI for checkpoint playback

## Typical Workflow

### 1. Build An Offline Dataset

The offline pretrainer consumes schema v4 tensor manifests. Shards contain observations, optional behavior actions or action probabilities, terminal outcome labels, and trajectory boundary flags.

```bash
.venv/bin/python scripts/download_kaggle_dataset.py \
  --output /path/to/high-level-rocket-league-replay-dataset

.venv/bin/python scripts/preprocess_kaggle_2v2.py \
  /path/to/high-level-rocket-league-replay-dataset \
  /path/to/pulsar_offline_2v2
```

Set `offline_dataset.train_manifest` and `offline_dataset.val_manifest` in `configs/2v2_bc.json`.

### 2. Run BC Pretraining

```bash
./build/release/pulsar_bc_pretrain configs/2v2_bc.json /path/to/bc_outputs
```

This writes the actor checkpoint at the output root.

### 3. Run Online DAPPO

Set `ppo.init_checkpoint` in `configs/2v2_appo.json` to the BC output directory, then launch:

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs
```

To run a bounded smoke or evaluation training slice:

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs 100
```

Self-play policy snapshots are written under `policy_versions/` when `self_play_league.enabled` is true.

### 4. Visualize A Checkpoint

```bash
pulsar-viz \
  --config /path/to/checkpoint/config.json \
  --checkpoint /path/to/checkpoint \
  --device cpu
```

The Python side loads the DAPPO config, loads the native Continuum actor, builds an evaluation environment, and runs a visualization episode through `RLViser` or `RocketSimVis`.
