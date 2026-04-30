# Pulsar

`Pulsar` is an LFPO-native Rocket League bot training stack built around a high-throughput C++ runtime and a thin Python visualization layer.

The training workflow has two stages:

1. `pulsar_lfpo_pretrain` trains a separate transformer future evaluator on replay trajectory windows and sparse terminal outcome labels, then behavior-clones the Continuum actor while training its unconditional action-conditioned latent future predictor.
2. `pulsar_lfpo_train` runs self-play only. It samples candidate actions, predicts each candidate's future embedding, scores those embeddings with the frozen/target evaluator, computes relative latent advantages, and applies the clipped LFPO policy update.

The actor remains the Continuum recurrent memory architecture, but its heads are only `pi(a|s)` and `F(s,a)`. The transformer future evaluator is a separate model and is refreshed online from completed self-play windows only.

## Design Goals

- LFPO as the only training path
- Sparse terminal outcomes as the only ground-truth reward signal
- Separate transformer future evaluator with fixed horizons `[8, 32, 96]`
- Continuum actor with policy and latent future prediction heads
- ROCm-friendly Linux build path

## Repository Layout

- `cpp/`: runtime, LFPO models, training code, tests, and benchmarks
- `python/pulsar_viz/`: visualization and evaluation package
- `configs/`: LFPO experiment configs
- `scripts/`: setup, preprocessing, smoke tests, and utility scripts
- `docs/`: platform-specific notes such as ROCm setup
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

## Validation

```bash
ctest --test-dir build/release --output-on-failure
ctest --test-dir build/release -L smoke --output-on-failure
./build/release/pulsar_bench
```

## Core Binaries

- `pulsar_lfpo_pretrain`: offline LFPO pretraining
- `pulsar_lfpo_train`: online LFPO self-play training
- `pulsar_bench`: LFPO model throughput benchmark
- `pulsar_native`: Python extension used by visualization
- `pulsar-viz`: Python CLI for checkpoint playback

## Typical Workflow

### 1. Build An Offline Dataset

The offline pretrainer consumes schema v4 tensor manifests. Shards contain observations, optional behavior actions or action probabilities, terminal outcome labels, outcome-known masks, trajectory starts, and end flags. Future windows are built on the fly during training.

```bash
.venv/bin/python scripts/download_kaggle_dataset.py \
  --output /path/to/high-level-rocket-league-replay-dataset

.venv/bin/python scripts/preprocess_kaggle_2v2.py \
  /path/to/high-level-rocket-league-replay-dataset \
  /path/to/pulsar_offline_2v2
```

Set `offline_dataset.train_manifest` and `offline_dataset.val_manifest` in `configs/2v2_offline.json`.

### 2. Run Offline LFPO Pretraining

```bash
./build/release/pulsar_lfpo_pretrain configs/2v2_offline.json /path/to/offline_outputs
```

This writes the actor checkpoint at the output root and the target future evaluator under `future_evaluator/`.

### 3. Run Online LFPO

Set `lfpo.init_checkpoint` in `configs/2v2_lfpo.json` to the offline output directory, then launch:

```bash
./build/release/pulsar_lfpo_train configs/2v2_lfpo.json /path/to/run_outputs
```

To run a bounded smoke or evaluation training slice:

```bash
./build/release/pulsar_lfpo_train configs/2v2_lfpo.json /path/to/run_outputs 100
```

Self-play policy snapshots are written under `policy_versions/` when `self_play_league.enabled` is true.

### 4. Visualize A Checkpoint

```bash
pulsar-viz \
  --config /path/to/checkpoint/config.json \
  --checkpoint /path/to/checkpoint \
  --device cpu
```

The Python side loads the LFPO config, loads the native Continuum actor, builds an evaluation environment, and runs a visualization episode through `RLViser` or `RocketSimVis`.
