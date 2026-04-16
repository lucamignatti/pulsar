# Pulsar

`Pulsar` is a modular Rocket League bot platform built around a high-throughput C++ training runtime and a thin Python evaluation and visualization layer.

The current training loop has two stages:

1. Offline pretrain a shared policy and next-goal head on replay-derived tensor data.
2. Run synchronous PPO self-play online, using the pretrained next-goal checkpoint as the reward model.

The project is organized around a single shared model stack, hard action masking, GPU execution, and a lightweight Python surface for inspection and playback.

## Design Goals

- Fast vectorized C++ trainer
- Shared C++ backbone for model architecture
- No hand-written reward function
- Good performance

## Repository Layout

- `cpp/`: runtime, model, training code, tests, and benchmarks
- `python/`: thin visualization package and CLI
- `configs/`: shared experiment configs
- `scripts/`: setup, preprocessing, and utility scripts
- `docs/`: platform-specific notes such as ROCm setup

## Requirements

- CMake 3.25+
- A C++20 compiler
- Python 3.10-3.13
- The `RocketSim` submodule
- `torch` and `pybind11` for the full trainer and Python bindings
- `.[viz]` extras for visualization
- `.[offline]` extras for replay preprocessing and offline pretraining

## Setup

Initialize dependencies:

```bash
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch pybind11
pip install -e .[viz]
pip install -e .[offline]
python3 scripts/collision_mesh_downloader.py
```

If you only need the Python visualization environment, `scripts/setup_python_dev.sh` is a shorter shortcut.

Build the project:

```bash
cmake -S . -B build/release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build build/release
```

Notes:

- If `torch` or Python binding dependencies are missing, CMake still builds the core-only targets and skips trainer or binding targets.
- ROCm is enabled automatically when detected. Use `-DPULSAR_DISABLE_ROCM=ON` to force a CPU-only build on ROCm hosts.
- For the intended deployment target, see [docs/rocm_linux.md](docs/rocm_linux.md).

## Validation

Run the default test suite after building:

```bash
ctest --test-dir build/release --output-on-failure
```

Useful focused commands:

```bash
ctest --test-dir build/release -L reference --output-on-failure
ctest --test-dir build/release -L smoke --output-on-failure
ctest --test-dir build/release -L rocm --output-on-failure
./build/release/pulsar_bench
```

## Core Binaries

- `pulsar_offline_train`: offline pretraining
- `pulsar_train`: online PPO training
- `pulsar_bench`: runtime throughput benchmark
- `pulsar_native`: Python extension used by the visualization layer
- `pulsar-viz`: Python CLI for checkpoint playback

## Typical Workflow

### 1. Build an Offline Dataset

The offline trainer consumes tensor manifests, not raw replay files. For the Kaggle high-level replay dataset, the normal path is:

```bash
.venv/bin/python scripts/download_kaggle_dataset.py \
  --output /path/to/high-level-rocket-league-replay-dataset

.venv/bin/python scripts/preprocess_kaggle_2v2.py \
  /path/to/high-level-rocket-league-replay-dataset \
  /path/to/pulsar_offline_2v2
```

Update `configs/2v2_offline.json` so `offline_dataset.train_manifest` and `offline_dataset.val_manifest` point at the generated manifests.

### 2. Run Offline Pretraining

```bash
./build/release/pulsar_offline_train configs/2v2_offline.json /path/to/offline_outputs
```

This writes an offline checkpoint directory plus `offline_metrics.jsonl`.

### 3. Run Online PPO

Set `reward.ngp_checkpoint` in `configs/2v2_ppo.json` to the offline checkpoint you want to use for reward inference, then launch training:

```bash
./build/release/pulsar_train configs/2v2_ppo.json /path/to/run_outputs
```

You can optionally pass a third argument to limit the number of PPO updates:

```bash
./build/release/pulsar_train configs/2v2_ppo.json /path/to/run_outputs 100
```

The shipped `configs/2v2_ppo.json` enables online NGP dataset export and in-process NGP refresh by default. When `reward.online_dataset.output_dir` is left empty, `pulsar_train` writes the online NGP shards under `online_ngp/` inside the run directory.

If `ppo.self_play.enabled` is true, self-play policy snapshots are written under `policy_versions/` inside the run directory.

### 4. Visualize a Checkpoint

```bash
pulsar-viz \
  --config configs/2v2_ppo.json \
  --checkpoint /path/to/checkpoint/model.pt \
  --device cpu
```

The Python side stays intentionally thin: it loads the shared config, loads the native model, builds an evaluation environment, and runs a visualization episode through `RLViser`.

## Optional Tools

- `wandb` logging is configured through the `wandb` block in the JSON experiment configs.
- `scripts/ngp_refresh_loop.py` orchestrates a longer-running reward-refresh loop around `pulsar_train` and `pulsar_offline_train` if you want automated NGP promotion.
