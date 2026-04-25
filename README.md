# Pulsar

`Pulsar` is a modular Rocket League bot training stack built around a high-throughput C++ runtime and a thin Python visualization/evaluation layer.

The current training path has two stages:

1. Offline pretrain a shared policy with behavior cloning, value pretraining, and a next-goal predictor on replay-derived tensor manifests.
2. Run synchronous PPO self-play online, using the pretrained next-goal checkpoint as the reward path and optionally exporting fresh online NGP data for refresh.

The codebase is centered on a shared model stack, hard action masking, GPU execution, and a lightweight Python surface for inspection and playback.

## Design Goals

- Fast vectorized C++ trainer
- Shared C++ backbone for model architecture
- No hand-written reward function
- ROCm-friendly Linux build path

## Repository Layout

- `cpp/`: runtime, model, training code, tests, and benchmarks
- `python/pulsar_viz/`: visualization and evaluation package
- `configs/`: shared experiment configs
- `scripts/`: setup, preprocessing, smoke tests, and utility scripts
- `docs/`: platform-specific notes such as ROCm setup
- `external/RocketSim/`: vendored RocketSim submodule

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
pip install -e '.[viz,offline]'
python3 scripts/collision_mesh_downloader.py
```

If you only need the Python visualization environment, `scripts/setup_python_dev.sh` is a shorter shortcut. It bootstraps a Python 3.12 virtualenv and installs the viz extras.

Build the project:

```bash
cmake -S . -B build/release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build build/release --parallel
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
ctest --test-dir build/release -L benchmark --output-on-failure
ctest --test-dir build/release -L rocm --output-on-failure
./build/release/pulsar_bench
```

`pulsar_bench` reports both collector-only throughput and a trainer-like collection loop that uses randomly initialized learner and NGP models, so the benchmark includes model inference overhead in addition to RocketSim stepping.

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

This writes `config.json`, `metadata.json`, `model.pt`, optimizer snapshots, and `offline_metrics.jsonl` into the output directory.

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
  --checkpoint /path/to/checkpoint \
  --device cpu
```

You can also target `RocketSimVis` instead of `RLViser`:

```bash
pulsar-viz \
  --config configs/2v2_ppo.json \
  --checkpoint /path/to/checkpoint \
  --device cpu \
  --renderer rocketsimvis
```

To write a video file from the actual `RLViser` window on macOS, add `--video-out`:

```bash
pulsar-viz \
  --config configs/2v2_ppo.json \
  --checkpoint /path/to/checkpoint \
  --device cpu \
  --video-out /path/to/eval.mp4
```

`--video-out` currently requires `--renderer rlviser`. The recorder captures the real RLViser window via macOS screen capture, so the terminal or Python process needs Screen Recording permission, and window inspection also requires Accessibility permission.

The Python side stays intentionally thin: it loads the shared config, loads the native model, builds an evaluation environment, and runs a visualization episode through `RLViser` or `RocketSimVis`.
