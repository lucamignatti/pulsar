# Pulsar

Rocket League bot training with **APPO**, **contrastive goal-conditioned auxiliary learning**, **all-mode self-play**, **PCGrad**, and **slow ES-LoRA**.

The training loop combines three complementary signals:

1. **APPO**: the main local optimizer.
2. **Contrastive goal-conditioned auxiliary**: self-supervised future-goal encoders (state-action and goal) trained with symmetric InfoNCE. Provides a representation-learning signal that never shapes the environment reward.
3. **Slow Rank-4 ES-LoRA**:  periodic global parameter-space search over a LoRA adapter on the final policy layer, driven by rollout reward with KL penalty.

The production setup trains 1v1, 2v2, and 3v3 from the start using one stable reward function. The curriculum layer is still available, but the default config uses it only as a mode-allocation wrapper rather than a staged reward schedule.

## Architecture

- **Encoder**: **Mamba-2**
- **Contrastive goal-conditioned encoder pair**: state-action encoder and goal encoder trained with symmetric InfoNCE
- **Rank-4 LoRA adapter** on the final policy layer: mutated by ES, also trainable by APPO
- **ES-LoRA** periodic global optimizer: population sampling, antithetic evaluation, reward-based fitness
- **Self-play league** with ELO ratings, snapshot management, and periodic evaluation against past policies
- **All-mode training** with PCGrad over per-mode minibatch groups
- 
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

## Training

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs
```

To run a bounded number of updates:

```bash
./build/release/pulsar_appo_train configs/2v2_appo.json /path/to/run_outputs 100
```

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
