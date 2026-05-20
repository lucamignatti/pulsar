# Pulsar

Rocket League bot training with **Asyncronous PPO**, **contrastive goal-conditioned auxiliary learning**, **all-mode self-play**, **PCGrad**, and **slow ES-LoRA**.

The training loop combines three complementary signals:

1. **APPO**: the main local optimizer.
2. **Contrastive goal-conditioned auxiliary**: self-supervised future-goal encoders (state-action and goal) trained with symmetric InfoNCE. Provides a representation-learning signal that never shapes the environment reward.
3. **Slow Rank-4 ES-LoRA**:  periodic global parameter-space search over a LoRA adapter on the final policy layer, driven by rollout reward with KL penalty.

The production setup trains 1v1, 2v2, and 3v3 using curriculum learning. It is able to achieve 188k collection SPS, 155k update SPS and 133K overall SPS on a 7900x+6800xt. 

## Architecture

- **Encoder**: **Mamba 2**
- **Contrastive goal-conditioned encoder pair**: state-action encoder and goal encoder trained with symmetric InfoNCE
- **Evolutionary Sampling** on the final policy layer via EGGROLL
- **Multi-mode training** with PCGrad over per-mode minibatch groups
- **Custom CUDA/HIP kernels** for mamba2, ppo, and other functions
- **Sharded collection** and **Distributed micro-batches** for efficient multi-gpu scaling
- **Overlap** between collection and update
- **half-stepping** per-shard for simultaneous action computation/env stepping during collection
  
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


## Works Cited

- (Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality)[https://arxiv.org/abs/2405.21060]
- (Loss of plasticity in deep continual learning)[https://www.nature.com/articles/s41586-024-07711-7]
- (Self-Supervised Goal-Reaching Results in Multi-Agent Cooperation and Exploration)[https://arxiv.org/abs/2509.10656v1]
- (Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning)[https://arxiv.org/abs/2509.10656v1]
- (Evolution Strategies at the Hyperscale)[https://arxiv.org/abs/2511.16652]

