# Pulsar

Sparse-reward Rocket League bot training with **Contrastive Goal-Conditioned APPO** and **Slow ES-LoRA**.

The training loop combines three complementary signals:

1. **Sparse APPO** — the main local optimizer, trained from terminal win/loss/touch outcomes only.
2. **Contrastive goal-conditioned auxiliary** — self-supervised future-goal encoders (state-action and goal) trained with symmetric InfoNCE. Provides a representation-learning signal without shaping the environment reward.
3. **Slow Rank-4 ES-LoRA** — periodic global parameter-space search over a LoRA adapter on the final policy layer, driven by true sparse winrate with KL penalty.

The real environment reward remains strictly sparse terminal win/loss/touch. The goal-conditioning machinery is an auxiliary self-supervised target, never the environment reward.

## Architecture

- **SWA Transformer encoder** with scalar value head
- **Contrastive goal-conditioned encoder pair** — state-action encoder and goal encoder trained with symmetric InfoNCE
- **Rank-4 LoRA adapter** on the final policy layer — mutated by ES, also trainable by APPO
- **ES-LoRA** periodic global optimizer: population sampling, antithetic evaluation, sparse-outcome fitness
- Optional **PCGrad** gradient projection and **success behavior cloning**

## Repository Layout

- `cpp/`: C++20 runtime, SWA Transformer model, APPO training loop, batched collector, tests, benchmarks
- `configs/`: experiment config files (JSON)
- `scripts/`: smoke tests, W&B streaming, utility scripts
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
./build/release/pulsar_bench 20 configs/2v2_appo.json cuda:0
```

## Core Binary

- `pulsar_appo_train`: online sparse APPO + goal-conditioned auxiliary + ES-LoRA self-play training

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
  "outcome":       { "score": 1.0, "concede": -1.0, "neutral": 0.0, "neutral_no_touch": -1.0 },
  "goal_mapping":  { "arena_max_distance": 8192.0 },
  "goal_critic":   { "goal_dim": 3, "hidden_dim": 256, "embedding_dim": 64, "logsumexp_penalty_coeff": 0.01, "lambda_Zg": 1.0, "lambda_goal_actor": 0.1, "contrastive_batch_size": 2048, "max_future_horizon": 256 },
  "es_lora":       { "rank": 4, "lora_alpha": 4.0, "population_size": 16, "sigma_ES": 0.01, "eta_ES": 0.003, "es_interval": 100, "beta_KL": 1.0, "antithetic_sampling": true, "require_winrate_signal": true, "min_winrate_std": 1e-6 },
  "self_play_league": { "enabled": false, ... }
}
```

Self-play policy snapshots are written under `policy_versions/` when `self_play_league.enabled` is true.

## Visualizing a Checkpoint

```bash
pulsar-viz --config /path/to/checkpoint/config.json --checkpoint /path/to/checkpoint --device cpu
```
