# Pulsar

`Pulsar` is a modular Rocket League bot platform built around a high-throughput C++ training runtime and a thin Python evaluation/visualization layer.

## Design Goals

- Mirror the `RLGym` component model in C++ so environment logic stays composable.
- Run the full `PPO` training loop in C++ with `libtorch`.
- Use `RocketSim` as the high-throughput simulation backend.
- Keep one model definition in C++ and expose that same implementation to Python.
- Use Python only for checkpoint-driven evaluation and `RLViser` rendering.

## Repository Layout

- `cpp/`: C++ runtime, environment modules, model, trainer, tests, and benchmarks.
- `python/`: Thin visualization package and CLI.
- `configs/`: Shared JSON experiment definitions.
- `docs/`: Platform-specific setup notes.
- `scripts/`: Local helper scripts for environment setup.

## Build

Initialize the `RocketSim` submodule first:

```bash
git submodule update --init --recursive
```

Then fetch the collision meshes used by both C++ `RocketSim` and the Python `rocketsim` package:

```bash
python3 scripts/collision_mesh_downloader.py
```

The project expects:

- Python 3.10-3.13
- `torch`
- `pybind11`
- `rlgym[rl-rlviser]`
- `rocketsim`
- `subtr-actor-py` for replay preprocessing

Configure with:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch pybind11
pip install -e .[viz]
pip install -e .[offline]

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build build
```

If `Torch` or Python binding dependencies are not available, CMake still builds the core environment/config targets and skips the trainer/bindings targets.

For the intended deployment target, see [docs/rocm_linux.md](/Users/lucamignatti/Projects/pulsar/docs/rocm_linux.md). On ROCm machines, keep using the `cuda` device string in the config because PyTorch ROCm exposes the CUDA device namespace.

## Targets

- `pulsar_core`: Config, action tables, rewards, done conditions, mutators, and environment scaffolding.
- `pulsar_torch`: Shared actor-critic model, normalization, GPU rollout storage, and PPO trainer.
- `pulsar_train`: Standalone trainer entry point.
- `pulsar_offline_train`: Standalone offline pretrainer for behavior cloning and next-goal prediction.
- `pulsar_native`: Python extension exposing the C++ model and checkpoint helpers.
- `pulsar_bench`: Lightweight benchmark target for core runtime throughput. It reports both env-steps/sec and agent-steps/sec so comparisons against `RLGym`-style vectorized trainers stay apples-to-apples.

## W&B Logging

Both `pulsar_train` and `pulsar_offline_train` can stream metrics to Weights & Biases while still writing the local JSONL logs.

Set the `wandb` block in your config:

```json
"wandb": {
  "enabled": true,
  "project": "pulsar",
  "entity": "",
  "run_name": "",
  "group": "ppo",
  "job_type": "ppo_train",
  "dir": "",
  "mode": "online",
  "python_executable": "python3",
  "script_path": "scripts/wandb_stream.py",
  "tags": ["ppo", "continuum", "dppo"]
}
```

Notes:

- `run_name` can be left empty; Pulsar will use the output/checkpoint directory name.
- `dir` can be left empty; Pulsar will use the run output directory for local `wandb` files.
- `mode` can be `online`, `offline`, or `disabled`.
- The logger uses the same config JSON you pass to the C++ executable, so the run config appears in W&B automatically.

## Python Visualization

The Python package is intentionally thin:

- Loads the same JSON config as the C++ trainer.
- Loads the shared model through the native extension.
- Builds an `RLGym` evaluation environment.
- Runs checkpoint-driven visualization episodes via `RLViser`.

The visualization path is aligned with current `RLGym` defaults:

- `RepeatAction(LookupTableAction(), repeats=tick_skip)`
- `DefaultObs(zero_padding=team_size)`
- `RLViserRenderer(tick_rate=tick_rate / tick_skip)`

## Throughput Notes

- `ppo.collection_workers = 0` means auto-size the collection thread pool from hardware concurrency.
- `collection_env_steps_per_second` counts one arena step as one step, regardless of team size.
- `collection_agent_steps_per_second` multiplies by the number of controlled cars. This is the closest comparison to the thesis repo's aggregate "steps per second" metric, which is driven by batched active-agent observations across many environments.
- `./build/<preset>/pulsar_bench <num_envs> [collection_workers]` lets you sweep arena count and collection parallelism independently.

## Offline Pretraining

The offline stage consumes tensor shards, not raw replay files directly. The intended path for the Kaggle high-level replay dataset is:

1. Extract the dataset locally.
2. Or download/materialize it through `kagglehub`:

```bash
.venv/bin/python scripts/download_kaggle_dataset.py \
  --output /path/to/high-level-rocket-league-replay-dataset
```

3. Convert the `2v2` replay split into Pulsar tensor shards:

```bash
.venv/bin/python scripts/preprocess_kaggle_2v2.py \
  /path/to/high-level-rocket-league-replay-dataset \
  /path/to/pulsar_offline_2v2
```

This writes sequence-safe shards:

- `train_manifest.json`
- `val_manifest.json`
- shard-local `obs.pt`, `actions.pt`, `next_goal.pt`, `weights.pt`, and `episode_starts.pt` tensor files

The preprocessor uses `subtr_actor` for frame extraction, preserves per-player trajectories, writes explicit `episode_starts` markers, and keeps whole trajectories inside a shard so recurrent pretraining does not cross replay boundaries. It zero-fills unavailable boost-pad state and derives discrete action labels heuristically from short-horizon car kinematics plus jump state. The resulting labels are good enough for policy warm-starting, but they are not replay-perfect controller reconstruction.

Then point [configs/2v2_offline.json](/Users/lucamignatti/Projects/pulsar/configs/2v2_offline.json) at those manifests and run:

```bash
./build/release/pulsar_offline_train configs/2v2_offline.json /path/to/offline_outputs
```

That produces:

- `policy/` checkpoint directory compatible with the existing shared C++ model loader
- `next_goal/` checkpoint directory containing the NGP model plus the same observation normalizer state
- `offline_metrics.jsonl`

The offline trainer now runs truncated BPTT over real per-player trajectories. `behavior_cloning.sequence_length` controls the chunk length used for recurrent updates.

## Current Status

This repository now contains a working v1 with source-informed defaults:

- The default action space is the 90-action `RLGym` lookup table.
- The default observation shape is `132`, matching `DefaultObs(zero_padding=2)` for `2v2`.
- Reward defaults follow the `RLGym` PPO guide direction more closely: goal event reward, touch reward, non-negative speed-to-ball reward, ball-to-goal shaping, and a small face-ball term.
- Terminal logic includes both goal termination and no-touch truncation.
- The C++ runtime now uses real `RocketSim` arenas and downloaded soccar collision meshes.
- The trainer batches policy inference across many independent arenas, writes checkpoints plus optimizer state, and logs per-update metrics to `metrics.jsonl`.
- Rollout collection now parallelizes observation packing, action decode, stepping, reward computation, and reset checks across arenas while keeping rollout tensors device-resident after ingress.
- The offline stage now includes a tensor-shard dataset format, a C++ pretrainer for behavior cloning plus next-goal prediction, and a Kaggle `2v2` replay preprocessing script.
- The Python evaluator loads the same native C++ model module and refuses config/checkpoint mismatches.

Verified locally in this workspace:

- Core and Torch CMake configure/build/tests pass.
- Offline pretraining smoke tests pass and emit both policy and next-goal checkpoints.
- `pulsar_bench` runs against real `RocketSim`.
- `pulsar_train` can run a smoke PPO update and write a checkpoint.
- Python can load that checkpoint through `pulsar_native`, construct the `RLGym` eval environment, and run a forward pass on live observations.
