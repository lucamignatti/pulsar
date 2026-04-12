# Pulsar

`Pulsar` is a modular Rocket League bot platform built around a high-throughput C++ training runtime and a thin Python evaluation/visualization layer.

## Design Goals

 - Fast vectorized C++ trainer
 - Shared C++ backbone for model architecture
 - No handmade reward function
 - Good performance


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
- `wandb`
- `rlgym-tools`, `pandas`, and `pyarrow` for replay preprocessing

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

For the intended deployment target, see [docs/rocm_linux.md](docs/rocm_linux.md).

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
- shard-local `obs.pt`, `actions.pt`, `action_probs.pt`, `next_goal.pt`, `weights.pt`, and `episode_starts.pt` tensor files

The preprocessor now uses `rlgym-tools` replay parsing and `replay_to_rlgym`, so the offline dataset is built from native `ReplayFrame` objects rather than hand-decoded replay tensors. That gives it:

- exact `DefaultObs(zero_padding=2)` observations aligned with the online schema
- direct `next_scoring_team` next-goal labels
- replay-native continuous controls mapped onto the 90-action lookup table with `pick_action`
- soft discrete policy targets in `action_probs.pt`

For accuracy, it filters out interpolated car updates by default with `--max-update-age 0.0` and splits trajectories whenever an agent becomes stale or a kickoff/game boundary is reached.

Recommended invocation:

```bash
.venv/bin/python scripts/preprocess_kaggle_2v2.py \
  /path/to/high-level-rocket-league-replay-dataset \
  /path/to/pulsar_offline_2v2 \
  --interpolation rocketsim \
  --action-target-mode weighted \
  --max-update-age 0.0
```

Then point [configs/2v2_offline.json](configs/2v2_offline.json) at those manifests and run:

```bash
./build/release/pulsar_offline_train configs/2v2_offline.json /path/to/offline_outputs
```

That produces:

- `policy/` checkpoint directory compatible with the existing shared C++ model loader
- `next_goal/` checkpoint directory containing the NGP model plus the same observation normalizer state
- `offline_metrics.jsonl`

The offline trainer now runs truncated BPTT over real per-player trajectories. `behavior_cloning.sequence_length` controls the chunk length used for recurrent updates.

To start PPO from the offline-pretrained policy instead of random initialization, set `ppo.init_checkpoint` in your PPO config to the offline policy checkpoint directory, for example `/path/to/offline_outputs/policy`.

To use the trained next-goal predictor as the online reward, set:

- `reward.mode = "ngp"`
- `reward.ngp_checkpoint = /path/to/offline_outputs/next_goal`
- `reward.shaped_scale = 0.0`
- `reward.ngp_scale = 1.0`

PPO training now writes:

- periodic `update_N/` checkpoints
- `best/` for the highest rollout reward seen so far
- `final/` for the final checkpoint at the end of training
