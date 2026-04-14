# Pulsar

`Pulsar` is a modular Rocket League bot platform built around a high-throughput C++ training runtime and a thin Python evaluation/visualization layer.

The current training path is a two-stage pipeline: offline pretrain a shared policy plus next-goal head on tensorized replay data, then run synchronous PPO self-play online using the pretrained next-goal checkpoint as the only reward signal. The runtime keeps hard-legality action masking, frozen-policy self-play with Elo evaluation, expanded per-update timing metrics, and GPU execution.

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

cmake -S . -B build/release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build build/release
```

If `Torch` or Python binding dependencies are not available, CMake still builds the core environment/config targets and skips the trainer/bindings targets. If ROCm is installed and discoverable, the ROCm libraries and ROCm-only tests are enabled automatically. Pass `-DPULSAR_DISABLE_ROCM=ON` to force a CPU-only build on a ROCm host.

For the intended deployment target, see [docs/rocm_linux.md](docs/rocm_linux.md).

## Validation

Use this as the normal post-build validation path:

```bash
ctest --test-dir build/release --output-on-failure
```

That default suite covers:

- core config, environment, done, and action-mask tests
- offline dataset and offline-pretrain smoke coverage
- PPO math, batched collector, next-goal reward, and self-play coverage
- train/offline/benchmark/preprocess binary smokes
- ROCm smoke automatically on ROCm builds

Useful focused invocations:

```bash
ctest --test-dir build/release -L reference --output-on-failure
ctest --test-dir build/release -L smoke --output-on-failure
ctest --test-dir build/release -L rocm --output-on-failure
./build/release/pulsar_bench
```

## Targets

- `pulsar_core`: Config, action tables, done conditions, mutators, and environment scaffolding.
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
- per-update metrics also include stage timings such as `obs_build_seconds`, `mask_build_seconds`, `policy_forward_seconds`, `env_step_seconds`, `done_reset_seconds`, `reward_model_seconds`, `gae_seconds`, and `ppo_forward_backward_seconds`
- `./build/<preset>/pulsar_bench <num_envs> [collection_workers]` lets you sweep arena count and collection parallelism independently.

## Online PPO Notes

- Hard-legality action masking is always enabled in the online trainer.
- `reward.ngp_checkpoint` is required. `pulsar_train` fails fast if it is empty.
- Online reward is always `reward.ngp_scale * (ngp_t - ngp_{t-1})` from the pretrained next-goal head.
- The reward model is a frozen copy of the shared actor/critic/NGP architecture. PPO trains the live policy/value model, while the frozen copy is only used for next-goal reward inference.
- Trainer metrics and checkpoint metadata record which NGP checkpoint was active, its source checkpoint step/update, and the current NGP promotion index.
- `reward.online_dataset` can export on-policy trajectories as NGP-only tensor manifests while PPO is running. This is intended for training a continuously updated candidate NGP on fresh bot gameplay.
- `reward.refresh` still supports direct checkpoint promotion at update boundaries, but the recommended automation path is the external dynamic refresh loop below.
- Self-play snapshots are stored under `policy_versions/` in the run directory when `ppo.self_play.enabled` is true.
- Only learner-controlled agents contribute rollout entries and `global_step` during self-play episodes.
- Mixed precision is not part of the current runtime. The trainer runs in FP32 on both CPU and ROCm.

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
  --workers 8 \
  --interpolation rocketsim \
  --action-target-mode weighted \
  --max-update-age 0.0
```

For throughput, `--workers` lets the preprocessor fan out across multiple CPU processes. `--interpolation rocketsim` gives the highest-fidelity reconstruction but is the slowest mode; `linear` and then `none` are progressively faster if you can tolerate lower replay reconstruction fidelity. `--action-target-mode best` is also cheaper than `weighted`.

Then point [configs/2v2_offline.json](configs/2v2_offline.json) at those manifests and run:

```bash
./build/release/pulsar_offline_train configs/2v2_offline.json /path/to/offline_outputs
```

That produces:

- `policy/` checkpoint directory compatible with the existing shared C++ model loader
- `next_goal/` checkpoint directory containing the NGP model plus the same observation normalizer state
- `offline_metrics.jsonl`

For dynamic NGP refreshes, the intended pattern is:

- keep the active online reward model frozen
- export fresh on-policy NGP data with `reward.online_dataset`
- continue fine-tuning a separate candidate NGP from the active checkpoint with `behavior_cloning.enabled = false`
- mix fresh online data with anchor replay data to avoid catastrophic forgetting
- evaluate active vs candidate on anchor and recent validation manifests
- promote only at PPO update boundaries after the candidate clears the promotion thresholds

The controller script for that loop is:

```bash
.venv/bin/python scripts/ngp_refresh_loop.py \
  --train-binary ./build/release/pulsar_train \
  --offline-binary ./build/release/pulsar_offline_train \
  --ppo-config configs/2v2_ppo.json \
  --offline-config configs/2v2_offline.json \
  --run-root /path/to/dynamic_refresh_run \
  --total-updates 1000
```

By default it:

- runs PPO in one-update chunks
- keeps the active reward NGP fixed within each chunk
- fine-tunes the candidate NGP from the current active checkpoint instead of retraining from scratch
- trains the candidate on a 70/30 mix of new online data and anchor replay data
- promotes when recent validation loss improves enough without anchor validation loss regressing past the configured tolerance

If you want to launch a one-off manual refresh yourself, `next_goal_predictor.init_checkpoint` can warm-start `pulsar_offline_train` from an existing `next_goal/` checkpoint. With `next_goal_predictor.reuse_normalizer = true`, the offline trainer keeps using the checkpoint's observation normalizer instead of refitting it.

The offline trainer now runs truncated BPTT over real per-player trajectories. `behavior_cloning.sequence_length` controls the chunk length used for recurrent updates.

To start PPO from the offline-pretrained policy instead of random initialization, set `ppo.init_checkpoint` in your PPO config to the offline policy checkpoint directory, for example `/path/to/offline_outputs/policy`.

To use the trained next-goal predictor online, set:

- `reward.ngp_checkpoint = /path/to/offline_outputs/next_goal`
- `reward.ngp_scale = 1.0`

PPO training now writes:

- periodic `update_N/` checkpoints
- `best/` for the highest rollout reward seen so far
- `final/` for the final checkpoint at the end of training
