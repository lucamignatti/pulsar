# Linux + ROCm Notes

`Pulsar` targets Linux for real training runs. The intended deployment stack is:

- `ROCm` installed on the host
- ROCm-enabled `torch`
- `RocketSim` available to the C++ build
- Python package extras for `rlgym`, `rocketsim`, and `rlviser-py`

## Suggested Build Flow

1. Install ROCm on the Linux host and verify it with the ROCm tooling.
2. Initialize the vendored `RocketSim` submodule with `git submodule update --init --recursive`.
3. Download the soccar collision meshes with `python3 scripts/collision_mesh_downloader.py`.
4. Install a ROCm-enabled PyTorch build in the virtualenv.
5. Use `python -c 'import torch; print(torch.utils.cmake_prefix_path)'` as the Torch `CMAKE_PREFIX_PATH`.
6. Create a Python 3.12 or 3.13 virtualenv and install the project dependencies.
7. Configure and build the project in `RelWithDebInfo`.

Example:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install torch pybind11
pip install -e .[viz]
pip install -e .[offline]

git submodule update --init --recursive
python3 scripts/collision_mesh_downloader.py

cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"

cmake --build build/release --parallel
```

`Pulsar` will enable ROCm support automatically when the ROCm packages are discoverable during CMake configure. To disable ROCm explicitly on a ROCm host, pass:

```bash
-DPULSAR_DISABLE_ROCM=ON
```

## Expected External Inputs

`Pulsar` vendors `RocketSim` as a git submodule, but a production build still needs:

- `torch` with ROCm support
- optional `pybind11` if you do not let CMake fetch it
- the `collision_meshes/` tree populated before running RocketSim-backed tests or binaries

## Validation Targets

After dependencies are installed, the minimum validation pass should include:

```bash
ctest --test-dir build/release --output-on-failure
ctest --test-dir build/release -L rocm --output-on-failure
./build/release/pulsar_bench
```

If the Torch targets are enabled, also validate:

- The intended two-stage path works on ROCm: `pulsar_lfpo_pretrain` writes the Continuum actor plus `future_evaluator/`, then `pulsar_lfpo_train` loads that checkpoint via `lfpo.init_checkpoint`.
- The trainer runs in FP32 on ROCm. Mixed precision is not part of the current runtime.
- ROCm-only tests are compiled automatically when ROCm is found at configure time.
- Python bindings are optional. If `Python3 Development.Module` is unavailable, CMake skips `pulsar_native` without blocking the C++ trainer build.
