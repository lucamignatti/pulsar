# Linux + ROCm Notes

`Pulsar` targets Linux for real training runs. The intended deployment stack is:

- `ROCm` installed on the host
- ROCm-enabled `libtorch`
- `RocketSim` available to the C++ build
- Python package extras for `rlgym`, `rocketsim`, and `rlviser-py`

## Important Device Note

PyTorch ROCm builds still use the `cuda` device namespace in both Python and C++ APIs.

- Use `"device": "cuda"` in experiment configs even on AMD/ROCm machines.
- In C++, `torch::Device("cuda")` is still the correct device selector for ROCm-enabled builds.

## Suggested Build Flow

1. Install ROCm on the Linux host and verify it with the ROCm tooling.
2. Initialize the vendored `RocketSim` submodule with `git submodule update --init --recursive`.
3. Download the soccar collision meshes with `python3 scripts/collision_mesh_downloader.py`.
4. Install a ROCm-enabled PyTorch build in the virtualenv.
5. Use `python -c 'import torch; print(torch.utils.cmake_prefix_path)'` as the Torch `CMAKE_PREFIX_PATH`.
6. Create a Python 3.12 or 3.13 virtualenv and install the visualization dependencies.
7. Configure and build the project in `RelWithDebInfo`.

Example:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install torch pybind11
pip install -e .[viz]

git submodule update --init --recursive
python3 scripts/collision_mesh_downloader.py

cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)"

cmake --build build/release --parallel
```

## Expected External Inputs

`Pulsar` vendors `RocketSim` as a git submodule, but a production build still needs:

- `torch` with ROCm support
- optional `pybind11` if you do not let CMake fetch it

## Validation Targets

After dependencies are installed, the minimum validation pass should include:

```bash
ctest --test-dir build/release --output-on-failure
./build/release/pulsar_bench
```

If the Torch targets are enabled, also validate:

- `pulsar_train` can start and write checkpoints
- `pulsar_native` builds
- the Python visualizer can load a checkpoint produced by the trainer
