# Linux + CUDA Notes

`Pulsar` targets NVIDIA CUDA systems for production APPO training. H100 is the high-throughput target, but CUDA validation runs on any CUDA-capable GPU. The intended deployment stack is:

- NVIDIA driver with CUDA 12.x support
- CUDA-enabled PyTorch
- `RocketSim` available to the C++ build
- Python package extras for `rlgym`, `rocketsim`, and `rlviser-py`

## Suggested Build Flow

1. Install a current NVIDIA driver and CUDA 12.x toolkit on the host.
2. Verify CUDA visibility with `nvidia-smi`.
3. Initialize the vendored `RocketSim` submodule with `git submodule update --init --recursive`.
4. Download the soccar collision meshes with `python3 scripts/collision_mesh_downloader.py`.
5. Install a CUDA-enabled PyTorch build in the virtualenv.
6. Use `python -c 'import torch; print(torch.utils.cmake_prefix_path)'` as the Torch `CMAKE_PREFIX_PATH`.
7. Configure and build the project in `RelWithDebInfo`.

Example:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install pybind11
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

## Runtime Defaults

- Use `ppo.device = "cuda"` or `"cuda:0"` in APPO configs.
- CUDA pinned host buffers are enabled for collector-to-GPU transfers when the runtime device is CUDA.
- The CUDA smoke test enables TF32 matmul and cuDNN paths through PyTorch when supported before running the BC pretrain/APPO train slice.

## Expected External Inputs

`Pulsar` vendors `RocketSim` as a git submodule, but a production build still needs:

- CUDA-enabled `torch`
- optional `pybind11` if you do not let CMake fetch it
- the `collision_meshes/` tree populated before running RocketSim-backed tests or binaries

## Validation Targets

After dependencies are installed, the minimum validation pass should include:

```bash
ctest --test-dir build/release --output-on-failure
ctest --test-dir build/release -L cuda --output-on-failure
./build/release/pulsar_bench
```

If the Torch targets are enabled, also validate:

- `pulsar_bc_pretrain` writes the Continuum actor checkpoint.
- `pulsar_appo_train` loads that checkpoint via `ppo.init_checkpoint` and runs on `cuda:0`.
- The CUDA smoke test is skipped only when no CUDA device is available or PyTorch is not CUDA-enabled.
- Python bindings are optional. If `Python3 Development.Module` is unavailable, CMake skips `pulsar_native` without blocking the C++ trainer build.
