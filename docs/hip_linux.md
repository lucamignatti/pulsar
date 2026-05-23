# Linux + AMD HIP ROCm Notes

`Pulsar` targets AMD ROCm systems using the HIP language for high-throughput production VRPO training. The setup supports AMD GPUs (e.g., Radeon RX 6000/7000 series, Instinct MI200/MI300 series). The intended deployment stack is:

- AMD ROCm Driver and Toolkit (version 6.x recommended)
- ROCm-enabled PyTorch
- C++ build targeting AMD GPU architectures
- `RocketSim` available to the C++ build
- Python package extras for `rlgym`, `rocketsim`, and `rlviser-py`

## Suggested Build Flow

1. Install the AMD ROCm driver and toolkit on the host. Make sure `/opt/rocm/bin` is added to your `PATH` so `hipcc` is visible.
2. Verify GPU visibility with `rocm-smi` or `rocminfo`.
3. Initialize the vendored `RocketSim` submodule with `git submodule update --init --recursive`.
4. Download the soccar collision meshes with `python3 scripts/collision_mesh_downloader.py`.
5. Install a ROCm-enabled PyTorch build in the virtualenv.
6. Configure and build the project in `RelWithDebInfo`. You can specify your GPU architecture (e.g., `gfx1030` for RX 6800XT, `gfx1100` for RX 7900XTX) via the `CMAKE_HIP_ARCHITECTURES` flag if necessary.

Example:

```bash
python3.12 -m venv .venv
. .venv/bin/activate

# Install ROCm 6.1 compatible PyTorch (verify your ROCm version first)
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch
pip install pybind11
pip install -e .[viz]
pip install -e .[offline]

git submodule update --init --recursive
python3 scripts/collision_mesh_downloader.py

# Configure C++ build with HIP enabled
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython3_EXECUTABLE="$(which python)" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030  # Adjust for your GPU model

cmake --build build/release --parallel
```

## Runtime Defaults

- Use `ppo.device = "cuda"` or `"cuda:0"` (PyTorch exposes ROCm/HIP devices through the `"cuda"` namespace for transparent API compatibility).
- Pinned host buffers are enabled for memory copies to/from the GPU.
- Environment variables like `ROC_SERIALIZATION=0` or `HIP_VISIBLE_DEVICES` can be used to manage multi-GPU sharding.

## Validation Targets

After dependencies are installed, the minimum validation pass should include:

```bash
ctest --test-dir build/release --output-on-failure
./build/release/pulsar_bench 20 configs/2v2_appo.json hip:0
```

If the Torch targets are enabled, also validate:

- `pulsar_vrpo_train` runs end-to-end with the ROCm/HIP device (the `cuda_smoke` test exercises the transparent CUDA-namespace device execution with downsized parameters).
- Python bindings are optional. If `Python3 Development.Module` is unavailable, CMake skips `pulsar_native` without blocking the C++ trainer build.
