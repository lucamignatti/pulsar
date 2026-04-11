#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch pybind11
pip install -e .[viz]

echo "Python environment ready at ${ROOT_DIR}/.venv"
