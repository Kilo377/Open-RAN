#!/usr/bin/env bash
# source this file from any working directory; it only affects this shell.
oran_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$oran_project_root/.venv-spark/bin/activate"
export MPLCONFIGDIR="$oran_project_root/.cache/matplotlib"
export DRJIT_CACHE_DIR="$oran_project_root/.cache/drjit"
export CUDA_CACHE_PATH="$oran_project_root/.cache/cuda"
export TORCH_HOME="$oran_project_root/.cache/torch"
export PIP_CACHE_DIR="$oran_project_root/.cache/pip"
mkdir -p "$CUDA_CACHE_PATH"
cd "$oran_project_root"
unset oran_project_root
