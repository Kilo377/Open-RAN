#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/activate.sh"
oran_project_root="$PWD"
# Sionna/Torch stay on CPU; CuPy executes the dense algebra on Orin CUDA.
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_CACHE_PATH="$oran_project_root/.cache/cuda-orin"
export CUPY_CACHE_DIR="$oran_project_root/.cache/cupy"
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
unset oran_project_root
