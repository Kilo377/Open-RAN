#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/activate.sh"
oran_project_root="$PWD"
# The selected launcher decides which computations use CUDA.
# lib64 resolves to aarch64-linux on JetPack 6 and sbsa-linux on JetPack 7.
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_CACHE_PATH="$oran_project_root/.cache/cuda-orin"
export CUPY_CACHE_DIR="$oran_project_root/.cache/cupy"
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
unset oran_project_root
