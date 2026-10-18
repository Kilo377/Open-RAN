#!/usr/bin/env bash
oran_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$oran_project_root/.venv-orin/bin/activate"
export MPLCONFIGDIR="$oran_project_root/.cache/matplotlib"
export DRJIT_CACHE_DIR="$oran_project_root/.cache/drjit"
export DRJIT_LIBLLVM_PATH=/usr/lib/aarch64-linux-gnu/libLLVM-15.so.1
# Conda's older libstdc++ cannot load the system LLVM 15 library.
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$oran_project_root/scripts/orin/runtime${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$oran_project_root"
unset oran_project_root
