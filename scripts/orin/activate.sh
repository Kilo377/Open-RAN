#!/usr/bin/env bash
oran_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$oran_project_root/.venv-orin/bin/activate"
export MPLCONFIGDIR="$oran_project_root/.cache/matplotlib"
export DRJIT_CACHE_DIR="$oran_project_root/.cache/drjit"
# JetPack 6 and 7 ship different LLVM versions. Respect a caller's valid path.
if [[ ! -f "${DRJIT_LIBLLVM_PATH:-}" ]]; then
    unset DRJIT_LIBLLVM_PATH
    for oran_llvm_lib in /usr/lib/aarch64-linux-gnu/libLLVM-*.so* \
                         /usr/lib/aarch64-linux-gnu/libLLVM.so*; do
        if [[ -f "$oran_llvm_lib" ]]; then
            export DRJIT_LIBLLVM_PATH="$oran_llvm_lib"
            break
        fi
    done
    unset oran_llvm_lib
fi
# Conda's older libstdc++ may not load the system LLVM library.
if [[ -f /usr/lib/aarch64-linux-gnu/libstdc++.so.6 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$oran_project_root/scripts/orin/runtime${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$oran_project_root"
unset oran_project_root
