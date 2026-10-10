#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export DRJIT_CACHE_DIR="$PWD/.cache/drjit"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCH_HOME="$PWD/.cache/torch"
mkdir -p "$CUDA_CACHE_PATH"
if [[ -n "${ORAN_PYTHON:-}" ]]; then
  python_runtime="$ORAN_PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  python_runtime=python3.11
else
  python_runtime=python3
fi
"$python_runtime" -c 'import sys,platform; assert sys.version_info >= (3,11); assert platform.machine() in ("aarch64","arm64") and platform.system()=="Linux"'
if [[ ! -d .venv-spark ]]; then "$python_runtime" -m venv .venv-spark; fi
.venv-spark/bin/python scripts/spark/install_environment.py
.venv-spark/bin/python scripts/spark/preflight.py --lock
.venv-spark/bin/python experiments/run_rt_sys.py --slots 40 --cells 2 --ues 4
.venv-spark/bin/python experiments/validate_rt_sys_mu.py
