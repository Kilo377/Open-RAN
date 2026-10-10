#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export DRJIT_CACHE_DIR="$PWD/.cache/drjit"
python_runtime="${ORAN_PYTHON:-python3.11}"
"$python_runtime" -c 'import sys,platform; assert sys.version_info >= (3,11); assert platform.machine() in ("aarch64","arm64") and platform.system()=="Linux"'
if [[ ! -d .venv-spark ]]; then "$python_runtime" -m venv --system-site-packages .venv-spark; fi
.venv-spark/bin/python -m pip install -r requirements-rt-sys.txt
.venv-spark/bin/python scripts/spark/preflight.py --lock
.venv-spark/bin/python experiments/run_rt_sys.py --slots 40 --cells 2 --ues 4
.venv-spark/bin/python experiments/validate_rt_sys_mu.py
