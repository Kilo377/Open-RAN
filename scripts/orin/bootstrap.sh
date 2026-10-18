#!/usr/bin/env bash
set -euo pipefail
oran_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_project_root"
# Use Python 3.13 (the installed and validated interpreter on agx1).
"${ORAN_PYTHON:-/home/lboroldn/anaconda3/bin/python}" -m venv .venv-orin
.venv-orin/bin/python -m pip install --no-cache-dir -r requirements-orin.lock.txt
source scripts/orin/activate.sh
python -m pip check
python scripts/spark/preflight.py --cpu-local
source scripts/orin/activate-gpu.sh
if ! python scripts/orin/preflight.py; then
    printf '%s\n' 'CUDA kernels did not pass preflight; scripts/orin/run.sh will use CPU.' >&2
fi
