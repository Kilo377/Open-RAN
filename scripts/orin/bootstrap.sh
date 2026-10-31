#!/usr/bin/env bash
set -euo pipefail
oran_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_project_root"
# Preserve the validated Conda interpreter on older devices; JetPack 7 has 3.12.
oran_python="${ORAN_PYTHON:-python3}"
if [[ -z "${ORAN_PYTHON:-}" && -x "$HOME/anaconda3/bin/python" ]]; then
    oran_python="$HOME/anaconda3/bin/python"
fi
"$oran_python" -m venv .venv-orin
oran_requirements=requirements-orin.lock.txt
if [[ -f /etc/nv_tegra_release ]] && grep -q '^# R39 ' /etc/nv_tegra_release; then
    oran_requirements=requirements-orin-jp7.lock.txt
fi
.venv-orin/bin/python -m pip install --no-cache-dir --timeout 120 -r "$oran_requirements"
unset oran_python oran_requirements
source scripts/orin/activate.sh
python -m pip check
python scripts/spark/preflight.py --cpu-local
source scripts/orin/activate-gpu.sh
if ! python scripts/orin/preflight.py; then
    printf '%s\n' 'CUDA kernels did not pass preflight; scripts/orin/run.sh will use CPU.' >&2
fi
