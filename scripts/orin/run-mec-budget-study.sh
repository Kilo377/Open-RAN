#!/usr/bin/env bash
# Finite measured QGBDT training and matched native/random-mitigation study.
set -euo pipefail
oran_budget_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_budget_root"
source "$oran_budget_root/scripts/orin/activate-gpu.sh"
exec python experiments/run_mec_budget_conflict_study.py "$@"
