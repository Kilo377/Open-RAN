#!/usr/bin/env bash
# Bounded actual X08 training and RR/neutral/fixed/MLP calibration study.
set -euo pipefail
oran_throughput_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_throughput_root"
source "$oran_throughput_root/scripts/orin/activate-gpu.sh"
exec python experiments/run_throughput_mlp_study.py "$@"
