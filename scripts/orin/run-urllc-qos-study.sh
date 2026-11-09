#!/usr/bin/env bash
# Native and explicitly labelled equal-mean burst traffic; real online RF.
set -euo pipefail
oran_urllc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_urllc_root"
source "$oran_urllc_root/scripts/orin/activate-gpu.sh"
exec python experiments/run_urllc_qos_study.py "$@"
