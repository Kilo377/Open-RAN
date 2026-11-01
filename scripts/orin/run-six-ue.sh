#!/usr/bin/env bash
# Selected AGX execution profile for the six mobile UEs and native network.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate-gpu.sh"
oran_has_duration=false
for oran_arg in "$@"; do
    case "$oran_arg" in --slots|--slots=*|--duration-s|--duration-s=*) oran_has_duration=true ;; esac
done
if ! "$oran_has_duration"; then set -- --slots 400 "$@"; fi
exec python experiments/run_rt_sys.py --platform agx --scene munich --cells 3 --ues 6 \
    --config configs/rt_sys/agx_munich_6ue_depth2_fast.json \
    --algebra-backend torch --recording-mode experiment --diagnostic-window-slots 40 "$@"
