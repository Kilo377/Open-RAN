#!/usr/bin/env bash
# Selected AGX execution profile for the six mobile UEs and native network.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate-gpu.sh"
exec python experiments/run_rt_sys.py --platform agx --scene munich --cells 3 --ues 6 \
    --config configs/rt_sys/agx_munich_6ue_depth2_fast.json --slots 400 \
    --algebra-backend torch --recording-mode experiment --diagnostic-window-slots 40 "$@"
