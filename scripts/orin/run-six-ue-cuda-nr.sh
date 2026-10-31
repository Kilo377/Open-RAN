#!/usr/bin/env bash
# Optional CUDA NR + Torch algebra; RT remains on the LLVM CPU backend.
# Requires a PyTorch wheel with working Orin CUDA kernels (e.g. JetPack 7.2).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate-gpu.sh"
exec python experiments/run_rt_sys.py --device cuda --rt-device cpu \
    --scene munich --cells 3 --ues 6 --slots 400 \
    --config configs/rt_sys/agx_munich_6ue_depth2_fast.json \
    --algebra-backend torch --nr-execution batched \
    --recording-mode experiment --diagnostic-window-slots 40 "$@"
