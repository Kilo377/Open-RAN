#!/usr/bin/env bash
# Explicit alias for the validated preset, now also the default fast channel.
set -euo pipefail
oran_fast_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$oran_fast_script_dir/run-six-ue-cuda-nr.sh" --model fast \
    --config "$oran_fast_script_dir/../../configs/rt_sys/agx_munich_6ue_depth2_fast_10ms.json" "$@"
