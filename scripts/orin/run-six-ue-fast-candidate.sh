#!/usr/bin/env bash
# Optional execution candidate; existing launch defaults remain independent.
set -euo pipefail
oran_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$oran_script_dir/run-six-ue.sh" \
    --config "$oran_script_dir/../../configs/rt_sys/agx_munich_6ue_depth2_fast_candidate.json" \
    --algebra-backend torch --live-xapp-test --diagnostic-window-slots 40 "$@"
