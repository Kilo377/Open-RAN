#!/usr/bin/env bash
# Run frozen X01 and experimental X12 through the native RIC control path.
# X12: 3.90% latency improvement in validation; the 5% efficacy target is unmet.
# See outputs/models/latency_scheduler_rf/deployment.json for acceptance status.
set -euo pipefail
oran_xapp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
oran_energy_model="$oran_xapp_root/outputs/models/ue_energy_rf/model.json"
oran_latency_model="$oran_xapp_root/outputs/models/latency_scheduler_rf/model.json"
for oran_model_file in "$oran_energy_model" "$oran_latency_model"; do
    if [[ ! -s "$oran_model_file" ]]; then
        printf '%s\n' "Missing deployed RF model: $oran_model_file" >&2
        exit 2
    fi
done
exec bash "$oran_xapp_root/scripts/orin/run-six-ue-cuda-nr.sh" \
    --model fast --xapp-pool chongyu --ue-services drx-mec \
    --energy-xapp-mode frozen --energy-xapp-model "$oran_energy_model" \
    --latency-xapp-mode frozen --latency-xapp-model "$oran_latency_model" "$@"
