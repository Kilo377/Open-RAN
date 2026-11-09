#!/usr/bin/env bash
# Frozen X01 + MEC budget QGBDT, with per-UE random offload arbitration.
set -euo pipefail
oran_mec_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
oran_mec_energy_model="$oran_mec_root/outputs/models/ue_energy_rf/model.json"
oran_mec_budget_model="$oran_mec_root/outputs/models/mec_offload_budget_qgbdt/model.json"
for oran_mec_model in "$oran_mec_energy_model" "$oran_mec_budget_model"; do
    if [[ ! -s "$oran_mec_model" ]]; then
        printf '%s\n' "Missing frozen model: $oran_mec_model" >&2
        exit 2
    fi
done
exec bash "$oran_mec_root/scripts/orin/run-six-ue-cuda-nr.sh" \
    --model fast --xapp-pool chongyu --ue-services drx-mec \
    --energy-xapp-mode frozen --energy-xapp-model "$oran_mec_energy_model" \
    --budget-xapp-mode frozen --budget-xapp-model "$oran_mec_budget_model" \
    --direct-conflict-mitigation uniform_random \
    --mitigation-seed 1701 --mitigation-window-ms 100 "$@"
