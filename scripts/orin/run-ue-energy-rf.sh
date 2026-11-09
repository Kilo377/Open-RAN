#!/usr/bin/env bash
# Accepted six-UE network with the deployed, frozen X-01 forest.
set -euo pipefail
oran_energy_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
oran_energy_model="$oran_energy_root/outputs/models/ue_energy_rf/model.json"
if [[ ! -s "$oran_energy_model" ]]; then
    printf '%s\n' "No accepted RF model at $oran_energy_model; run the online study and deploy its accepted artifact first." >&2
    exit 2
fi
exec bash "$oran_energy_root/scripts/orin/run-six-ue-cuda-nr.sh" \
    --model fast --ue-services drx-mec --energy-xapp-mode frozen \
    --energy-xapp-model "$oran_energy_model" "$@"
