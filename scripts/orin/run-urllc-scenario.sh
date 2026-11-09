#!/usr/bin/env bash
# Dedicated URLLC experiment defaults; the general simulator remains native.
set -euo pipefail
oran_urllc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_urllc_root"
oran_has_duration=false
oran_has_scenario=false
for oran_arg in "$@"; do
    case "$oran_arg" in
        --slots|--slots=*|--duration-s|--duration-s=*) oran_has_duration=true ;;
        --urllc-scenario|--urllc-scenario=*|--traffic-profile|--traffic-profile=*) oran_has_scenario=true ;;
    esac
done
if ! "$oran_has_duration"; then set -- --duration-s 3.2 "$@"; fi
if ! "$oran_has_scenario"; then set -- --urllc-scenario burst "$@"; fi
exec bash scripts/orin/run-six-ue-cuda-nr.sh \
    --model fast --traffic-load 2 \
    --xapp-pool chongyu --ue-services drx-mec --record-compression gzip \
    --urllc-xapp-policy-version qos_v2 "$@"
