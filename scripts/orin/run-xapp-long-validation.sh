#!/usr/bin/env bash
# Run one durable queue of frozen xApp evaluations in this project copy.
set -euo pipefail
oran_queue_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$oran_queue_root"
source scripts/orin/activate-gpu.sh
exec python -m experiments.run_xapp_long_validation "$@"
