#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate.sh"
exec python experiments/run_rt_sys.py --platform agx --algebra-backend torch "$@"
