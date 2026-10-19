#!/usr/bin/env bash
# Run our RT/SYS engine and display, with an optional registered-xApp experiment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate-gpu.sh"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
live_view=integrated
live_test=1
sim_args=()
while (($#)); do
    case "$1" in
        --read-only) live_test=0; shift ;;
        --view)
            if (($# < 2)); then printf '%s\n' '--view requires integrated, telemetry, or scene' >&2; exit 2; fi
            live_view="$2"; shift 2 ;;
        *) sim_args+=("$1"); shift ;;
    esac
done
case "$live_view" in integrated|telemetry|scene) ;; *) printf 'Unknown view: %s\n' "$live_view" >&2; exit 2 ;; esac
demo_dir="$PWD/outputs/live_project/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$demo_dir"
printf 'Actual project live run: %s\n' "$demo_dir"
test_args=()
if ((live_test)); then test_args+=(--live-xapp-test); fi
bash scripts/orin/run.sh --scene munich --cells 3 --ues 12 --slots 400 \
    "${test_args[@]}" \
    --recording-mode experiment --diagnostic-window-slots 50 \
    "${sim_args[@]}" --output "$demo_dir" --live-state "$demo_dir/live.json" >"$demo_dir/simulation.log" 2>&1 &
sim_pid=$!
cleanup() {
    if kill -0 "$sim_pid" 2>/dev/null; then
        kill "$sim_pid" 2>/dev/null || true
        wait "$sim_pid" 2>/dev/null || true
        python - "$demo_dir" <<'PY'
import json, sys, time
from pathlib import Path
for name in ('live.json', 'run_status.json'):
    path = Path(sys.argv[1]) / name
    if path.exists():
        data = json.loads(path.read_text())
        if data.get('stage') not in ('completed', 'failed'):
            data.update(stage='stopped', updated_unix_s=time.time())
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
PY
    fi
}
trap cleanup EXIT
.venv-orin-gui/bin/python scripts/orin/project_live_viewer.py --state "$demo_dir/live.json" --view "$live_view" 2>&1 | tee "$demo_dir/viewer.log"
