#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate.sh"
if [ -z "${DISPLAY:-}" ]; then
  export DISPLAY=:1
  export XAUTHORITY=/run/user/1000/gdm/Xauthority
  export XDG_RUNTIME_DIR=/run/user/1000
  export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
fi
exec .venv-orin-gui/bin/sionna-rt-gui --config configs/orin/gui.yaml "$@"
