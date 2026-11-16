# Warm multi-xApp switching

`--live-xapps` loads all six trained Chongyu models once, initially OFF.
Each UI button independently enables or disables its xApp. Several instances,
including the MEC and URLLC X-12 versions, can run concurrently. Disabling one
retains the other selections and their generation tokens. Re-enabling reuses
the loaded learner, while invalidating old observation windows and actions.
The network, queues, mobility and simulation clock continue without a restart.

The default direct-conflict policy is seeded `uniform_random`, applied to an
offload probability or a complete scheduler-mode/UE-weight bundle. Disjoint
controls execute together. `--direct-conflict-mitigation priority` is also
available. The MEC scheduler explicitly fills non-target UE weights with their
neutral value only in this live mode so its existing policy is an atomic
scheduler candidate. This does not change its learned objective or model.

## Start

Use the current six-UE scenario. The bundled release is SHA-verified and
automatically installed under `outputs/trained_xapps/20261109` on first use.
There is no package installation or global environment modification.

```bash
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --duration-s 10 \
  --live-xapps --xapp-pool chongyu --ue-services drx-mec \
  --ap-services-profile configs/rt_sys/ap_services_reference.json \
  --xapps-start x01,x13 \
  --output outputs/hot-demo --live-state outputs/hot-demo/live.json
```

For the existing integrated GUI, use the same flags with
`bash scripts/orin/live.sh --live-xapps --duration-s 10 ...`.
Use a fresh output directory for each finite run. DGX can use the same
`experiments/run_rt_sys.py` flags with `--platform dgx` and the six-UE config.
`--xapp-model-dir` selects a different complete, verified release installation.
The legacy `--live-xapp-test` mode keeps its original one-at-a-time behavior.

IDs: `x01`, `x02`, `x08`, `x12_mec`, `x12_urllc`, `x13`; implementation names
and paper-style IDs are accepted too. `--xapps-start all` enables all six.
X-08 and X-12-URLLC need All cells because their objective and observations
cover the whole network. Other controllers can be selected per cell.
Missing UE/MEC or AP physical models make the corresponding buttons unavailable
with an explicit reason. They are not silently enabled by the switching UI.

## Change a running simulation

The UI uses the same local, run-scoped command channel as this stdlib-only CLI:

```bash
python scripts/orin/xapp-control.py --state outputs/hot-demo/live.json enable x01 x08 x13
python scripts/orin/xapp-control.py --state outputs/hot-demo/live.json disable x01
python scripts/orin/xapp-control.py --state outputs/hot-demo/live.json set x08 x12_urllc
python scripts/orin/xapp-control.py --state outputs/hot-demo/live.json disable_all
```

An enable adds instances; a disable removes only the specified instances.
`set` atomically replaces the enabled set. A batch with an unknown or unavailable
instance is rejected as a whole. Old-run requests are rejected. Admission is
distinct from actual execution: normal RIC ticks, arbitration and delayed
actuation still apply. ON identifies an enabled contender, not a guarantee that
its request won. Per-app execution details show version-matching receipts.
When no contender remains, the usual withdrawal restores the original fields.

For reproducible time-based switching, add
`--xapp-toggle-timeline configs/rt_sys/hot_xapps_timeline.example.json` to a run
of at least 0.8 s. `at_ms` is simulation time, aligned to slots and strictly
before the run end. The JSON supports `enable`, `disable`, `set`, `disable_all`,
and optional `cell`. The example covers all-six concurrency and full withdrawal.

## Retained evidence

Models stay frozen: online collection/refitting remains an explicit separate
experiment. Each hot run retains:

- `xapp_model_checkpoints/`: exact six JSON models, also containing measured
  training rows; verified unchanged at run end.
- `xapp_hot_start.json` and optional `xapp_switch_timeline.json`: model SHA,
  activation settings and the original timeline.
- `xapp_test_events.jsonl`: complete admitted/rejected switch history and active
  sets; `xapp_selection` and `control_receipts` record actual RIC/executor events.
- Existing per-app measurements, network/packet/task traces, final summaries,
  live segments and automatic self-contained `network-replay.html`.

Live before/after segments are observational; changing position, traffic and
queues can also change KPIs. The GUI's matched OFF/ON replay starts fresh arms
with the same exogenous inputs and frozen models. Model training and capability
availability do not establish a fresh KPI improvement for a new checkpoint.

The [2026-11-16 validation record](hot_xapps_validation_20261116.json) covers
273 affected regression tests, three CLI default checks and three actual AGX
runs. The final 1,600-slot random-mitigation run exercised all six controllers,
including rapid OFF/ON, stale-action rejection and restoration of the original
control fields. Raw records and all six per-run model checkpoints are retained
on the AGX and in a SHA-verified Mac archive; the record includes its checksum.

## Cache cleanup

`python scripts/orin/clean-project-caches.py` previews explicit rebuildable
project caches; add `--apply` to delete them. Active project Python processes
block cleanup. All run/model data under `outputs/`, `artifacts/`, `data/`,
`datasets/`, `results/` and `checkpoints/` are protected and their inventories
are checked before and after. Run-local channel caches inside `outputs/` remain
available for future training. Environments and global caches are untouched.
