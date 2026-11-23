# Group-wide RCP response identification

Treat the UE population as the monitored system. A group intervention changes
one common control for every applicable group member, while the external
environment and all other controls remain paired. It does not require enabling
DRX on traffic whose native profile disables it, or creating MEC tasks on UEs
without that workload.

All memberships are derived from the actual scenario and run-local service
profile, pinned in the plan and checked again by the simulator before launch.
No member is selected based on observed KPI performance.

| RCP | Group action | Context held fixed |
|---|---|---|
| MEC offloading probability | All task sources: 0.4 / 0.5 / 0.6 | Native RR and DRX, active cells |
| DRX inactivity timer | All DRX-enabled users: 8 → 4 ms | Cycle/on-duration and native disabled profiles unchanged |
| MEC scheduling priority | Entire MEC group: 0.75 / 1 / 1.25 | Latency-aware scheduler, other traffic weights 1 |
| URLLC scheduling priority | Entire URLLC group: 0.75 / 1 / 1.25 | QoS-deadline scheduler, other traffic weights 1 |
| Scheduler policy | All cells: RR → latency-aware / QoS-deadline | All weights 1 |
| Cell sleep mode | All cells: active → micro-sleep | Existing 1 ms per 20 ms profile |

The scheduling scores consume multiplicative weights. Scaling every user's
weight by the same positive factor preserves ranking; it does not establish a
useful resource-priority effect. Therefore the two priority probes vary a
complete service-class group relative to other traffic. They remain distinct
contexts on the same RCP node. A unit check verifies uniform QoS-weight scaling
preserves order using the actual scheduler score function.

Two seeds × eleven arms × 0.8 s = 22 runs, 17.6 s simulated. Native warmup is
100 ms; ordinary RIC actuation delay is one slot. Scores cover `(200,700] ms`,
with terminal task follow-up to 800 ms. Moving UEs, native 1× traffic, the
accepted fast system PHY, depth-2 RT, radio capacity and deadlines are held
unchanged. Learned xApps/rApps stay off and their model hashes are pinned.

Each contrast changes one RCP. Group action vectors pass through ordinary
admission, transaction validation and delayed execution. The analyzer checks
all 1,000 scoring slots against the complete held vector and rejects failed
controls. It also checks paired native prefixes, exogenous arrivals, task
births, shared channel cache keys, packet-byte conservation, terminal cohorts,
completed status and every checkpoint file hash.

Energy sums all UEs. MEC demand sums all task sources. Mean UE goodput includes
zero-rate users; completion delays and success ratios pool event counts. The
independent replication unit is the seed pair, not a slot or packet. Numerical
responses use symmetric central differences and retain the center-linearity
diagnostic. DRX/scheduler/sleep actions use named finite contrasts. Two-seed
Student intervals are descriptive and can be very wide; zero observed change
does not establish no effect at another load or horizon.

The coefficient is for the named common group action. It is not a Jacobian for
arbitrary independent per-UE actions. Existing deployed xApp algorithms and
their contracts are unchanged by the experiment driver.

```bash
python experiments/run_rcp_perturbation.py --population \
  --output outputs/population-rcp --parallel 2
python experiments/run_rcp_perturbation.py --population \
  --output outputs/population-rcp --resume
python -m smo.identify --view population --study outputs/population-rcp \
  --output outputs/population-identification
python -m smo --view population \
  --replay outputs/population-rcp/cases/s20261116_native \
  --influences outputs/population-identification/influences.json
```

Use a separate workspace/output directory; preserve the original study and
trained models. The runner stores an immutable plan, all raw traces, automatic
network replay HTMLs, per-case file-hash checkpoints and resumable progress.
Retain channel caches as well. The complete response matrix includes zeros
and uncertain outcomes; graph links retain scope, context, units and evidence.
