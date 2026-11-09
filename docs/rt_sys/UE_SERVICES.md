# UE power, connected DRX and task services

This extension works with both `--model full` and `--model fast`. The accepted
Munich launcher still uses the original six UEs (0, 1, 2, 4, 6, 8), three cells,
10 MHz and depth-2 RT. Its configuration now enables these system models.
It does not add another waveform simulator, radio resource grid, or RT request.

## Sources and modeling boundary

- [Sionna SYS 2.2](https://nvlabs.github.io/sionna/sys/index.html) documents PHY
  abstraction, link adaptation, power control and scheduling. Its documented
  API does not supply the UE power/DRX/MEC components used here. Existing
  Sionna RT and NR/SYS transmission calculations remain the radio substrate.
- [3GPP TS 38.321, Release 17, section 5.7](https://www.etsi.org/deliver/etsi_ts/138300_138399/138321/17.13.00_60/ts_138321v171300p.pdf)
  defines connected DRX monitoring, on-duration, inactivity and HARQ-related
  activity. Our slot-clock model follows the long-cycle/on-duration and
  inactivity concepts, with a conservative active guard while HARQ, handover or
  upload/SR work is pending. It does not implement all symbol-level timers,
  short-cycle transitions, MAC CEs, RRC signaling, or CSI masking in that spec.
- [OAI NR UE scheduler](https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c)
  was inspected as a possible implementation reference. No OAI code is copied
  or executed. The implementation is a system-level extension, not an OAI or
  Sionna DRX implementation or a claim of standards conformance.
- FIFO computation and transport queues implement explicit service budgets.
  They are engineering models, not a supplied Sionna/OAI MEC package.

The existing ideal-CSI/reference-measurement assumption remains. References
can still be computed during sleep; they are measurement hypotheses, not extra
radiated data or a measurement of handset consumption. DRX here controls MAC
service availability; it is not a complete UE receiver/measurement simulator.

## Radio activity and energy

Sleeping/waking UEs cannot receive DL grants. An arriving DL packet remains in
the existing MAC/HARQ ledger: sleep never deletes offered traffic. Successful
new-grant opportunities extend the inactivity deadline; retransmissions and
handover are conservatively protected. The receiver wakes ahead of the next
on-duration. Pending uploads model the corresponding uplink activity request.

Energy is integrated in joules from actual slot duration, active/sleep/wake
state, scheduled receive airtime, abstract uploaded bits, local CPU work and a
discrete wake transition cost. Radio and CPU components are additive. An idle
UE consumes its specified idle/sleep power. Retransmissions consume receive
energy without counting payload a second time. This is distinct from the
existing gNB radiated-RF counter; neither is whole-device gNB energy.

The example powers (sleep 0.04 W, monitor 0.25 W, receive 0.9 W, upload 1.4 W,
CPU idle 0.05 W and CPU dynamic 0.8 W) and wake cost (0.2 mJ) are **uncalibrated
scenario assumptions**, not measurements or normative 3GPP power values.
Use them for controlled sensitivity/comparison experiments; calibrate them
before making absolute handset battery-life claims. `receive_w` and
`transmit_w` are total radio-mode powers; increments above monitor power are
weighted by airtime. Independent abstract UL/DL service can overlap and the
incremental energy model reflects that abstraction, not a modeled TDD pattern.

## MEC task pipeline

Task generation has a separate seeded RNG and never perturbs the original
traffic generator. Each task is routed either to a per-UE local FIFO, or to:

1. A finite shared logical uplink budget for the current serving cell.
2. One-way backhaul delay and a FIFO at the selected MEC server.
3. A bounded GFLOP service budget, then return backhaul delay.
4. Result packets added to the existing DL ledger with a task identity.

Uploads are paused when disconnected or when the serving RT reference has no
path; other eligible UEs can still use the shared budget. Uploads move to the
new radio cell after handover; the already chosen MEC server remains pinned.
The uplink is a configured-capacity transport abstraction, not an NR UL BLER,
grant, interference or TDD simulation. Result delivery is the actual existing
DL NR/SYS path and competes with ordinary traffic under the same scheduler.

A local task completes only after its required computation. An offloaded task
completes only when all of its own result bytes are delivered; unrelated
goodput cannot complete it. Duplicate HARQ ACKs cannot double-count it. Queue
overflow is a task failure. Overdue pending tasks remain visible and do not
count as successes. At all times:

`generated = completed + failed + pending`.

Success is on-time completed tasks divided by generated tasks. Task E2E delay
includes upload/transport, queueing, compute and returned DL data as applicable;
it is separate from the original packet-completion delay. A returned result
received within a slot still contributes that final slot to task delay.
Events are quantized to slots, and stages cannot spend the same interval twice.
Outstanding tasks are bounded per UE across all stages to bound runtime memory.

The example adds 20 tasks/s to UE0 and UE4: 1,000-byte input, 200-byte result,
0.05 GFLOP and 100 ms deadline. Local capacity is 2 GFLOPS/UE, server capacity
40 GFLOPS/cell, shared abstract UL capacity 5 Mbit/s/cell and one-way backhaul
2 ms. All are declared scenario parameters. Default offload probability is
0.5. Original business traffic and its offered load remain separate and intact.

## Configuration and controllers

```sh
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --slots 400
# Controlled legacy-equivalent radio/traffic baseline:
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --slots 400 --ue-services off
# Passive accounting, without DRX or task traffic:
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --slots 400 --ue-services energy
# Explicit complete preset (also works with --model full):
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --slots 400 --ue-services drx-mec
```

Use fresh `--output` directories. The full preset has a 40 ms DRX cycle,
4 ms on-duration, 8 ms inactivity and 0.5 ms wake interval. DRX is disabled on
UE2 and UE6 because their existing control-flow delay budgets are 10 ms.
The default six-UE config embeds its resolved parameters. Other scenarios are
unchanged unless explicitly enabled; CLI presets retain only configured UEs.

`ConfigAction.ue_drx_configs` maps UE IDs to validated DRX fields;
`ue_offload_fraction` maps task-enabled UE IDs to probabilities in [0,1].
Actions follow the existing arbitration, cell-scope checks, delayed actuation
and transaction validation. Out-of-scope, disabled-model or malformed actions
reject the whole transaction. New policies affect subsequently generated tasks
and do not retroactively reroute in-flight work. These are system-model control
interfaces, not a claim that these knobs are standard E2SM controls. No new xApp
algorithm is registered in the Chongyu pool by this change.

Delayed public reports expose `ue_energy_total_j`, `ue_power_w`,
`ue_radio_state`, `ue_active_fraction`, `ue_task_metrics` and `ue_service_policy`.
Energy/active fraction/task counters are cumulative; instantaneous power and
radio state describe the produced slot. Missing task models report unavailable
metrics rather than an invented success rate. The replay adds UE energy, radio
state, task counts and task delay, with final-counter consistency audits.

## Validation and cost

Tests cover energy/time units, idle DRX savings, pending-HARQ protection, real
scheduler eligibility, CPU/UL service budgets, task/byte conservation, deadline
misses, result loss, result-specific and duplicate-ACK accounting, upload path
loss, handover continuity, scoped actions and delayed telemetry. Full NR
integration fixtures exercise the task result path; the performance benchmark
uses the real Munich scene and accepted fast profile on AGX1.

Run `python scripts/validation/benchmark_ue_services.py --output
outputs/ue_services_acceptance_TIMESTAMP` from the AGX GPU environment. It uses
400 slots, warms geometry once, reverses the order across two repeats and
compares OFF, passive energy and DRX+MEC. Every measured case exports/audits its
HTML. Passive accounting must preserve all packet and HARQ event values. DRX
and MEC intentionally change service availability and offered result traffic,
so their resulting KPI/time changes are not interpreted as implementation-only
overhead. `module_wall_s` measures the extension's own hooks; it excludes the
additional generic report serialization and trace-output cost.

## AGX1 acceptance, 2026-11-03

83 targeted/regression tests passed. Real Munich, six UEs, 400 slots
(0.2 s), seed 1, two repeats in reversed order, warmed identical geometry.
All six HTML data audits passed; passive energy preserved all packet and HARQ
event values. Mean main-loop times: OFF 19.200 s, energy 19.622 s (+2.20%),
DRX+MEC 19.710 s (+2.66%). Direct module-hook time for DRX+MEC was 0.093 s.
Mean full internal times including export were 27.766 / 28.223 / 27.975 s.
Peak main-process RSS increased about 13.4 MB; measured output size increased
from 3.59 to 4.58 MB (+27.5%). These are short-run observations, not a guarantee
for higher task loads or long outage queues. See `ue_services_acceptance.json`
for the source fingerprint and machine-readable acceptance.
