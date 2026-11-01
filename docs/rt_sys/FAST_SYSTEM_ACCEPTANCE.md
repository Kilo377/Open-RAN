# System fast profile — 2026-11-01

Fast now defaults to `system`, with a 2 ms spatial/reference gain update bound.
`--fast-profile efficient` retains the previous per-slot fast model;
`--fast-profile reference` retains the historical full-resource-grid fast model.
Full remains the default model when `--model` is omitted.

The physical abstraction caches absolute RT-derived RZF spatial fields separately
for the actual stream set, resource pattern, beam/codebook and RET state. An
anchor LMMSE combiner provides scalar desired/interfering gains. Every slot uses
the actual per-stream/per-PRB power and RX noise. A changed power state gets its
own combiner; changed stream/beam/RET states get their own spatial fields. Sleep
removes the corresponding emissions. Fields never cross a gain epoch or RT
geometry epoch. No fixed MU penalty, Shannon-capacity substitute, MCS remapping
or fitted KPI multiplier is used.

The official SYS EESM/BLER calculations, NR TBS, explicitly counted actual NR
fallback, MAC/queue/HARQ clocks and RIC admission/actuation remain. The full
resource mask and DMRS/TBS accounting are retained. System currently requires
ideal CSI. Spatial updates are approximate between anchor slots; selecting 0.5 ms
updates refreshes them every slot in the current configuration.

Reference gains are cached before applying current power/sleep settings, so
control effects remain immediate. Measurement filtering, report production and
delayed observations still advance every 0.5 ms. Detached report copies isolate
historical observations. Experiment recording builds diagnostic state snapshots
at window boundaries and the final slot; audit recording retains every snapshot.
Packet/HARQ/control and per-slot KPI traces are not decimated.

New AGX, same prewarmed RT cache, serial runs, automatic HTML enabled: Munich,
three cells, six moving UEs, 24 PRBs, 0.5 ms slots, 400 slots (0.2 s simulated).

| Variant | Launcher total | Main loop |
| --- | ---: | ---: |
| Previous efficient fast | 70.46 s | 59.75 s |
| Reports/snapshots only | 64.68 s | 54.07 s |
| Reference gains only, 5 ms | 60.80 s | 50.27 s |
| Spatial gains only, 5 ms | 53.02 s | 42.44 s |
| Combined, 1 ms | 42.58 s | 32.05 s |
| Combined, 2 ms | 39.84 s | 29.00 s |
| Combined, 5 ms | 37.37 s | 26.79 s |

The selected 2 ms mode is 1.77× faster including startup/export and 2.06× faster
in the main loop. Report-only optimization preserves packet/HARQ events exactly.
The initial 2 ms case preserved all per-UE delivered totals and P95 completion
latencies; approximate modes do not generally preserve individual events.

An additional final-code 0.5 ms run took 49.06 s total (38.34 s main loop) and
reproduced the seed-1 efficient packet/HARQ events exactly. Use
`--fast-gain-update-ms .5` to retain per-slot gain updates. This single-case
result does not establish bitwise equivalence for arbitrary inputs.

Follow-up seeds 2 and 3 preserved every UE's delivered total at 2 ms. P95
completion-latency shifts reached 1 ms; first-TX BLER changed by at most 0.136
percentage points. With doubled traffic and fixed power/quota/sleep stimuli,
network goodput changed by −0.0352%, the largest per-UE goodput change was 0.406%,
RF energy and meaningful control receipts matched, and deadline misses were
1700 versus 1701. The 5 ms option was faster but shifted some P95 latencies by
1.5 ms and changed the stress-case retransmission error rate by 2.71 percentage
points. The inherited repeated expired target-state submissions remain unchanged.

Follow-up wall times may include concurrent regression and are not speed claims.
Seed 2 and stress references are preserved production-efficient runs from the
preceding acceptance; seed 3 includes a fresh efficient reference. These checks
cover this topology/load/velocity and do not establish URLLC tail reliability or
universal xApp equivalence. Long-run timings have not been measured here.

Detailed checks, source hashes, regression and browser results are in
[`fast_system_acceptance.json`](fast_system_acceptance.json). Raw evidence remains
under `validation_candidates/fast_system_20261101/outputs/benchmark/` on the new
AGX. Every successful run includes its self-contained `network-replay.html`.
