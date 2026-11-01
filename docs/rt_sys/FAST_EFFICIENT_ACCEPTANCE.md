# Efficient fast profile — 2026-11-01

Historical efficient-profile acceptance. The current fast default is described
in [system profile acceptance](FAST_SYSTEM_ACCEPTANCE.md).

`--model fast` now selects `efficient`. The previous fast computation remains
available with `--model fast --fast-profile reference`. `full` remains the default
model and retains its existing actual NR execution. JSON configuration uses
`rt_sys.fast_profile`; CLI overrides it.

Efficient uses four channel frequency samples per PRB and one time coordinate
per slot within each exact co-scheduled stream pattern. RZF/LMMSE uses CPU small
matrices. SINRs are expanded onto the original resource grid before the original
data/DMRS mask and NR TBS are applied. Official Sionna EESM/BLER calculations are
batched on CPU, with cached NR transport-block segmentation; unused SYS random
and decoded-bit outputs are omitted. Missing/nonfinite table entries still use
actual NR on the configured fallback device.

No changes were made to RT geometry, slot duration, scheduling, packet generation,
queues, HARQ identities, OLLA, RF power accounting, control admission or actuation.
The DGX/AGX switch still chooses OptiX/LLVM independently. Saved configuration,
performance, live telemetry and automatic HTML identify the profile and sampling.
Historical fast exports and live comparison specifications retain reference semantics.

The production implementation was tested in an isolated directory on the new
Orin AGX, with Sionna 2.2.0, CUDA 13.2 and Torch 2.12. Runs were serial, using the
same prewarmed RT cache: Munich, three cells, six moving UEs, 24 PRBs, RT depth 2,
400 slots of 0.5 ms (0.2 s simulated), identical traffic and seed.

| Implementation | Launcher wall time | Simulation loop |
| --- | ---: | ---: |
| Previous fast (`reference`) | 144.98 s | 134.39 s |
| New fast (`efficient`) | 71.19 s | 60.17 s |

The total-time speedup is 2.04×, a 50.90% reduction. Each clean implementation
exactly reproduced its corresponding isolated experiment: per-UE packet metrics,
link statistics, packet/HARQ event hashes and RF energy. The new and previous
profiles also matched those event streams in this scenario.

The preceding study tested three seeds and doubled traffic with power, PRB quota
and sleep/wake stimuli. Its largest effective-SINR difference was 0.105 dB;
packet/HARQ events and KPI counters matched. This is finite scenario evidence,
not mathematical equivalence or a guarantee for every channel/xApp configuration.
The inherited repeated expired target-state submissions were reproduced rather
than changed by this optimization.

The clean implementation also repeated seed 2 and the doubled-load control case;
both exactly matched their experimental packet/HARQ events, link statistics,
per-UE packet metrics, RF energy and meaningful control receipts. The broad
regression passed 136 tests. A final 51-test pass covered CPU/CUDA SYS probability
equivalence, fast HARQ, sampling/resource accounting, replay, CuPy and the existing
execution optimizations. Offline browser checks passed at 320–1440 px with zero
external requests or JavaScript errors.

Detailed production acceptance checks and source hashes are recorded in
[`fast_efficient_acceptance.json`](fast_efficient_acceptance.json). Raw runs remain
under `validation_candidates/fast_promote_20261101/outputs/acceptance/` on the new
AGX. Each completed run includes its self-contained `network-replay.html`.
