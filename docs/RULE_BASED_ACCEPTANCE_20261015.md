# Rule-based xApp short acceptance and long-run handoff

Source fingerprint: `789b8d11b86139a8c993c8ada1a5b3008a408801facdf7aa06b299461e013c04`.
Implementation commits: `2c812d3`, `af45999`.

## Completed before long-run launch

- Spark: 342 tests passed, plus 2 subtests (59.95 seconds).
- New library: 43 local deterministic/known-channel tests passed.
- 44 real RT/NR short cases passed, 128 slots each, on both maps.
- Four disabled-feature comparisons with frozen original source passed:
  schedule/MCS/ACK/TB/delivery/queues/packet and HARQ event streams equal;
  radio tolerance rtol 1e-6, SINR atol 1e-4 dB, RF power atol 1e-10 W.
- 28,739 actual decodes across the 44 cases. Online decoder probes,
  incompatible-resource no-TB attempts and post-send capacity guard failures
  were zero; existing power/resource/packet/HARQ audits passed.
- One target-not-available command rejection was retained. It is not a
  transmission/decoder failure or a completed handover failure.
- MRO lacks the required mobility samples in the short natural fixtures;
  sleep residence/evacuation is not established by a 64-ms test. Controlled
  tests cover their triggering logic; long runs must report natural coverage.
- Short gate elapsed 666.01 seconds, including serial reference comparisons.
  Enabled-case timings use two processes and are not single-process speedups.

Early development attempts are retained only as failure/debug evidence:
v1 driver path resolution, v2/v3 stopped while correcting held RET/beam
requests and legacy candidate-report formats. Only `short_v4` is accepted.
A monitor logging-key collision found in the first smoke run was also fixed.

## Long-run budget and limits

48 cases, 168,000 slots total, two independent worker processes:
24 single-app 4-UE/4000-slot runs; 16 combination runs at 4 UEs/4000 slots
(seeds 512/513); 8 combination pressure runs at 24 UEs/1000 slots (seed 512).
Each run has a 200-ms warmup, original geometry, 1.5 m/s trajectories,
actual NR decoding and the same physical precision. This is finite functional
acceptance, not proof of optimality, universal SLA satisfaction or stability.

Measured short-rate extrapolation: 3.77--6.87 hours. Four 4-UE long channel
traces already have all 4000 geometry blocks cached. The two 24-UE traces each
need 1000 blocks. Independent cold samples cost 0.153 s/block (street) and
0.140 s/block (Munich), including CFR reconstruction; warm reads cost 0.022 s
and 0.018 s. Approximate additional generation cost is 5--10 minutes, with
movement, path counts and concurrent cache publication uncertainty. Round the
whole unattended job to **4--7 hours**; this is a range, not a completion promise.

## Spark-only evidence and operator return

Project: `/home/njupt/oran_slot_sim_sionna/`.
Evidence root: `outputs/rule_based_20261015/`.
Accepted gate: `short_v4/validation.json` and `short_v4/SHA256.json`.
Frozen original source: `reference_source.tar.gz` and `reference_source/`.
Cold samples: `cold_rt_street24/`, `cold_rt_munich24/`.
Cache inventory: `long_cache_inventory.json`.
Long target: `long_v1/`; driver log: `long_v1_driver.log` (outside manifest).

Launch detached with stdin closed. Do not poll after launch and do not install
an automation. On the user's next instruction, inspect `long_v1/pipeline_status.json`,
case exit/status logs, validation and SHA256 inventory. Completed process and
accepted results are different states. The driver checks each case, records
failure and stops remaining cases on serious errors. Raw results/cache/logs
stay on Spark; this document is only a lightweight handoff summary.
