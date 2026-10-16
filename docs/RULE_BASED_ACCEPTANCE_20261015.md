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

## Post-run review — 2026-10-16

The detached job exited 0. All 48 cases completed and passed structural audits.
Actual interval: **2026-10-15 17:02:23--19:23:45 CST**, 8482.15 s
(2 h 21 min 22 s). The initial 4--7 h estimate was conservative; it is not a
measured runtime claim. Two-worker case elapsed ranges were 223.9--463.5 s
for 4 UEs/4000 slots and 296.7--600.1 s for 24 UEs/1000 slots.

Post-run integrity verification read and hashed all **1157 files / 5,193,689,638
bytes** in the long manifest: no mismatch, missing or unexpected files.
There were **514,776 actual TB decodes**, zero AMC/BLER/calibration decoder
probes, zero incompatible-resource/unsupported no-TB attempts, and the existing
post-send capacity, resource, RF energy and packet/HARQ byte audits passed.
One target-not-available command rejection was retained; it is not an accepted
handover attempt. Aggregate events contain 87 handover attempts and 84 successes;
these totals alone must not classify the three remaining attempts as failures.

Remaining functional/performance evidence gaps:

- MRO issued no parameter changes in natural RT runs. Controlled tests cover
  its rule branches, but this matrix does not establish natural activation or
  mobility improvement under its minimum sample requirement.
- RET issued 30 requests; 10 feedback evaluations were flagged unattributable
  because other configuration changed. This does not establish RET convergence.
- Sleep issued 14 requests and TS issued 3 handover commands. Request counts
  are not counts of unique applied changes or proof of optimization benefit.
- In the business/resources combination, street `ue_3` and Munich `ue_21`
  remained connected for all 600 post-warmup slots, but had no scheduled slots.
  Final PRB caps were 1 and the active scheduler was latency-aware. Street
  `ue_3` ended at RSRP -76.59 dBm / reference SINR 9.74 dB. This requires
  inspection of priority, PRB-cap/TBS feasibility and competing traffic; it
  must not be explained solely as weak coverage or called long-term starvation
  from this finite 0.5-s run.
- Munich `ue_18` remained unconnected in all four pressure combinations, with
  weak reference RSRP. In the power/energy combination `ue_0` and `ue_6` were
  also unconnected. Coverage protection and the effect of power coordination
  need separate comparison; implementation integrity does not imply SLA success.

Detailed review and compact zero-delivery evidence were saved only on Spark as
`outputs/rule_based_20261015/long_v1_postrun_review.json`, outside the immutable
long-run manifest. No additional simulation, automatic retry or tuning was
started during this review. Mac retains only this lightweight documentation.
