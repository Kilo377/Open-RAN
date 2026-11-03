# Fast channel computation acceptance — 2026-11-02

Adopted as the default `--model fast` system channel on 2026-11-03. The normal
`run-six-ue-cuda-nr.sh --model fast` entry now resolves to the configuration below.
Full and the older efficient/reference fast profiles retain their configured
geometry. The following alias still pins the named preset explicitly:

```sh
bash scripts/orin/run-six-ue-fast-10ms.sh --duration-s 10 --seed 1
```

The existing base configuration retains 5 ms geometry for full/older profiles.
For the previous system reference, explicitly request `--path-update-ms 5
--max-geometry-displacement-m .01 --fast-channel-mode dense --rt-prefetch-workers 1`.
This preset uses 10 ms path updates, a 0.015 m displacement cap at the current
1.5 m/s UE speed, 2 ms spatial/reference gains, and two CPU path workers. Actual
NR fallback stays on CUDA. Slot, measurement, queue, HARQ and control clocks are
unchanged. HTML replay still exports automatically.

The same settings are individually selectable with `--fast-channel-mode sampled`,
`--path-update-ms 10`, `--max-geometry-displacement-m .015` and
`--rt-prefetch-workers 2`. `--fast-channel-mode dense` retains dense CFR; full and
other fast profiles keep their existing channel path. On-demand system channels
require ideal CSI and compact paths. The default model remains full.

The channel accessor evaluates the original absolute CIR/delay/Doppler path sum
at the same physical coordinates selected by the system model and MU scheduler.
The physical resource grid, DMRS, TBS, EESM and actual NR fallback are retained.
Reference powers use the same full-symbol and 12-tone/PRB averages: synthetic
array path cross-products factor the finite frequency/time sums without a dense
CFR allocation. Antenna-specific delays/Dopplers use Cartesian reconstruction.
Coherent path cross-terms and RET/beam responses are retained. Scalar and batch
coordinate evaluation introduce no extra sampling approximation beyond the
existing system profile; finite-precision packet identity is not universal.

The new approximation is the longer geometry interval. Doppler evolution stays
active inside a block. The displacement cap and trajectory turning points still
bound/interrupt blocks. At higher speed the effective interval can be shorter.
A per-block OS lock gives foreground and background producers exclusive
ownership; publication is atomic and quota checked. Workers handle disjoint
block sequences, with existence rechecked under the lock. No caches are evicted.

Serial same-device, same-seed 400-slot timings (0.2 s simulated), including
startup and automatic HTML; warm values average two repetitions:

| Variant | Cold total (s) | Warm total (s) | Warm simulation loop (s) |
| --- | ---: | ---: | ---: |
| Previous 5 ms / dense | 44.02 | 39.78 | 29.31 |
| Lock only, one worker | 49.54 | 40.09 | 29.58 |
| 10 ms / dense, one worker | 40.11 | 39.19 | 28.69 |
| On-demand coordinates, 5 ms, dense reference powers | 43.55 | 35.54 | 25.05 |
| Above plus 10 ms | 35.43 | 34.64 | 24.14 |
| Direct reference power aggregation, 10 ms, two workers | 33.24 | 31.92 | 21.40 |
| Final batch implementation, 10 ms, two workers | 32.75 | 31.73 | 21.22 |

Cold trials are single measurements and initial JIT/cache effects are present;
lock-only does not establish a standalone speedup. Four workers did not improve
on two. Secondary vectorization was only ~0.6% faster than direct aggregation in
warm total time, within observed variation; no distinct speedup is credited to
it. The final batched version is retained as the exact completed 10 s version.

At unchanged 5 ms geometry, the final 400-slot case reproduced the reference
packet/HARQ events and packet KPIs. Direct reference-power aggregation matched
all 12 predecessor cases' packet/HARQ events, link counters and RF energy.
Additional seeds 2/3 preserved delivered totals in the short cases. Doubled-load
power/quota/sleep stimuli produced 33.29788 versus 33.29532 Mbps, equal per-UE P95
and RF energy, and 1,699 versus 1,700 deadline misses. This checks the fixed test
stimuli; it is not universal equivalence for arbitrary xApps or radio states.

The completed same-device/seed/traffic/control 10 s pair measured:

| Metric | Previous | Candidate |
| --- | ---: | ---: |
| Total wall including launcher/HTML | 28m 51.25s | 17m 22.66s |
| Simulation wall | 1,687.99s | 998.53s |
| Network goodput | 26.385495 Mbps | 26.373595 Mbps |
| UE8 P95 completion | 266.5 ms | 271.5 ms |
| UE8 overflow | 105,925 B | 121,600 B |
| UE6 P95 completion | 150.025 ms | 145.5 ms |
| First-TX BLER | 6.09888% | 6.06337% |
| Retransmission error rate | 30.53019% | 30.64825% |
| RF energy, all cells | 30 J | 30 J |

Total wall fell 39.77% (1.66x throughput in wall-clock terms), not 50%. The runs
were separate cold-cache measurements, not a universal hardware speed guarantee.
Goodput changed -0.0451%; the largest per-UE delivered change was UE8 -0.1631%.
UE0's weak coverage and UE8's queue problems remain. Tail counters are not exact:
UE0 deadline misses changed 2,075 to 1,880; UE6 1,697 to 1,674; UE8 20,256 to
20,298. Use the 5 ms reference for latency-sensitive/mobility sensitivity checks;
do not infer URLLC reliability from aggregate goodput matching.

Exactly 1,001 unique path blocks were generated in the candidate: 1 foreground
and 1,000 background, with no duplicates. The old run generated 2,000 background
blocks plus 313 foreground calls including block zero (312 redundant calls).
The candidate foreground snapshot stage fell from 463.56 to 11.77 seconds, but
some channel work now occurs at its consumer; this stage ratio is not an
end-to-end speedup. Background RT and foreground stages overlap and must not be
summed. Byte accounting and replay audits passed. The self-contained 10 s HTML
passed offline 3D/controls/KPI/responsive browser checks with no JS errors or
network requests.

94 relevant regressions passed before final vectorization; 71 physics,
concurrency, integration, platform, live and replay checks passed for final code.
The measured computational source hash is
`ecab8bcffeef4cbf194dbefc457a213da268be38e448abc8308cd4d34fcb7de2`.
The named preset/launcher were subsequently added and rerun for 400 slots.
Manifest, resolved configs, packet metrics and link statistics matched the
explicit-flags case exactly. Its computational/config inventory hash is
`3075e1ab64af9162249ffd3eb6ccd98d97e2225ea45d08105c7ee9321e3e1147`.
The November 2 validation left defaults unchanged; adoption was authorized and
implemented separately on November 3. Resolved geometry/channel/worker options
are frozen into live comparison arguments. The adoption verification is recorded
in `fast_default_adoption.json`.

See `fast_channel_acceptance.json` for numbers. Raw cases and traces remain on
AGX2 under `validation_candidates/fast_channel_20261102/`; the completed long case
is `batched/outputs/20261102_fast_channel_10ms_10s_w2`. A redundant second launcher
was cancelled and is excluded from the comparisons. Prior measurements and
outputs are preserved.
