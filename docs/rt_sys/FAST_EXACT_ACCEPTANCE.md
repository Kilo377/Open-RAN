# Fast implementation efficiency — 2026-11-03

This experiment targets implementation overhead in the accepted fast system
model. It does not add a channel approximation, change the scheduler algorithm,
or reduce slot, measurement, queue, HARQ, controller or recording frequency.
The 10 ms geometry and 2 ms spatial/reference gain intervals remain unchanged.

Selected settings under `rt_sys` (resolved for the system fast profile only):

```json
{
  "compact_sinr": true,
  "cached_static_phy": true,
  "compiled_report_copy": false
}
```

The normal `--model fast` entry resolves these defaults. Full and older
efficient/reference profiles remove these system-only options. The independent
compiled report copier remains opt-in; its short-run results do not establish a
speed benefit and it is not selected by default.

## Implementation

The system PHY retains representative SINRs with their exact data-RE/layer
multiplicities. DMRS masks, zero-power allocated-resource erasures, stream
patterns and row-major NR fallback order are retained. Weighted EESM replaces
expansion followed by reduction; the official beta parameters, clamp bounds,
Sionna BLER interpolation and NR transport-block segmentation remain in use.
Actual NR fallback expands the original resource order and keeps its original
TB identity/codeword and receiver.

Weighted exponential sums use double precision before returning the normal SYS
dtype. This avoids float32 cancellation at very low SINR. It is the same EESM
equation, with a different finite-precision reduction. Tests compare against the
installed official double EESM, and separately bound differences from the
previous float32 expanded path (effective SINR within 0.01 dB and probability
within 1e-5 in the fixed oracle cases). This is not a promise of bit-identical
Bernoulli outcomes for every possible input or seed.

Immutable DFT bases are cached by antenna count, codebook, profile and beam.
Mutable RF powers, reference measurements and MAC/control state are not cached
by this change. Logical-channel grants copy scalar metadata and detach their
mutable constraint list while retaining their immutable TB contract. Other
grant types keep the generic deep-copy path. Prepared data masks/SINRs are reused
within a slot, instead of being rebuilt for feedback, output and diagnostics.
Interference diagnostics use exact resource multiplicities instead of large
Python float lists.

## 400-slot measurements

Same AGX2, seed 1, six moving UEs, Munich depth 2, MAXN, CUDA NR fallback and two
CPU RT workers. Geometry is warm for the short tests. A primer populates the RT
cache; the candidate receives a physical copy of the same immutable geometry
cache within its own project directory. Runs are serial, two repetitions, with
reverse variant order in the second pass. Wall times include interpreter
startup, simulation, cleanup and self-contained HTML export.

| Variant | Mean total (s) | Mean simulation (s) |
| --- | ---: | ---: |
| Previous implementation | 31.647 | 21.037 |
| Shared preparation reuse | 31.166 | 20.631 |
| Above + compiled report copy | 31.317 | 20.786 |
| Above + compact SINR | 28.549 | 18.054 |
| Above + static-data reuse | 30.625 | 20.131 |
| All three candidates | 27.852 | 17.355 |
| Selected: compact SINR + static-data reuse | 27.862 | 17.380 |

Selected total time is 11.96% lower; simulation time is 17.38% lower. Compiled
report copying has no demonstrated independent benefit. These short tests do
not support a further 2x end-to-end speedup.

All short variants, including both final selected runs, reproduced the original
packet-event and HARQ-event file hashes. The expanded/compact PHY tests also
cover 14-symbol grids, FFT guards, partial zero-power PRBs, mixed MU layouts,
rank two, DMRS patterns, no-path outages and actual NR fallback ordering.

The candidate passed 111 regression tests; after disabling compiled report
copying by default and adding a strong-link underflow regression, the final
configuration passed 44 focused regression tests
(these sets overlap). The executed final source fingerprint is
`e465f7e5519b13b3cc142ffcae85c788b26c88f7b903c83a1e3ec4ad0b8e9a9f`.

## Long-run validation

Fresh geometry caches, same device, seed, traffic, geometry and recording:

| Metric | Previous | Selected |
| --- | ---: | ---: |
| Total including startup/cleanup/HTML | 1,040.48 s | 798.80 s |
| Simulation wall | 997.23 s | 755.94 s |
| PHY inclusive wall | 422.04 s | 245.35 s |
| Scheduling wall | 152.16 s | 132.21 s |
| Network goodput | 26.3735952 Mbps | 26.3735952 Mbps |
| Actual NR decodes | 1,093 | 1,093 |
| SYS sampled attempts | 99,451 | 99,451 |
| Total emitted RF energy | 30 J | 30 J |

Total time fell **23.23% (1.303x)**: 17m 20.48s to 13m 18.80s, saving 4m 1.68s.
This does not reach the requested further 2x speedup. The main PHY improvement
is substantial but the rest of the per-slot simulation remains. Inclusive
stages are nested; batch-feedback time is already inside PHY time. RT workers
overlap the main loop and must not be added to it. Both runs independently
generated 1,001 native RT blocks, with zero skipped blocks. The longest RT
worker took approximately 425.32 s in each run. These are one long pair and two
short repetitions, not a universal performance guarantee.

The full packet-event and HARQ-event files are byte-identical. Packet metrics,
link counters, per-UE delivery/latency/overflow statistics and RF energy are
identical. Across 120,000 UE-slot records there are zero differences in ACK,
delivered bytes, MCS, layers, resource mode or actual-NR route. The manifest is
identical. Raw mean SINR/interference diagnostics differ by at most 2.85e-14 dB.

**Effective SINR is a numerical correction, not bit-identical telemetry.** The
previous float32 exponential sum sometimes underflows on strong links and
incorrectly selects the 30 dB maximum clamp. The selected double reduction gives
26.5057646 dB instead of 30 dB for UE2 at slots 500–503, a maximum recorded
difference of 3.4942354 dB; both BLER predictions are zero. The strong-link
regression independently checks a constant 700 linear SINR: its EESM must be
700, as returned by the selected calculation and official double EESM; the old
float32 path incorrectly returns the 1,000 clamp. Thus the 0.01 dB comparison
bound above applies only to its fixed random oracle cases, not to legacy
float32 underflow cases. The default official beta minimum is 1.6; double
precision avoids exponential underflow within the unclamped range up to 1,000.

An independent 400-slot control pair also reproduced packet/HARQ hashes and RF
energy. Cell 0 power at slots 79/81/161/241/321 was respectively
1 / 0.5011872336 / 1 / 0 / 1 W, verifying power changes and sleep/wake effects
without stale static caches. The automatic replay audits passed.

Detailed machine-readable timings, event hashes, trace differences and control
witnesses: [fast_exact_acceptance.json](fast_exact_acceptance.json).
