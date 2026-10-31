# Native MU baseline: physics verification and KPI scope

The native network is deliberately an unoptimized baseline for xApp experiments.
This investigation does not disable MU by default, tune the RZF regularizer or
OLLA limits, change the MCS mapping, or add an automatic pairing/recovery policy.

## What was tested

The completed DGX run used Munich, three cells, twelve moving UEs at 1.5 m/s,
3.5 GHz, 10 MHz, 24 active PRBs, 0.5 ms slots, actual `nr_direct` CRC feedback,
and no active xApp. Its two simulated seconds took approximately 36 minutes;
this is offline simulation, not wall-clock real-time emulation.

The original `ue_0` outcome was 1885 failed first transmissions out of 1892,
but only seven failed retransmissions out of 1891 actual retransmissions.
The feedback-consumed OLLA summary can lag these actual-decode counters by one
slot. The reference feedback and the executed post-LMMSE data SINR describe
different transmission hypotheses.

Fixed-grant replay of the saved absolute channel reproduced the low MU SINR
with NumPy complex128, Torch complex64 CUDA and Torch complex128 CUDA. At slot
3999, removing only the other same-cell UE changed `ue_0` from about -4.52 dB
to 18.11 dB on the same channel. The native RZF and LMMSE formulas were also
compared with Sionna 2.2's independent implementations, using the same channel,
regularization, transmit power and noise covariance. The largest RZF weight
difference was below 9e-16; the largest LMMSE relative difference was below
1.7e-13 in the sampled double-precision comparisons.

These checks establish consistency for the sampled equivalent-link workload.
They do not validate a full RF/OFDM receiver, real-world coverage, or URLLC tails.
No BLER table was fitted and no throughput improvement was an acceptance gate.

## Matched closed-loop A/B

The MU-enabled and MU-disabled runs keep the physical manifest and channel cache
key identical for each seed. The only changed network configuration is
`simulation.scheduler_enable_spatial_multiplexing`. Both variants use the same
native scheduler, fixed power, reference reporting and NR receiver, with xApps
off. Warmup is 50 ms, and each short case is 400 slots / 200 ms. The short cases
diagnose the mechanism; they are not target-BLER or long-term QoS certificates.

Initial UE rank is now explicitly configured as `ue_initial_rank: 1`, independent
of the MU switch. Historical MU-off snapshots left `rank`/`ri` null, although
all executed transmissions in the matched tests were also one layer. Legacy
comparison normalizes only those null defaults and checks the actual layers;
new runs require identical UE configurations without normalization.

Observed short-run results (first BLER over all actual first transmissions;
goodput over the post-warmup 150 ms interval):

| Seed | MU | ue_0 first BLER | ue_0 goodput Mbps | Network goodput Mbps |
| --- | --- | --- | --- | --- |
| 1 | on | 100% (192/192) | 4.7555 | 26.4161 |
| 1 | off | 5.42% (16/295) | 10.1385 | 30.8514 |
| 2 | on | 0% (0/288) | 10.4573 | 31.6197 |
| 2 | off | 0% (0/288) | 10.4573 | 28.5084 |

MU is harmful in the first case and beneficial to aggregate throughput in the
second. Seed-2 `ue_0` is not paired in the first observed slot; enabling MU
globally does not imply every UE uses MU. These results support conditional
policy experiments rather than a blanket MU disablement.

Reproduce in an already activated project environment:

```bash
python scripts/validation/validate_native_mu_baseline.py \
  --platform dgx --seeds 1 2 --slots 400 \
  --output outputs/native_mu_validation
```

Use `--platform agx` from the Orin activation environment to select the existing
LLVM RT / CuPy algebra / CPU NR execution profile. A new output directory is
required. Validation results include the source run paths, exact channel keys,
per-UE load/queue/delay data, and separate first-TX/retransmission statistics.

## Reporting repairs

* Live and final reports now expose first-TX and retransmission attempts, NACKs
  and ratios separately, globally and per UE. Empty denominators remain `null`.
  Idle, unsupported and no-TB cases are not counted as actual decoder trials.
* SU, MU and mixed resource modes come from executed same-cell RE overlap, not
  the number of users scheduled in a cell or a provisional scheduler group ID.
  Two frequency-orthogonal users are both SU. Single-user multi-layer MIMO is
  also SU. Old artifacts without mode evidence remain `unknown`.
* The PHY trace reports received other-UE intra-cell interference power before
  receive combining. This is distinct from inter-cell power and post-LMMSE SINR.
* The live UI shows actual data SINR alongside delayed reference feedback and
  makes the per-UE SU/MU/error breakdown inspectable. These are observer exports;
  no instantaneous PHY truth is added to the controller observation contract.
  Existing delayed `first_tx_bler`/sample-count observations remain available to
  xApps through the normal observation path.
* Existing completion-delay values retain their historical slot-index meaning.
  The UI now labels them as queue completion delay: same-slot completion is zero,
  within-slot airtime and processing/transport/core delay are excluded, and
  unfinished/dropped arrival units are excluded. They are not end-to-end latency.

The observed MU/SU discontinuity is a legitimate outcome of this native policy
on the tested channel, rather than evidence of a CUDA or SINR arithmetic error.
Pair selection, interference coordination and scheduling remain open to xApps.
An xApp should improve the measured result through declared controls; the
simulation must not silently rescue its baseline or overwrite actual CRCs.

## Execution checks on 2026-10-22

The final DGX regression suite passed 514 tests with seven skips and two
subtests. The two warnings concern the installed Torch GB10 capability range
and one existing high effective-code-rate test case. AGX's reporting/platform
compatibility subset passed 32 tests.

For each MU setting, a 400-slot seed-1 regression compared all 4,800 UE-slot
rows before and after the repairs. ACK/NACK, MCS, transport-block IDs, delivered
bytes, allocations, queue state and final packet metrics were identical; the
maximum SINR difference was zero. The MU-off regression includes the explicit
initial-rank correction. Live and final decoder counters agreed.
The reusable validation command also completed a seed-1, 120-slot on/off smoke
pair, including strict configuration equality without legacy normalization.

An AGX 20-slot smoke run completed with 171 actual decodes, using CuPy CUDA for
channel reconstruction and linear algebra, LLVM CPU for RT and CPU for NR.
Its trace, live state, final summary and runtime diagnostics agreed on the
separate first-TX/retransmission counts. The native UI changes were not visually
inspected in this validation.

Separately, the original-source AGX cold-cache retest completed all 4,000 slots
with exit code zero in 8,745.97 seconds. This did not reproduce the earlier
segmentation fault; its root cause remains unestablished.

Official mathematical reference: [Sionna RZF](https://nvlabs.github.io/sionna/phy/api/mimo/precoding.html)
and [Sionna LMMSE equalization](https://nvlabs.github.io/sionna/phy/api/mimo/equalization.html).
