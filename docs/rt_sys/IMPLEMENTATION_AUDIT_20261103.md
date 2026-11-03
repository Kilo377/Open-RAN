# Implementation audit — 2026-11-03

The executed RT/SYS path uses real Sionna scene/path generation, absolute path
coefficients with delay/Doppler, RZF/LMMSE stream interference, official SYS
EESM/BLER, and explicitly counted actual NR fallback. No fixed throughput/BLER,
KPI playback, fitted result multiplier or mock PHY was found in this execution
path. Production NumPy-only fixture mode is rejected. This does not certify every
legacy repository module or constitute full 3GPP/O-RAN protocol conformance.

The audit checked deployment `76c23b8`, computational source
`346ddf41c8771f3ce4c98332ef921a4f400ec8931aa93cbbb8fddb99ba9089d1`.
Final fixes were verified at computational source
`3d7988189f8b593c04bd8a9d032a43799c05c051554860a0bbaa8b6b3c5743e4`.
Machine-readable evidence is in `implementation_audit_20261103.json`; raw logs,
traces and fault-injection drivers remain on AGX2 under
`validation_candidates/implementation_audit_20261103/outputs/implementation_audit_20261103/`.
Original pre-fix fault evidence is under the main copy's
`outputs/implementation_audit_20261103/` and was preserved.

The independent numerical oracle was Sionna's own `Paths.cfr`, not the project's
other reconstruction implementation. A real Munich PathSolver execution checked
18 UE/BS links, 96 physical frequency coordinates per slot and slots 0/1/3 with
14 OFDM symbols. Maximum relative L2 CFR error was 5.48e-7. Reference power
averages were also compared to native CFR projected onto independent random
orthonormal beam vectors. These positive tests used no fake backend or cached
path inputs.

A fresh-cache 400-slot run generated exactly 21 geometry blocks with two workers.
Warm-cache replay generated zero blocks but re-executed every slot. Packet/HARQ
event hashes matched across cold, warm and final fixed-code runs. Independent
trace analysis reconstructed offered/admitted/overflow/delivered bytes from
packet events, verified nonoverlapping delivered byte ranges and unique TB ACKs,
matched PHY delivered bytes to packet totals, and recomputed ACK/NACK counters
and RF energy from raw per-slot records. All checks passed.

Fixed control stimuli changed actual cell_0 RF output, not just the control log:

| Slot | Actual RF output |
| --- | ---: |
| 79 | 1 W |
| 81 | 0.501187 W |
| 161 | 1 W |
| 241 | 0 W |
| 321 | 1 W |

The controlled 400-slot goodput changed from 30.00944 to 21.1536 Mbps and packet
and HARQ histories changed. This is a response test, not an algorithm improvement
claim. The normal fast run recorded 77 actual NR decodes and 2,149 SYS-sampled
outcomes. A separate 20-slot full run recorded 120 actual NR decodes and zero
statistical outcomes. Full and fast have different simulation horizons here;
their goodput figures are not an accuracy comparison.

Fault injection made RT generation and official SYS probability evaluation
raise exceptions. Both aborted without a final summary or successful status;
neither fabricated substitute KPIs. Cleanup failure exposed a real bug:
`completed=true` could remain on disk after resource close failed. Cleanup now
attempts every resource, records failures, preserves an earlier exception,
and publishes completion only after cleanup. Status writes are atomic. Internal
HTML export can read finished simulation data while the run is finalizing;
public export rejects failed/cancelled/unfinished runs. An actual injected
KeyboardInterrupt now propagates and records `cancelled`, `completed=false`.
A cancelled prefetch worker now reports incomplete/cancelled and its assigned
and processed counts, rather than claiming all work completed.

A second real telemetry defect was the replay resource-utilization denominator:
it included FFT guard tones. The current grid is 14x512 FFT REs versus
14x24x12 active REs. Full active allocation therefore displayed 56.25% rather
than 100%. Traces now retain both FFT and active-grid sizes, and replay uses the
active denominator. Historical real traces infer it from recorded numerology.
The original 10-second dataset was re-rendered without rerunning physics; only
utilization and source-folder/final-wall-time metadata changed. Original HTML
was preserved. Offline browser checks passed for this 10-second HTML and the
new 400-slot HTML, with no external requests or JavaScript errors.

Legacy diagnostic fields require caution. `pre_eq_sinr_db` currently aliases
post-LMMSE arithmetic-mean SINR. Legacy BER, beam gain and precoder-match values
are not independent measurements. Legacy CQI is MCS-derived and rank/RI is the
allocated layer count. These fields are explicitly annotated in final metadata;
they are not exposed as valid measured xApp KPIs. A contract needing an
unimplemented measured precoder score is unavailable and is rejected. Supported
reference-based CQI/beam reports are separate. Generic public BLER metadata now
says ACK/NACK rather than incorrectly labelling every fast outcome actual CRC.

The declared approximations remain: ideal CSI for system mode, 10 ms geometry,
2 ms gain anchors, four frequency samples/PRB, and predominantly statistical
TB feedback without waveform soft combining. These are real calculations with
approximations, not full waveform execution. RF energy is emitted scheduled RF
energy, not device/baseband/circuit energy; reference reporting uses a separate
configured transmission hypothesis. No-path slots are counted separately as
outages and excluded from transmitted-TB BLER. Queue-completion latency excludes
within-slot airtime, transport/core processing and undelivered/dropped units.
These boundaries matter for selecting xApp experiments and interpreting KPIs.

145 tests passed for the status fixes and 173 final integration, contract,
physics and replay tests passed; these overlapping suite counts are not added.
The final 400-slot packet/HARQ histories were unchanged by the audit fixes.
Nine replay checks additionally passed after tightening the terminal-stage gate
to reject historical `completed=true` flags with failed/unfinished stages. The
last 400-slot gate verification at the final source hash also matched the
original packet/HARQ event hashes and packet/link counters exactly.
