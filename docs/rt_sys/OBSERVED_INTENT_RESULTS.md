# Observed intent results

`baseline_relative_intent_v2` in `smo/result_evaluation.py` evaluates only the
unique frozen intent KPI set from actual equal-window BEFORE/AFTER readings.
It does not read KG gains, proposals, conflict scores or unrequested KPIs.
Every BEFORE improvement score is zero. A positive result indicates observed
partial benefit; it does not by itself establish full interval attainment.

For each requested KPI, omit a boundary that is merely its natural physical
domain endpoint. Keep both bounds for a point target. Define the raw outside
interval distance as `d = max(0, L - observed_min, observed_max - U)` over the
requested boundaries. Use the recorded window mean unless the measurement
contract explicitly requires its extrema. Invalid physical-domain readings,
missing/invalid measurements and unresolved task samples remain unavailable.
A valid target equal to the entire physical domain is neutral: its distance
and improvement are zero; it does not create a missing-boundary error. If an
extreme scale causes nonfinite normalized violation, the complete result is
unavailable rather than reporting a misleading zero-change score.

For one effective requested boundary, the oriented improvement is
`delta = after - before` for a lower bound and `before - after` for an upper
bound. A half-line therefore rewards useful movement even after its boundary
is reached. For a genuine band or point, `delta = d_before - d_after`: movement
inside an already satisfied band is neutral, with no invented preference for
the middle. Each KPI improvement is `g = 100 * delta / s`.

The same nonzero `s` is frozen from the durable common BEFORE reference for all
members/windows, normally its absolute magnitude. A preconfigured positive
`result_evaluation_scales` entry may override that scale. AFTER data, member
order, KG coefficients and measured deployment outcomes cannot set the scale.
Zero baselines for ordinary dimensional KPIs require an explicit scale; there
is no invented one-unit fallback. The shared `reference_scale` helper used by
conflict normalization retains its existing behavior and is not changed by
this observed-result method.

`urllc_failure_fraction` is already a ratio: use `s = 1` for ratio/fraction
units (`s = 100` for percent units), so `g` is a percentage-point change. Thus
0% -> 7% scores -7, 7% -> 5% scores +2, and 0% -> 0% scores 0. No division by
the old zero ratio occurs. Historical `urllc_failure_events_per_s` remains a
different KPI; events/s cannot be silently relabelled or converted without its
packet denominator. Ratio measurements outside their declared physical domain
are unavailable, never clamped into a favorable result.

The equal-value aggregate has a noncompensable regression veto:

    E = min(g_k)     if any requested KPI regresses
        mean(g_k)    otherwise

All requested KPIs have weight one after normalization. An unchanged KPI does
not block another target's finite benefit. Any regression cannot be offset by
another target's surplus. This hard admissibility boundary is intentionally
discontinuous, rather than a smooth utility or failure probability. The result
exports `no_regression` / `admissible` and the affected `regression_kpis` as well
as the scalar. A missing requested measurement or scale invalidates the whole
result and cannot be dropped to increase its mean.

Remaining violation is separately retained for every requested KPI as raw `d`
and normalized `100 * d / s`; aggregate mean and maximum violation are exported
for both phases. An unchanged, already violated intent has improvement zero
and positive remaining violation. Boundary-gap completion percentage is
reported only when BEFORE has a positive distance to close. A strict open
endpoint can have geometric distance zero yet remain unsatisfied. Original
open/closed rules always determine attainment; no epsilon is invented. If an
already satisfied finite band moves exactly onto an excluded endpoint, the
admissibility flag is false even when its geometric improvement is 0.

The compatibility fields are `before.score = 0`, `after.score = E` and
`change = E`. Per-metric phase values retain raw signed boundary margin,
`distance`, `normalized_violation` and the original assessment status;
`improvement_score` / `change` carries g. The normalized margin is available
only as `margin_score` and is not the primary evaluation score.

All score computation is a read/export-time derived copy. Original frozen
intent ranges, source traces, archived batch JSON, raw KPI readings and
pass/fail assessments are not rewritten. Candidate selection and independent
conflict quantification do not consume the new deployment result.

`python -m experiments.evaluate_intent_results --batch ... --output ...`
re-scores retained data without launching simulations. A risk report may be
joined by group ID for side-by-side inspection; it does not enter E. A 500-slot
window is a short-term observed response, not proof of long-term behavior or a
causal estimate. The formula also supports longer equal windows.

Actual common-origin validation remains separate. `smo/evaluation_reference.py`
audits configuration, seed, predeployment states and controls across completed
runs. An identical fresh-run baseline across groups is insufficient to claim
that an arbitrary earlier UI snapshot was restored. Existing incompatible
origins must be labelled rather than repaired by moving frozen thresholds.
A paired continuation retaining the previous policy is needed to attribute
change beyond natural time variation; the score alone makes no causal claim.

The UI's “对齐评估起点” prepares a certified BEFORE prefix from retained
recordings, with equal-window KPI readings and recorded shadow proposals.
No preparation simulation or AFTER KPI is needed. A pinned source archive
identifies the original network dynamics; the new worker checks dynamics,
models, canonical JSON configuration, seed, exogenous arrivals and BEFORE
state before any candidate is activated. This reconstructs the native prefix;
it is not a checkpoint restore of an arbitrary old UI state. Earlier active
intent bounds remain frozen and differing old anchors stay in the audit.

New prepared-origin runs use the identical absolute AFTER interval
`(baseline_slot, baseline_slot + observe_slots]`, including actuation delay.
The first application receipt remains separate evidence and cannot shift a
group's observation window. Legacy batches have an explicit unaligned-origin
audit flag; read-time rescoring does not retroactively repair their origins.
For extrema-constrained KPIs, the UI shows the window mean and explicitly
states that the full-window bounds determine both attainment and scoring.
