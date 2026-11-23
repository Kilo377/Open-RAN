# Conflict methods

The default UI method is now `goal_obstruction_v2`, documented in
[`GOAL_OBSTRUCTION.md`](GOAL_OBSTRUCTION.md). The selector also retains
`goal_obstruction_v1` and `intent_control_dynamics_v4` for comparison. None selects
or deploys a combination. The sections below describe that retained method.

The UI and its conflict JSON export use `intent_control_dynamics_v4`, the pure
`SmoConflictRisk` module in `smo/web/app.js`. Inputs are the knowledge graph,
recorded shadow proposals, their exact contemporaneous frame, frozen intent
intervals and candidate memberships. The scoring engine never reads measured
post-deployment outcomes, arbitrates commands, ranks, prunes or deploys apps.
`POST /api/conflicts/inputs` provides the trusted recorded inputs and complete
graph-derived catalog without invoking the legacy response model. Its only
request fields are saved intent IDs, an observation slot, window length and
the dashed-edge option. It does not condition input availability on arbitration
policy or on the ability to predict net KPI outcomes. The legacy response
endpoint remains diagnostic; its forecasts and response-fit eligibility are
not inputs to this score. Older running servers use the compatibility route
until the service is reloaded.

## Unified control dynamics

For each requested KPI k and app a, project each parameter action onto its
measured reference control direction, then sum **within the app**:

    d[a,k] = sum_p G[p,k] * delta_p[a]
    z[a,k] = d[a,k] / s[k]

Only requested KPIs participate. Multiple parameters of the same app never
create self-conflicts. Shared writes and different-parameter pathways use the
same comparison; no parameter family is removed merely because apps share a
resource. Direct/indirect/implicit graph counts are optional topology metadata,
never separate dynamic scores or terms in the formula.

For each distinct app pair and shared visible KPI pathway:

    strength[a,b,k] = abs(z[a,k]) + abs(z[b,k])
    cancellation[a,b,k] = strength[a,b,k] - abs(z[a,k] + z[b,k])
    risk = 100 * sum(w[k] * cancellation) / sum(w[k] * strength)

The ratio is clipped to [0,100] for numerical roundoff. This is a weighted
all-pair cancellation proportion, **not** the previous worst-local maximum.
Consequently a known zero/same-direction app can lower the proportion by
adding denominator strength. Missing effects cannot add that denominator.
This property is explicit, tested and must be assessed before selection.
A single app has no inter-app opposition even if its own paths counteract.
No comparable nonzero magnitude means zero only when evidence is otherwise
complete; wholly unsupported active effects return null. Known partial
projections can contribute, with partial evidence and coverage exposed.

## Intent scale, urgency and direction

Each KPI has one scale for every combination in a review. Use the first frozen
intent anchor (not each combination's outcomes) as baseline. Its distance to
the nearest finite requested boundary provides a target-change/tolerance
scale. If the baseline is already on a boundary, use its absolute magnitude.
At a zero baseline, use a nonzero bound or a one-unit numerical fallback;
this fallback is labeled rather than treated as a calibrated tolerance.
Scales have a numerical floor. Single-sided intervals never use infinity as
a scale. The same frozen scales persist as current telemetry changes.

The margin is the minimum signed distance of the current observed KPI to the
finite intent boundaries. The initial urgency rule is explicitly heuristic:

    w[k] = 1 / (1 + max(0, margin[k]) / s[k])

Unmet/on-boundary targets have weight 1; interior targets have lower weights
as their safety margin grows. All candidate groups use identical current
weights. The rule is not learned or a calibrated satisfaction probability.
Missing current telemetry is flagged, and fallback urgency 1 is not silently
presented as complete intent context. Empty joint intervals block scoring.

Direction roles are independent of the opposition fraction: outside the
interval, an effect moves toward or away from the allowed region; inside it,
an effect consumes/builds the nearest boundary margin. At a symmetric interior
point there is no uniquely preferred direction. A point-hold interval rejects
either nonzero direction. All apps can agree on a harmful direction and still
have zero inter-app opposition; the UI explicitly reports adverse influence
counts. Scores never certify attainment or produce a net KPI prediction.

## Reference weights and uncertainty

Nonuniform population actions use the mean parameter step over the calibrated
population. This is an explicitly named common-direction projection, not an
identified derivative for each UE. Known DRX inactivity weights remain usable
when cycle/on-duration change; missing components remain visible. Numeric
extrapolation keeps the local reference weight and marks the approximation.
An unrelated missing component does not discard known influence.

Categorical strategies use measured finite contrasts, including differences
composed through a common baseline. Uncertainty is retained on the composed
reference difference rather than deleting the measured chain. Interval subtraction is a conservative
marginal envelope, not a newly estimated confidence interval. There is no
numeric ordering invented for round-robin, latency-aware and QoS modes.

When a profile-specific contrast is missing, the measured active-to-micro-sleep
contrast can serve as an explicit **binary active/sleep-family reference weight**. All these
activations have reference step 1, deactivation -1. This is an assumption for
rough influence comparison, not an estimated light/deep profile's actual KPI
change or a claim that the profiles are equally strong. Different unmeasured
sleep-to-sleep transitions remain unsupported. The JSON and UI mark this
unmeasured profile substitution as partial evidence.

New context-identification matrices measure all three sleep profiles directly,
DRX cycle and inactivity separately, and a joint DRX profile as a validation
contrast. Available profile-specific evidence takes precedence over the
fallback. Matrices are selected by traffic context, never pooled across native,
steady high load and burst high load. Raw grid points, per-seed slopes, fit
residuals and confidence intervals remain attached to the graph's links.
When a structured DRX proposal matches a measured joint profile, that finite
reference weight takes precedence over its separate timer slopes. Its covered
fields are not counted twice. Reverse transitions and nonuniform group changes
remain explicit common-reference projections. A different on-duration profile
cannot silently borrow the joint measurement. A measured joint response takes
precedence even when its direction interval includes zero; that uncertainty
is recorded as partial evidence instead of silently replacing the weight.

The complete-graph policy retains finite measured weights and declared target
paths by default. The dashed toggle hides only uncalibrated candidate side
effects; it no longer hides a measured relation because its direction interval
includes zero. Numerical uncertainty notes and partial-evidence flags remain.
No gain, unavailable proposal, measured zero, unchanged parameter and proven
no-action inference remain distinct. Numeric coverage is not causal validity.
An observed zero member/component provides a neutral reference, including an
edge whose measured components are all zero. For candidate coverage, a zero
response does not add unrelated KPI coverage; an app's declared target path
remains available as structural coverage, never proof of improvement. An
uncalibrated declared path remains numerically unknown. A zero sleep profile is not
borrowed as evidence that an unmeasured sleep profile also has zero influence.

## Replay and experiment context

Scores use actual recorded shadow inference. A replay frame between decision
boundaries may use the nearest earlier decision with its **own** telemetry
frame; lag is visible. An app already enabled in that replay may emit no new
action. Independent deployment trials reset to native state, so their cold
activation is a different intervention. The UI retains this mismatch warning;
we do not substitute another slot's cold-start proposals or use feedback to
manufacture matching risk inputs. Existing trials and raw data are retained.

## Retained first-order response diagnostics

`smo/conflict_quantification.py` implements `proposal_first_order_v1`.
`POST /api/conflicts/quantify` consumes saved intent IDs, an actual snapshot slot,
the observation length, and the dashed-edge option. It reads trusted graph
calibrations and recorded shadow inference; clients cannot supply actions,
coefficients or a selected subset of combinations. It neither executes models
nor advances arbitration RNGs, changes controls, ranks or removes combinations.

A control chain consists of an intent interval, current KPI/parameter values,
the trained xApp's actual shadow proposal, and an applicable measured gain.
Numeric range fits use `gain × parameter_delta`; categorical changes use a
matching named contrast (or its inverse), never numeric category IDs. Common
population gains are applied once to a matching common-population step. A
partial UE intervention is not assigned an arbitrary fraction of that gain.
Structured DRX changes must match the measured component and hold other fields
fixed. Unmeasured transitions, mixed population levels, extrapolation, missing
fresh proposals and unsupported arbitration remain explicit unknowns.

Direct conflicts compare full requests on the same actual resource: an offload
probability per UE, or an atomic scheduling mode/complete UE-weight map per
cell. Identical writes are compatible. Distinct xApps retain uniform request
probability; identical-valued requests are combined without losing probability
mass. The module enumerates possible frozen-proposal arbitration outcomes,
not a chosen winner. Disjoint resource requests compose. Above 256 distinct
outcomes, target risk is unknown rather than estimated from truncated branches.
Only currently implemented random arbiters may receive numeric probabilities.

Indirect conflicts aggregate each xApp's unique-RCP chains per requested KPI.
Shared RCPs go through the direct-arbitration channel and are not added twice.
For opposite effects `a,b`, cancellation ratio is `2 min(|a|,|b|)/(|a|+|b|)`
and cancelled amount is `min(|a|,|b|)` in KPI units. Same-direction effects have
zero cancellation. Zero/tiny magnitudes remain visible; unknown effects have
no numeric cancellation ratio. Different chains belonging to the same xApp
never become application-pair conflicts. Shared-RCP-only pairs do not receive
an additional indirect score. Existing graph risks, including implicit graph
semantics, remain separate from these proposal-conditioned estimates.

Joint KPI values are computed from each unique, composed parameter vector,
not by summing alternative shared writes. Every intent endpoint (including
strict and two-sided bounds) is checked. Output includes predicted min/max,
coefficient-interval envelopes, maximum interval gap, minimum interval margin,
and uniform-arbitration probability mass outside the interval. These are
instantaneous frozen-proposal first-order estimates, not simulated attainment,
empirical failure rates, a multi-window trajectory or calibrated SLA risk.
No cross-KPI average utility or single combined conflict score is introduced.

Operating-context shifts, multiple changing parameters, limited linear fits,
uncertain signs and different response windows remain conditional flags. Gain
effects are not rescaled by an invented time constant. Coefficient envelopes
use interval arithmetic and are not joint confidence intervals; fit residuals,
state transfer error and nonlinear cross-effects are not included. Future
closed-loop decisions and controller state changes are not predicted.

The UI shows the current method, per-combination summaries, expandable control
chains and target risk, selected-combination graph-inspector details, and JSON
export with input hash/provenance. Its green/red empirical result blocks remain
based on actual deployment measurements. Current predictions never recolor a
historical result. Missing recorded decision frames can be explicitly aligned
using the nearest earlier retained exact decision snapshot. Replay retains a
bounded additional set of decision frames; no metrics or intent anchors are
interpolated or rebased.

For a replay telemetry frame without an exact decision recording, the UI
automatically retries the nearest earlier retained decision suggested by the
service. It quantifies using that decision's own exact frame and reports both
the requested frame and the decision timestamp/lag in the UI and JSON export.
Future or non-retained snapshots and cross-source responses are refused.
Intent bounds and the running batch stay unchanged. Unknown gain effects and
unresolved indirect pairs remain explicit; they are not displayed as zero.

An optional operator-owned `--empirical-response` artifact supplies a scoped
observational MEC scheduler fallback when controlled gains do not describe
the actual joint action. It has separate provenance and error envelopes;
original unknown per-parameter links remain unknown. Configuration, horizon,
native other controls, observed range and identified subspace must all match.
See [MEC_EMPIRICAL_RESPONSE.md](MEC_EMPIRICAL_RESPONSE.md) for extraction,
validation, refusal conditions and the current study's limitations.
