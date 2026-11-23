# Frozen-proposal goal obstruction

`goal_obstruction_v2` is the default method in `smo/web/goal_obstruction.js`.
The UI retains `goal_obstruction_v1` and `intent_control_dynamics_v4` for comparison.
Candidate selection, the knowledge graph and the trained xApps are unchanged.
The new score uses only frozen intent intervals, the recorded decision preview,
its contemporaneous observation and existing parameter–KPI reference weights.
It neither fits to measured outcomes nor forecasts deployed KPI values.

Influence dimensional alignment version 1 uses `smo/influence_units.py`.
Raw graph gains and their statistical evidence remain unchanged. A derived
`projection_units` record validates the output KPI dimension and the input
coordinate. It converts only projection coefficients, intervals and tested
domains into the parameter's native coordinate (fraction, ms, scheduling weight
or a named contrast) and the graph KPI unit. For example W/s is converted to
W/ms before multiplying a timer delta in ms; a categorical finite difference
is not multiplied by a made-up numeric category distance. Missing/incompatible
units and stale contracts make an active projection unavailable.

Current method KPI value weights are all 1. Online normalization uses one
frozen intent-reference scale per KPI across all candidate combinations. It
reads only the interpreted intent/context and current graph/recorded decision;
it never reads deployment batches, measured BEFORE/AFTER outcomes or measured
result normalization scales. No deployment needs to exist for quantification.
Non-ratio zero/missing intent references require an explicit meaningful scale;
no one-unit fallback is invented. The canonical URLLC failure fraction uses
scale 1 (or 100 in percentage coordinates), matching percentage-point semantics
even at zero. Frozen anchors are selected by earliest network snapshot, then
intent ID and value, so reordering the same active intent set cannot change the
reference scale. Raw gains keep their unit-alignment contracts. The
experimental result evaluation separately normalizes its real before/after
measurements; its scales cannot change a risk score.
The retained legacy methods are unchanged. V2 uses absolute reference obstruction
on this common normalized scale; V1 uses fractional capability loss.

## Current V2 strength score

Use the same signed intent-margin derivative Q[k,E] and complete-policy-subset
reference envelope P[k,S] defined below. For each required KPI:

    L[k,E] = max(P[k,S] - Q[k,E], 0)
    u[k] = 1 / (1 + max(current_normalized_intent_margin[k], 0))
    B[k,S] = 1 * u[k] * expectation_E(L[k,E])
    B[S] = max_k B[k,S]
    displayed_score[S] = 100 * B[S] / (1 + B[S])

The leading 1 is the equal KPI value weight. u is a separate fixed heuristic
for intent headroom, never a learned KPI value or a coefficient fitted to the
deployment outcomes. Unmet/on-boundary goals have u=1; interior goals have u<1.
The coefficient definition and the 0–100 display mapping are identical for all
groups. The raw dimensionless B is exported and shown in details. There is no
division by each group's positive potential P and no one-unit clipping before
aggregation. Negative Q can therefore cause loss larger than P. An unavailable
proposal, gain, unit or normalization still makes the score unavailable.

Single-controller own-goal mismatch is target obstruction, not an invented
inter-app conflict. A complete policy is never split into favorable fields.
V2 does not use V4's pairwise cancellation contributors as score attribution.
Topology counts remain independent metadata. Zero without an active push does
not imply intent satisfaction. Neither B nor its display score is a KPI forecast
or calibrated probability. The measured before/after evaluation remains unchanged.

## Retained V1 fractional score and shared execution model

For each required KPI k, use one scale s[k] across every combination. A
one-sided goal has signed margin m(z) = z-L or U-z. A genuinely bounded goal
has m(z) = min(z-L, U-z). Natural domain endpoints do not become optimization
objectives: the physical 0 W floor does not turn a request for power below 90 W
into a target of 45 W. At a band centre the directional derivative uses the
minimum of both boundary derivatives. An exact hold target rejects motion in
either direction. Strictness remains part of intent satisfaction; the derivative
is an influence heuristic rather than proof of satisfying a strict inequality.

For an execution state E, project each changed parameter exactly once using
the existing signed statistical reference weights, then take the derivative of
the current intent margin:

    F[k,E] = sum_p reference_project(p, current[p], executed[p]) / s[k]
    Q[k,E] = D m[k](current[k]; F[k,E])

The reference envelope only removes **complete xApp policies**. It cannot keep
one favorable field and discard an unfavorable field of the same xApp:

    P[k,S] = max over complete xApp subsets T of S,
                    and legal execution states E of T: max(Q[k,E], 0)
    r[k,E] = clip((P[k,S] - Q[k,E]) / P[k,S], 0, 1)     if P > 0
             1                                        if P = 0, Q < 0
             0, explicitly no_active_push             if P = 0, Q = 0
    score[S] = 100 * max over required k: expectation_E(r[k,E])

Thus positive base-station savings cannot dilute obstruction of the throughput
goal. Harmful contributions aligned with one another no longer score zero merely
because they agree. One xApp's own favorable and unfavorable paths are summed
before comparison; they do not fabricate a conflict with itself. Zero without
active push does not imply a capable, feasible or optimal strategy.

Shared writes produce execution states, not additive parameter changes. The
current implementation exactly integrates the **fresh uniform request**
distribution: per-UE offload selection and per-cell atomic scheduler selection.
A scheduler mode and its complete UE priority map stay together. Equal writes
merge without being counted twice, retaining their source multiplicities in
the random distribution. Independent resource distributions are multiplied.
The engine never draws random numbers or sends commands. It checks the recorded
mitigation configuration; conflicting resources without a supported arbiter
are unavailable rather than silently treated as uniform random.

Unknown proposals, missing reference projections and enumeration limits return
null, not zero. Finite uncertain and population-projected reference weights stay
usable with explicit approximation notes. Defaults cap each execution space at
4,096 outcomes and the whole reference search at 65,536 outcomes / 16 apps.
They are computation guards, not pruning of the candidate list.

Projection reports exact action coverage. A known coefficient for one UE or
DRX timer field cannot certify an additional unmeasured UE or changed field.
V2 requires complete coverage of changed components in every active admitted
chain; legacy V4 may retain a numeric known part with explicit incomplete
coverage. An intent equal to the entire physical domain is explicitly neutral,
with zero obstruction strength rather than a missing per-KPI score.

When the active KG uses significance screening, excluded coefficients do not
participate in scoring. This applies both to an entirely removed P–K relation
and to an explicitly measured action profile removed from a retained relation.
The latter requires audited action/member/component/domain applicability;
another sleep profile or UE population cannot stand in for it. A never-measured
action or an invalid unit contract still makes the score unavailable. Excluded
measurement provenance is used only for this distinction, never for its gain.
The result exports supported, excluded and unknown links for every intent KPI.
If a required KPI has no applicable retained link, any numeric score is labeled
`partial_model_score`, with KPI coverage visible in the UI. It quantifies only
the retained model and must not be presented as complete zero conflict risk.

The `significant_relations` service policy admits a P–K relation when at least
one valid measured condition has a nonzero confidence interval. Inside an
admitted relation, other valid measured action estimates remain available with
their original point values and explicit uncertain/observed-zero annotations.
This is distinct from the legacy `significant_only` policy, which screens each
action row. No entirely excluded relation is reinstated and no gain is borrowed
from another action or traffic scenario. Coverage completeness refers to the
selected graph and requested KPIs, not all physical effects. The score formula
and its independence from deployed KPI feedback are unchanged.

This is one frozen decision. Held arbiter leases, future policy inference,
traffic changes and the subsequent closed-loop window are not reconstructed.
An enabled xApp may propose no change; a cold-start independent evaluation is
a different state. UI snapshot fallback uses the earlier preview's own frame
and reports its age. Graph confidence and reference projection limitations
remain; the score is neither a calibrated failure probability nor a causal
prediction of deployment outcomes.

`experiments/evaluate_goal_obstruction.py` compares both production engines at
the same recorded predeployment frames, saves all scores, then joins the actual
measured labels. It asserts frozen controller versions and never launches a
simulation. The initial eight-group replay used slot 400 proposals, 60 ms before
the baseline at slot 520 and 80.5 ms before application at slot 561. Labels are
from the existing 500-slot evaluation windows. One seed and overlapping groups
are exploratory evidence only; no parameters are fitted to these labels.
