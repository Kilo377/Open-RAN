# Independent quantification study

The study started on 22 November 2026 at 01:47 UTC with a ten-hour budget.
The production UI retains its audited `goal_obstruction_v2` default while the
research methods are tested separately. No final combination selector is added.

Every score consumes a frozen intent interval, the common BEFORE snapshot,
trained xApps' recorded shadow proposals, the screened KG and the configured
random arbiter. It cannot consume a deployed combination's AFTER measurements.
The original four methods and their complete input/code hashes are sealed
before nonempty controlled combination runs are admitted. The later fifth
method has a separate forward-only boundary, described below. AFTER measurements are a separate validation
channel; they do not fit score weights or parameter–KPI coefficients.

The primary research method, `net_intent_obstruction_v1`, forms the normalized
joint reference influence vector after accounting for shared control resources
once. It averages arbitration outcomes before applying the intent functional.
For KPI k, let z_k be that reference influence, divided by one common KPI scale.
One physical-domain projection applies to the whole expected joint vector.
A single effective boundary gives U_k = direction_k × z_k. A genuine bounded
interval J_k gives U_k = distance(0, J_k) − distance(z_k, J_k), where J_k is the
requested interval translated by the current observation and divided by the
same scale. A neutral full-domain request contributes zero.

If any U_k is negative, U is the worst U_k; otherwise U is their equal mean.
The obstruction strength is D = max(0, −U), displayed as 100D/(1+D).
The signed reference effect 100U is exported separately. The scale is a frozen
finite intent reference, with probability metrics using their canonical unit
range instead of dividing by a possibly zero observed error rate. These values
are dimensionless first-order reference effects, not forecasts of future KPIs.
Direct/indirect conflict counts are diagnostics and have no separate weights.

The temporal comparator uses only same-time complete shadow proposals in the
preceding 500 slots. Its intent origin, requested intervals and KPI scales stay
fixed. It neither mixes different xApps' timestamps nor multiplies a gain by
the observation horizon. The extra `net_profile_gain_obstruction_v1` comparator
dispatches the same net formula with independently measured joint sleep-profile
contrasts. It keeps the original graph topology, candidate pool and input
snapshots; it does not introduce another scoring formula.

The original three methods retain their earlier archived projector. The profile-G
fourth method and the fifth method use the subsequent empty-cell scheduler-bundle
correction. Consequently, a four/five-method comparison is a comparison of
complete frozen versions, including these implementation differences. The
fourth-versus-fifth pair alone holds both G and the projector fixed to isolate
the observation-aggregation functional. Archived module hashes make this boundary
explicit; no earlier score is recomputed under the newer projector.

The fifth method, `aggregation_aware_obstruction_v1`, is a separately registered
addition after original outcome availability. It uses the same profile G,
recorded states/proposals, intent intervals and candidate flags as the fourth
method. Its predictions cover all 1,980 derived cases, but independent forward
validation is restricted to 111 physical jobs (1,665 intent cases) proven never
started in both metadata scans bracketing scoring. The other 21 jobs (315 cases)
are diagnostic only. They cannot enter this method's independent validation.
There are no additional physical runs or independent seeds for this addition.

The fifth method explicitly opts into `current_observation_v1` coordinates.
This helper takes the current observed frame as reference, without falling back
to earlier intent anchors or actual deployment feedback. Ordinary KPI scales
are the magnitude of that observation; an ordinary zero observation remains
unavailable rather than receiving an invented scale. The failure ratio retains canonical percentage-point
scaling. Earlier net methods retain their frozen earliest-finite-intent-reference
scales, even when current observation differs. These coordinates happen to agree
on the sealed common-origin inputs in this study; that is not a general identity.
The helper and fifth method are opt-in; the UI default and earlier archives stay
unchanged.

For mean-valued observations, the fifth method builds each joint reference
outcome, applies its physical clamp, then forms the probability-weighted mean
before applying the intent functional. For an extrema-constrained observation,
it translates the entire observed baseline [minimum, maximum] by each positive-
probability joint reference shift, clamps the endpoints, and takes their support
envelope. A no-op therefore preserves the original range. This baseline-range ×
action-support envelope is conservative: it does not assert that both extrema
will occur within the finite AFTER window. The same signed progress aggregation
and 100D/(1+D) display mapping are then applied. It is a reference obstruction
model, not a closed-loop KPI predictor or an intent failure probability.

Two forward comparisons were fixed before reading their future outcomes: all
five methods on their joint common numeric pool, and the fourth/fifth same-G
ablation on its own two-method common numeric pool. Temporal-history gaps in a
different method do not silently remove otherwise valid fourth/fifth cases.
Both comparisons retain every requested stratum, including undefined correlations.
The original four-method full-matrix analysis remains separate.

Three traffic contexts, two new validation seeds and twenty fixed physical arms
are shared by fifteen intent tests, including twelve non-preset intents. Each
physical run has the same native prefix, a common baseline at slot 2040 and a
2000-slot AFTER window. Original OFF-prefix and exogenous-arrival proofs guard
the origin. Reusing a physical run across intents does not create independent
replicates. Score coverage, constant-risk strata, ties and unresolved KPI
references remain visible in the report rather than being dropped to improve
correlation statistics.

A separate prospective coverage supplement adds the BCE and BCDE arms to each
of the six origins (twelve physical runs). Here B is the AP-energy xApp, C the
throughput scheduler, D the MEC-latency scheduler and E the URLLC scheduler.
This supplement was selected from BEFORE-only score variation: the original
matrix's throughput-plus-BS-power intent had only BC/ABC candidate arms with
identical scores. It preserves the original twenty-arm plan and seals, reuses
the certified OFF origins, and has its own prediction/admission gate. It does
not create additional independent seeds. Its results are distinguishable from
the original preregistered matrix.

The added sleep-profile calibration uses separate training seeds and paired
REF/profile runs. Only `cell_sleep_mode` is requested once; the native dispatcher
and its normal receipts remain unchanged. The paired post-settle window is
(2440, 4440]. Student intervals use independent seed pairs, not individual slots.
An undefined MEC task mean stays missing. A whole-cell profile is one finite
contrast and is not divided into invented per-cell derivatives. New conditional
rows can be appended only to relations already present in the frozen KG.

Risk is dimensionless on a 0–100 display scale, not a probability. Actual Metric
is separate, zero-anchored observed intent progress in normalized points. It
uses only requested KPIs: a regression selects the worst signed KPI change;
otherwise the equally weighted mean is used. A single active boundary scores
directional margin change, while a bounded band or point scores reduction of
distance to the set. URLLC ratio changes are percentage points, not relative
changes divided by the old error rate. Actual deterioration is
D_obs = max(0, −Metric/100). Neither a zero risk nor a zero Metric establishes
strict interval attainment; its boolean test is reported separately.

Actual measurements use equal matched windows. Mean/counter statistics and
full-window extrema retain their recorded semantics; a missing mature MEC task
cohort is unavailable. These observations are not adjusted by the OFF trajectory.
OFF natural drift is shown separately. Related reference/Evaluation geometry
does not guarantee future ranking in a nonlinear closed loop. Risk zero does not imply target attainment;
strict-boundary violations, background drift, arbitration tails, missing action
calibration and sparse completed-task cohorts are reported explicitly.

Only two independent seeds are available. Reassessing one physical run against
15 intents, five methods or several report partitions does not create independent
replicates. Macro summaries weight intent × scene × seed strata equally; no IID
interpretation of derived rows or confidence/generalisation claim is made.

The reproducible entry points are `experiments.study_quantification`,
`scripts/run-quantification-study.py`, `scripts/run-sleep-profile-calibration.py`,
`experiments.fit_sleep_profile_gains`, `experiments.report_quantification_study`,
`experiments.analyze_quantification_study`, and `experiments.forward_method_analysis`.
The local preregistrations, sealed inputs, raw runs, checkpoints and reports are
retained under `Desktop/ORAN/SMO/quantification-study-20261122`.

## Completed validation

All 132 physical runs completed on 22 November 2026 with no formal-run failures:
120 original-matrix runs and 12 separately registered supplement runs. The
8,952 retained raw files (6.52 GB) passed checksum verification. Fifteen intent
tests include twelve non-preset intents. There are only two independent
validation seeds and six scene/seed origins; the derived intent rows are not
independent replications.

The table reports descriptive macro Spearman correlation between predicted
obstruction and observed deterioration, on the same numeric candidate cases
and the same defined intent × scene × seed strata for all four methods.
The original matrix has 151 common rows and 14 common defined strata; the
supplement has 45 common rows and 11 such strata. These are separate pools.

| Frozen method | Original matrix | Supplement |
|---|---:|---:|
| Goal obstruction V2 | 0.642 | -0.091 |
| Net intent obstruction | 0.674 | 0.273 |
| Temporal net obstruction | 0.690 | 0.273 |
| Net with joint sleep-profile gains | 0.724 | -0.091 |

The net formulation has positive descriptive association in both pools, but
this does not establish stable generalization or uniform superiority over V2.
In the separately forward-sealed comparison, the fourth and fifth methods
have identical within-stratum rankings on their common samples. The aggregation
correction changed some score magnitudes without improving those rankings.

Two severe MEC-delay false-safe cases remain: the native-seed baseline mean
10.5625 ms increased to 38.5294 and 45.9063 ms for DE and CDE, while their
predicted obstruction was exactly zero. OFF changed only to 10.9412 ms. The
screened KG omitted a large but uncertain scheduling-mode contrast, and the
remaining local priority gain did not represent subsequent policy/FIFO queue
behavior. This is a model-coverage limitation, not proof of a zero influence.
No deployed AFTER value was fitted back into a score, gain or selection rule.

The production UI now exposes the unchanged `net_intent_obstruction_v1` as an
optional method. Its default remains `goal_obstruction_v2`. Temporal scoring
remains offline because production does not yet provide verified historical
frame/proposal packets. The frontend integration changes no graph, candidate
selection, Evaluation or frozen scientific scoring engine.

Full artifacts and the two independent final reviews are retained in
`Desktop/ORAN/SMO/quantification-study-20261122/analysis/`; the UI release and
its state-preservation evidence are in `ui-net-option-v1/`.
