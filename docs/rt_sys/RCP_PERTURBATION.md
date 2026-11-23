# Local RCP influence identification

This pilot measures the existing network model, with no xApp training,
algorithm selection, PHY simplification or load tuning. All six learned xApps
remain OFF. A fixed test driver uses the ordinary RIC admission, actuation
delay, transaction validation and execution-receipt path.

## Preregistered design

Two seeds (`20261116`, `20261117`) × eleven arms × 0.8 s = **17.6 s simulated**.
Every case uses three cells, six moving UEs, the accepted fast system PHY,
depth-2 ray tracing, native 1× traffic and unchanged radio/task deadlines.
The explicit existing AP power profile and native UE/DRX/MEC models are enabled.

All arms share the native state for the first 100 ms. The intervention request
is issued at 100 ms, follows the ordinary one-slot execution delay, and remains
held through the end. The score interval is `(200,700] ms`: 100 ms wash-in,
500 ms measurement and 100 ms terminal follow-up. The endpoint is 600 ms after
the requested intervention. This is a local finite-horizon response, not an
instantaneous or globally stationary derivative.

| Contrast | Perturbed control | Other controls |
|---|---|---|
| Offload | UE4 probability 0.4 / 0.5 / 0.6 | Native RR, native DRX, active APs |
| DRX | UE4 inactivity 8 → 4 ms | Other DRX fields fixed; native RR, offload 0.5 |
| AP sleep | All cells active → micro sleep | Existing 1 ms per 20 ms micro profile |
| Scheduler latency | RR → latency aware | All UE weights remain 1 |
| Scheduler QoS | RR → QoS deadline | All UE weights remain 1 |
| MEC weight | UE4 weight 0.75 / 1 / 1.25 | Latency-aware mode held fixed |
| URLLC weight | UE6 weight 0.75 / 1 / 1.25 | QoS-deadline mode held fixed |

DRX is deliberately unlocked by the **experiment only**. The released X-01
deployment still keeps DRX native-locked. Weight derivatives are per-component;
they are not derivatives with respect to every UE weight simultaneously.

The complete stimuli and contrasts are recorded before execution in `plan.json`
and individual arm specifications. Each numerical contrast changes exactly
one RCP; categorical controls use named action contrasts. The two native arms
run first to generate each seed's shared RT cache. Independent cases run two
at a time; no global environment, package or hardware configuration is changed.

## Identification and uncertainty

For a numerical RCP, fit the seed-blocked local linear model
`KPI[s,x] = intercept[s] + beta × (x - x0) + residual` on symmetric low/center/high
levels. With this balanced design, `beta` is the mean of the seed-paired central
differences `(KPI_high - KPI_low) / (x_high - x_low)`. The center arm provides a
linearity-residual diagnostic; it is not used to disguise a nonlinear response.
Categorical/structured actions use mean paired differences without calling
them numerical partial derivatives.

The independent unit is a **seed pair** (`n=2`), not a slot, packet or cohort.
Store both paired outcomes, their range, sign agreement and a descriptive
paired Student interval. Its two-sided 95% critical value has one degree of
freedom (`12.706`), so intervals may be very wide. The paired-observation
procedure and t critical values follow the
[NIST paired-analysis description](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)
and [NIST t table](https://itl.nist.gov/div898/handbook/eda/section3/eda3672.htm).
This two-seed pilot cannot establish robust uncertainty or a globally zero
effect. Equal zero outcomes are explicitly labelled as unresolved in this pilot.

Keep separate estimates for different scopes, scheduler contexts and action
contrasts. Per-UE goodput is a vector with per-UE effects; opposing responses
are never replaced by a misleading average.

Only IEEE counter-ratio roundoff is suppressed before dividing a paired
difference: relative tolerance `1e-11`, absolute `1e-12` in the KPI's unit.
Full unrounded input pairs are retained. These tolerances are far below the
packet, slot, energy and latency resolutions and are not performance thresholds.

## Required evidence

Before publishing any estimate, the analyzer checks:

- Completed status, automatic replay audits, pinned source hash and all
  per-run checkpoint file SHA-256 values.
- Exactly matched initial exogenous environment, RT cache key, exogenous
  arrival sequence, task birth sequence and native pre-intervention state,
  separately for each seed. Endogenous task routing and HARQ outcomes are
  allowed to change as consequences of the intervention.
- All 1,000 scoring slots retain the exact requested DRX, offload, scheduler,
  weight and AP-policy vector. Rejected controls invalidate the case.
- Final packet-byte conservation and five mature scoring cohorts per MEC UE.
  All recorded terminal outcomes through 800 ms remain available. Censored or
  failed tasks stay in the success denominator and separate diagnostics;
  completion-delay means use completed tasks only.
- No unrelated rApp/xApp activation or learner fitting. Existing model
  checksums are also verified throughout the sweep by the runner.

## KPI definitions

Energy is the UE0+UE4 counter delta over the score window. AP power and
goodput are energy/byte counter deltas divided by the actual window duration.
Packet delays weight by completed-packet count. MEC task means use the five
eligible birth cohorts, conditional on completed/offloaded-completed tasks;
success, pending and failure counts are retained. Changing offload can change
the population of the conditional offloaded-delay KPI.

Queue and HOL use the completed score-end state. Recorded PHY queue values
precede same-slot delivery, so the graph and analyzer use the post-delivery
packet ledger. UE/AP activity fractions use score-window counter ratios.
Expected MEC demand is derived from the **executed** offload fraction and native
task profile; realized demand uses observed generated offloaded-task counters.
URLLC failure-event rates include disjoint overflows and carryover. The
failure fraction is the cumulative score-end event/offered ratio, not a mature
arrival-cohort SLA probability. Energy, power and service-delay conclusions
apply to the configured simulation models, not physical AGX device measurements.

## Reproduce and retain

```bash
python experiments/run_rcp_perturbation.py --output outputs/rcp-pilot --parallel 2
# Continue a checkpointed sweep; completed cases are retained and verified.
python experiments/run_rcp_perturbation.py --output outputs/rcp-pilot --resume
python -m smo.identify --study outputs/rcp-pilot --output outputs/rcp-identification
python -m smo --view entity --replay outputs/rcp-pilot/cases/s20261116_native \
  --influences outputs/rcp-identification/influences.json --port 8891
```

Each case retains raw packet, PHY, control, task and traffic records plus its
automatic self-contained network replay. The sweep keeps an immutable plan,
per-case file-hash checkpoints, logs and progress. Resuming preserves an
incomplete attempt in `partial/` before retrying. A disk reserve blocks new
cases without deleting existing data.

The analyzer writes the audited measurement dataset, scalar CSV, complete
effect matrix, graph-ready estimates and identification report. Existing
contract candidates receive estimates even when unresolved/zero. Extra
exploratory graph links require an interval excluding zero in a tested
contrast; they carry full provenance and no multiple-testing significance
claim. The complete matrix includes results not promoted to graph links.

For the population monitoring view, run the analyzer with `--view population`
into a separate output directory. This recomputes all-UE energy, mean UE
goodput and total workload compute demand from the same immutable records.
Its graph import includes only group-wide control contrasts. It retains the
single-member probes in the full matrix without promoting them to population
control effects. Existing trained xApp policies and the experiment design are
unchanged.
