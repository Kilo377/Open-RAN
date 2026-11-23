# Multi-point population response identification

This extends the retained group perturbation pilot. Each numerical control is
sampled across a declared domain before fitting a first-order approximation.
The domain and sampling levels are fixed before inspecting the new results.

| Common group control | Levels | Fixed scheduling context |
|---|---|---|
| MEC offloading probability | 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1 | Round robin |
| DRX inactivity timer / ms | 0, 2, 4, 8, 16, 24, 32, 40 | Round robin |
| MEC relative scheduling priority | 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2 | Latency aware |
| URLLC relative scheduling priority | 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4 | QoS deadline |

Probability and priority bounds follow their actual accepted control ranges.
The inactivity timer has no finite model maximum: 0–40 ms is an explicitly
chosen calibration domain, one native DRX cycle, rather than a claimed hard
upper limit. Native DRX-disabled traffic remains disabled. Priority is common
within the complete service class, relative to other traffic at weight 1.
Multiplying every UE's weight together would preserve scheduling order.

Three independent seed blocks each contain 30 distinct conditions. Compatible
points from the completed two-seed pilot are reused only after runtime source
and executed-vector validation. There are 90 fit observations: 22 reused and
68 newly simulated. Three additional cross-device bridge runs are quality
checks and do not increase the statistical replication count. Each seed stays
on one hardware/software environment. The second device must reproduce three
native/scheduler reference conditions before it can admit its new seed block.

Runs retain the existing 6 mobile UEs, native 1× traffic, accepted fast system
PHY, depth-2 channel, capacity, deadlines and frozen learned model files. All
xApps/rApps are off. Only one RCP changes through normal RIC admission and
one-slot actuation delay. The duration is 800 ms: request at 100 ms, scoring
over `(200,700] ms`, terminal task follow-up to 800 ms. Per-case auditing checks
the held control in all 1,000 scoring slots, raw checkpoint hashes, paired
exogenous inputs, byte accounting and mature task cohorts.

For a numerical control, fit `KPI(seed,x) = a(seed) + beta*x + residual` with
ordinary least squares. Complete levels have the same three seeds and receive
equal weight. This balanced design makes the common slope equal to the mean
of the three separately fitted seed slopes. The 95% Student interval uses
these **three independent slopes**, with two degrees of freedom; grid points,
slots, packets and task events are not independent replications.

Retain every measured point, each seed fit, the shared slope, tested/valid
domains, within-seed R², RMSE, maximum residual and leave-one-level-out RMSE.
Flag the first-order approximation as limited when R² < 0.7 or held-level
RMSE exceeds 30% of the mean within-seed response span. These thresholds are
diagnostic heuristics, not hypothesis tests. A precise nonzero average slope
does not establish monotonicity or a useful local linear control model.
The control-eligibility flag also requires the slope interval to exclude zero;
adequate linearity alone does not resolve a noisy response direction.
Constant observed responses are explicit and are not evidence of universal
independence. No extrapolation beyond the valid domain is supported.

Energy sums the entire UE population; demand sums all MEC sources; delays and
success ratios pool their relevant event counts. At zero offloading, latency
of offloaded tasks is undefined. Such values remain null, never zero, and a
level is fitted only if its KPI is finite in all three seeds. Completed-task
latency remains conditional on completion and should be read with success and
pending-task counts. If fewer than four complete levels remain, no slope is
published for that response.

Goodput, AP power, failure-event rates, packet completion latency and realized
MEC demand use scoring-window counter deltas. Queue size and maximum HOL are
read at 700 ms. The URLLC failure fraction remains the native cumulative
event/offered ratio through 700 ms, not a mature arrival-cohort SLA estimate.
Expected MEC demand is analytic from configured task rate, compute and the
executed offloading probability; its exact slope is not independent empirical
validation of the radio/task path.

Categorical scheduling and cell-sleep actions remain named finite contrasts;
their category IDs are never treated as a continuous axis. Their intervals
also use three seed pairs. The graph shows range fits with measured points,
units, domains and residual diagnostics. Line certainty and linear fit quality
are separate: an interval excluding zero supports an average direction, not
an exact causal derivative at every operating point.

Raw data, checkpoints, source/model hashes, caches and the previous study are
retained. The low-disk worker uses a private RAM workspace; sealed checkpoints
are continuously copied to durable local storage with transport and raw-file
SHA-256 verification. Keep a second durable copy before releasing that worker.

```bash
python experiments/run_rcp_grid.py --seeds 20261116 20261117 \
  --reuse /path/to/retained-pilot --output outputs/study --parallel 2
python -m smo.grid_identify --studies /path/to/agx1 /path/to/agx2 \
  --reuse /path/to/retained-pilot --output /path/to/population-grid
python -m smo --view population --replay /path/to/native-run \
  --influences /path/to/population-grid/influences.json \
  --export /path/to/intent-graph.html
```
