# Scoped MEC response approximation

`experiments.fit_mec_local_response` reuses retained completed runs and fits
`smo.mec_empirical`. The response is mean offloaded MEC task end-to-end latency
over 500 slots (250 ms). Inputs are executed per-cell scheduler modes and
per-UE weights, reconstructed from accepted control receipts, not enabled
xApp labels or proposed-but-rejected actions.

Only runs with native DRX, offloading and AP policies throughout the preceding
history and observation window are pooled. Source/configuration hashes, actual
UE associations, traffic/task births and the pre-intervention trace establish
compatible contexts and matched time strata. Exact executed trajectories are
deduplicated. Missing mature-task outcomes are excluded, never replaced with
zero latency. Every extraction is checkpointed with source hashes; source
runs remain unchanged.

The model uses time-stratum intercepts and regularized linear response in the
observed SVD subspace. Nested validation leaves an entire executed trajectory
out, including every repetition of it, and chooses regularization inside the
training fold. Individual slots/tasks do not count as independent experiments.
The exported model includes rank, support, validation predictions and data
provenance. A regression coefficient in an unobserved direction is not an
identified causal parameter gain.

The conflict module prefers applicable controlled gains. When those cannot
describe a joint scheduler action, it can use this model only for MEC latency,
the matching network configuration and 250 ms horizon, native other controls,
observed feature ranges, and supported action/operating-point directions.
It evaluates each composed random-arbitration outcome once. It does not
invent individual-parameter gains, rank/prune combinations, deploy controls,
or overwrite the measured result blocks. The UI labels regression separately.
The descriptive contrast envelope is twice the maximum held-out prediction
error on either side; it is not a confidence interval or SLA probability.

Reproduce extraction and fitting:

```sh
python -m experiments.fit_mec_local_response \
  --jobs-root "$JOBS" --output "$OUTPUT" --workers 4 \
  --supplement-root "$SUPPLEMENTS_AGX1" \
  --supplement-root "$SUPPLEMENTS_AGX2"
python -m smo --replay "$RUN" --influences "$GAINS" \
  --empirical-response "$OUTPUT/model.json"
```

The 2026-11-19 minimal study reused 126 runs and added four fixed scheduler
conditions, split between two AGXs. Three new conditions had usable labels;
the URLLC-only condition retained one unfinished mature MEC task in the
250 ms window and was excluded. Two earlier runs whose stimulus was rejected
are retained separately for audit and are not counted as calibration data.
Final extraction: 130 runs, 35 usable windows, 15 unique trajectories, rank
9/12, two time strata, one seed. Nested held-out MAE: 5.612 ms versus
7.489 ms for the stratum-mean baseline (25.1% lower); direction agreement:
80%. These are conditional same-seed results, not independent-seed validation.
Small predicted differences remain unresolved at this error scale. The four
new simulations completed in about 3.5 minutes with two devices in parallel,
excluding transfer and analysis; no learned xApp or simulator configuration
was changed globally.
