# Rule-based v1: twelve measured-feedback xApps

This is an opt-in deterministic functional library. It does not reproduce the
trained algorithms or published gains of its sources. No model training,
pretrained weights, BLER lookup tables, online decoder probes, CSI access or
future observations are used. DU OLLA, actual NR decoding, none/Chase boundaries,
RT precision and existing 5/10-slot controller invocation periods are retained.

## Usage and source traceability

The machine-readable `rule_based_v1_catalog.json` records each function's goal,
required measured KPIs, owned actions, source links and default engineering
parameters. The `rule_based_v1` section in the dedicated RT JSON presets selects
applications, static scope and per-application parameters. Legacy xApps and
rApps are not implicitly registered. Use `--control-mode orchestrated` to execute
the presets; `baseline` explicitly disables all applications.

Example (Spark, raw output must remain there):

```sh
.venv-spark/bin/python experiments/run_rt_sys.py \
  --config configs/rt_sys/rule_based_v1/business_resources.json \
  --control-mode orchestrated --scene simple_street_canyon --cells 3 --ues 4 \
  --seed 512 --slots 4000 --initial-queue-bytes 0 --nr-execution batched \
  --output outputs/rule_based_20261015/example
```

Functions: throughput frequency-power allocation (COMIX functional adaptation),
RF power saving (COMIX/Orange), guarded migrate-then-sleep (Orange), directed CIO
load balancing (AI-powered conflict-management functional example), measured
traffic steering (OSC/multi-metric handover/distillation), event-based MRO,
weighted max-min UE PRB caps (srsRAN/distillation), soft slice-SLA weight control
(NexRAN/ORANSlice), HOL/deadline guard, graph-colored frequency power coordination
(graph spectrum allocation/COMIX adaptation), RET coverage feedback and measured
legal-codebook selection. The last two are local functional baselines, not
claims that the cited projects supplied those exact algorithms.

The four local PDFs retain the source classifications in
`PAPER_FIELD_TRACEABILITY.md`: trained decisions, synthetic P/K fields and
conflict annotations are not ported. The multi-metric handover artifact uses
weights motivated by offline RF analysis: v1 uses explicit equal weights instead.
Port ON/OFF, actual admission requests, cross-frequency control, uplink, TCP and
whole-device power are outside the model and are not emulated by unrelated fields.

## Observations and timing

All inputs pass through the existing public independent projection and report
delay. Per-entity validity/sample/age metadata controls decisions. Invalid UE
inputs are skipped individually, never filled with zero. Sleep holds active
when coverage evidence is incomplete. Unconnected and weak UEs remain in overall
statistics. Optional demand, BLER and deadline samples are never invented.

An opt-in producer publishes executed configuration/fingerprints, bounded public
mobility events (one second), same-window business counters and legal-codebook
reference reports. Configuration is public executed state, not wireless truth.
Beam reports are analytical reference-transmission hypotheses using the same RT
CFR inside the producer; they incur no extra actual transmissions or decoding,
and traverse the same noise/EWMA/delay chain. They are not NR CSI feedback.

Controller calls remain 5/10 slots. Default minimum parameter changes are 20 ms,
sleep evaluation 100 ms and state dwell 200 ms, MRO/RET 200 ms. TS requires three
distinct reports and 100 ms per-UE request cooldown plus the existing mobility
state machine's dwell protection. Throughput history uses a 100 ms EWMA. MRO
requires five completed attempts in its one-second window. A->B->A within 100 ms
is the experiment's ping-pong definition, not standardized MRO failure attribution.
Target-unavailable failures do not become too-late handovers.

## Actual execution and policy integration

Multiple cell-scoped ConfigActions per application are now collected and
validated; commands are collected once per decision. Rule requests remain
subject to existing field priorities and physical transaction checks. Requests
with RF vectors exceeding a scalar/cap budget are explicitly rejected, not
renormalized. Frequency restrictions can legally defer an existing HARQ contract.
Slice control also selects `slice_aware` so that its weights have a real consumer;
the latency application can supersede that policy. Soft weights are not hard
slice quotas, and UE PRB caps do not guarantee allocation.

The read-only KPI monitor reports validity and declared SLA breaches, not causal
conflicts. `PolicyUpdate.rule_based_parameters` carries explicit per-kind or
per-name numeric overrides and survives `PolicyStore` snapshot copying. Existing
enabled/disabled xApp selection and scope contracts apply; a separate dynamic
capability catalog exposes the new namespace without changing legacy defaults.
No new non-RT decision algorithm has been installed.

Rule leases retain lower-priority live requests for deterministic arbitration.
On disable they clear that application's state and restore another live request
or the first delivered pre-control configuration for fields the library owned.
Queued commands/configs from withdrawn owners are rejected before execution.
Commands already executed are never reversed. Sleep restore to active follows
the existing sleep/power transaction model. A restored power vector is explicitly
cleared if the baseline was scalar. External controllers should use the same
arbiter; out-of-band mutations cannot supply coherent ownership semantics.

RET changes at most one cell at a time, 1 degree within 0--15 degrees. Evaluation
waits for executed configuration and settling; concurrent configuration changes
make the effect unattributable. It holds if its action was rejected or superseded.
Beam control yields when RET owns the analog subspace. Digital RZF stays in the
physical constrained subspace. No sandbox counterfactual decoder execution occurs.

## Acceptance and unattended long runs

`validate_rule_based.py --stage short` checks 44 128-slot fixtures: disabled
4/24-UE regressions, twelve single xApps on both maps and four combinations on
both maps at 4/24 UEs. The disabled cases are compared with the frozen original
source. Local deterministic/known-channel tests exercise decisions that natural
64-ms RT fixtures may not trigger (sleep dwell, MRO samples, handovers).

Only after the short gate passes may `--stage long --gate ...` execute 48 cases:
24 single-app runs, 16 four-profile runs (seeds 512/513), all 4 UEs/4000 slots;
8 pressure runs, 24 UEs/1000 slots. Warmup is 200 ms. Non-delivery, overload,
insufficient mobility samples and no natural trigger are reported, not erased.

Before launch, give a short-measurement runtime range, worker count and separate
RT generation evidence. Launch with nohup/setsid and external driver log, then
stop agent polling; no scheduler/automation is installed. The driver itself
checks source invariance, GPU execution, decoder/probe counts, power/resource/TB
and packet conservation, persists failure status, and stops remaining children
on serious failure. Completion and algorithm benefit are separate findings.
Source archives, configuration snapshots, per-case logs/receipts and a verified
SHA256 inventory stay only on Spark. Mac holds source, presets and light reports.
