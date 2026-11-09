# AP energy and throughput pilot

`run_ap_energy_study.py` compares energy-first X02 (`ap_energy_rf`) with the
existing throughput-first X08 MLP. Both use real online-trained model artifacts;
the existing X08 bytes are read-only. X02 controls only `cell_sleep_mode`, while
X08 controls scheduler mode and UE weights. Both must actually execute in the
combined case. Random direct-conflict arbitration is not used.

All runs explicitly enable the same profile, with three initially active APs,
the native six moving UEs and unchanged PHY, channel and 2× traffic. The profile
in `configs/rt_sys/ap_services_reference.json` records the origin of its timing
and relative power assumptions. Absolute watts are engineering assumptions,
not measurements of the AGX or standard-mandated watts. The primary ledger is
AP static + resource-load + PA + sleep + waking + additional wake-transition
energy. Emitted RF joules are reported separately. Waking power and transition
energy are both counted as explicitly configured; the transition charge must
not be added a second time to the integral of recorded AP power.

```bash
python3 -m experiments.run_ap_energy_study \
  --output outputs/ap_energy_case_20261105 --phase all --eval-workers 3 \
  --ap-services-profile configs/rt_sys/ap_services_reference.json \
  --throughput-model outputs/x08_load2x_recovery_20261105/throughput_mlp_model.json \
  --reference-off outputs/scheduler_conflict_case_20261105/seed55_off \
  --reference-x08 outputs/scheduler_conflict_case_20261105/seed55_x08
```

The sequence is fixed: OFF, fixed empty-only negative control, isolated online
X02 training, then frozen X02 / X08 / BOTH. The first OFF is reused as the
matched evaluation OFF and warms the shared cache. Only the final three cases
may run concurrently (`--eval-workers 1|2|3`, default 1); training stays serial.
The negative control must save measured modeled AP energy with no observed
packet-KPI harm and no unavailable occupied cell. It is an analytic control,
not an RF result. OFF and X08 additionally require exact packet, UE-service and
TB-counter parity with their immutable all-active historical references; only
the UE-service host elapsed timing field is excluded from outcome equality.

The original strict v2 empty-control receipt is retained as failed. Removing an
empty AP also changes the full-band reference-measurement interference hypothesis
and CQI/AMC despite no scheduled data RF from that empty AP. Aggregate throughput
improved, but some finite-horizon UE deliveries/pending bytes changed, so strict
no-harm did not pass. Before the first RF sample, the protocol added an opt-in:

```bash
python3 -m experiments.run_ap_energy_study \
  --output outputs/ap_energy_case_20261105_v3 --phase all --eval-workers 3 \
  --allow-empty-reference-coupling
```

This accepts only structurally valid empty-cell control: actual control, positive
modeled AP-energy saving, zero occupied-cell unavailability, and all existing
energy/packet conservation audits. It changes no numerical tolerance, QoS
threshold, traffic, RF model or learned algorithm. `negative_control.passed` and
`no_observed_packet_kpi_harm` retain their strict meanings and may remain false.
All per-UE QoS deltas are recorded. Separate `empty_control_acceptance` declares
`reference_cqi_coupled_structural`, the strict failure and protocol amendment;
it must never be described as a no-harm pass. Without the flag the strict gate
remains. Frozen reuse re-audits the explicit original acceptance and raw control:
old failed receipts cannot be retroactively accepted, and old successful strict
receipts remain valid without new metadata.

Training is 4 simulated seconds, with one remaining-budget extension up to
6 seconds solely if real fitted action coverage is insufficient. There is no
performance-driven extension or hyperparameter search. One completed global
window gives one row, not three copies of the same network reward. At least
24 rows and six per action are required; the retained-row cap is 512 (expanded
from the pilot's 48 to load the released 250-row model). The four actions are
all active, empty-only deep sleep, empty-only plus micro sleep on all occupied
cells, and empty-only plus light sleep on all occupied cells. Occupied cells
never receive persistent deep sleep from this app.

The common cycle is 100 ms (LCM of 20 ms micro and 100 ms light). Each label
starts after a full common period since its actual applied ACK, with initial
deep-exit wake complete, then spans an integer number of periods with the same
start/end phase. The phase can be arbitrary; delayed report cadence need not
hit phase zero. Actual cyclic sleep and wake intervals remain inside the
measurement. Labels are measured network AP power, network DL goodput, and
UE2/UE6 unique first-failure-plus-overflow rates. They include carryover and are
not arrival-cohort failure probabilities. Bad windows remain real observations;
invalid ownership, changed membership or mixed controls cannot become labels.

The driver reconstructs slot energy components, cumulative joules, availability
and wake costs from recorded actual states, checks unavailable APs emit no
data/grants, and verifies labels against raw energy and packet records. This
transition audit covers the declared single AP owner, with one policy change
per cell per slot; hidden multiple in-slot sleep-depth changes are unsupported.
Packet-byte conservation includes queued and in-flight pending data. A separate
internal queue-versus-HARQ-reservation identity is not inferred from records
that do not publish it. Runtime HARQ tests remain a separate validation layer.

Frozen RF forecasts and selected actions are replayed from the strict JSON
artifact, as are the existing X08 forecasts. Missing reference measurements
during DTX may reduce X08 inference counts while held scheduler controls
persist; the audit uses actual decisions, not an assumed count per second.
Model hashes, source fingerprints, profile and historical references remain
protected. Separate `--phase evaluate --ap-model outputs/.../ap_rf_model.json`
can re-audit an original completed training history without retraining.

The initial frozen pilot is 1 second at calibration seed55. AP-energy reduction
and X08 goodput gain each use a predeclared 5% standalone target. Combined
energy cost versus X02 and goodput loss versus X08 use a descriptive 2% harm
threshold. These are operational comparisons, not significance tests. QoS is
reported across every UE and pooled traffic class: failure fractions use summed
unique mature packet counts and each original deadline; quantiles use pooled
completed-packet samples, not averages of UE quantiles. Pending and overflow
remain explicit. This research policy does not guarantee an SLA.

`acceptance.json` is atomically updated; each actual run automatically exports
its own replay HTML. `comparison-data.json` contains the four-case network,
traffic-class and per-UE results, energy/RF separation, actual control evidence,
factorial interaction and provenance. Disjoint controls and non-additivity alone
do not establish M2: the report requires actual control and standalone objective
evidence, reports target harm separately, and may conclude no detected conflict.
Longer runs or new seeds require an explicit budget and corresponding matched
historical all-active parity references. `--dry-run` starts no jobs.
