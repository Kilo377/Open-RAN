# MEC offload-first pressure scenario

Use `--ue-services drx-mec --ue-services-config configs/rt_sys/ue_services_mec_offload_first.json`
for every arm of a Base / analytic-rule / frozen-QGBDT comparison.
The CUDA wrapper forwards both options unchanged.

The explicit profile differs from the native `ue_services.json` only in UE4's
default `offload_fraction`: 1.0 instead of 0.5. Native defaults are unchanged.
Task arrivals remain 20/s, work 0.05 GFLOP/task, deadline 100 ms, and the
UE4 expected offload-workload quota remains 0.75 GFLOP/s. Thus the configured
uncontrolled demand is 1.0 GFLOP/s; routing at p=0.5 offers 0.5 GFLOP/s.
These are policy-budget values, not evidence of physical server congestion.

The runner records the chosen profile path and SHA-256 in `ue_services_profile`
in its dry-run output and `config_snapshot.json`. Audits must also compare the
resolved service configurations of all arms. Task outcome, energy and pending
counts must accompany quota compliance; completed-task mean delay alone can
hide failures or unfinished work. Retain the original native zero-gain result.

Frozen QGBDT and analytic rules use the same feasible candidate set. Report any
match between those arms; quota compliance alone does not establish a benefit
from learning. No change to task-profile validation or the learned algorithm
is needed for this scenario.
