# Trained Chongyu xApps · 20261109

Six portable JSON checkpoints with measured training rows, fitted parameters,
SHA-256 provenance and a short target-KPI evidence record. No pickle is used.

| Model | Measured / fitted rows | Target KPI | RCP |
|---|---:|---|---|
| X-01 RF | 221 / 221 | UE0 + UE4 energy | UE offload fraction; native DRX retained |
| X-02 RF | 250 / 250 | Whole-network AP energy | Cell sleep mode |
| X-08 MLP | 254 / 254 | Network delivered downlink goodput | Scheduler mode and UE weights |
| X-12 MEC RF | 212 / 212 | Completed offloaded-task E2E latency | Scheduler mode and UE weights |
| X-12 URLLC RF | 207 / 200 | UE2 + UE6 deadline failures and packet latency | QoS scheduler and UE weights |
| X-13 QGBDT | 13 / 12 | UE4 expected MEC demand within 0.75 GFLOP/s quota | UE offload fraction |

Install from the repository root, using the project's existing Python environment:

```bash
python scripts/orin/install-trained-xapps.py
```

The installer verifies every release checksum, stages checkpoints into
`outputs/trained_xapps/20261109/`, and refuses to overwrite a changed installation.
It writes `runtime-arguments.json` for all six apps and `models.json` for the
four-model evaluation driver. Runtime model files are under `models/` in that
output directory. The simulation requires working Sionna and the appropriate
AGX/DGX environment; the installer itself uses only the Python standard library.

For example, run the MEC scheduler alone:

```bash
bash scripts/orin/run-six-ue-cuda-nr.sh --model fast --duration-s 3 --seed 211 \
  --xapp-pool chongyu --ue-services drx-mec \
  --latency-xapp-mode frozen \
  --latency-xapp-model outputs/trained_xapps/20261109/models/x12_mec.json \
  --output outputs/x12_mec_release_check
```

URLLC uses the opt-in `--urllc-xapp-policy-version qos_v2`; its checkpoint is
incompatible with the legacy packet-RF family. X-02 also requires the explicit
AP service model, e.g. `--ap-services-profile configs/rt_sys/ap_services_reference.json`.
X-01, X-12 MEC and X-13 require `--ue-services drx-mec`.

X-12 MEC is retrained using the completed 30-second record. Its prior model's
10-second KPI improvement is not validation of the new checkpoint; the release
manifest distinguishes fit/load verification from a fresh network evaluation.
Other models retain their individually measured evaluation scopes in `evidence/`.
X-13 uses an expected-workload operator quota, not physical server admission;
the analytic guard matches QGBDT in the recorded pressure test. URLLC's fixed
weak QoS policy likewise matches its RF burst-scene success rate.

The primary acceptance criterion is each xApp's target KPI; other KPI changes
are recorded tradeoffs for conflict studies. Results use different workloads,
horizons and seeds and do not rank the xApps or establish learned superiority.

Full simulation logs and checkpoints remain on the original devices and in
verified Mac backups. They are retained separately from Git. The release records
their archive SHA-256 values; numeric model training rows remain in each JSON.
The [data retention index](data-retention.json) lists the retained
run archives, including the accepted shorter X-02 recording.
