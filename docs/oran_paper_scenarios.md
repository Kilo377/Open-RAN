# O-RAN Paper Evaluation Scenarios

This document defines the paper-facing O-RAN scenario layer added before the
method/baseline alignment work. The goal is to provide a stable evaluation
substrate for reflective memory, failed-case avoidance, candidate repair,
conflict handling, and Near-RT tuning experiments.

## Scenario Matrix

The main matrix is `oran_paper_eval_suite` in
`scenarios/scenario_matrix_templates.py`.

The matching manifest is:

- `experiments/manifests/oran_paper_eval_manifest.csv`

The matching runner config is:

- `experiments/configs/oran_paper_eval.yaml`

Validate expansion with:

```powershell
C:\Users\Asus\miniconda3\python.exe -m experiments.run_cross_scenario_generalization --config experiments/configs/oran_paper_eval.yaml --dry-run
```

## Scale Tiers

| Case | Cells | UEs | Purpose |
|---|---:|---:|---|
| `oran_small_debug_mixed` | 3 | 24 | Debug and low-cost ablations |
| `oran_medium_nominal_mixed` | 7 | 42 | Main reference case |
| `oran_medium_dense_hotspot_high_mobility` | 7 | 56 | Load steering and mobility conflict stress |
| `oran_medium_urllc_burst` | 7 | 48 | Low-latency protection stress |
| `oran_medium_interference_edge` | 7 | 48 | Weak SINR and inter-cell interference stress |
| `oran_medium_hotspot_shift_blockage` | 7 | 50 | Hotspot shift and blockage-style NLOS stress |
| `oran_large_19cell_scalability` | 19 | 120 | Scalability and latency-overhead study |

The medium 7-cell / 42-UE case is intended as the primary paper reference
case. The 19-cell case is reserved for scalability and should not be used as
the first correctness benchmark.

## Traffic and Slice Mix

Each case includes mixed:

- eMBB/video or broadband users
- URLLC/control users with 8-10 ms latency budgets in burst scenes
- mMTC/IoT users with lightweight periodic traffic
- phone/voice-like users with intermediate latency budgets

The UE population is deterministic so seed changes affect channel and control
randomness without changing the scenario identity.

## Radio and Channel Stressors

The suite covers:

- FR1 macro/sub-6 cells
- FR2 mmWave small cells
- normal mixed traffic
- dense hotspot overload
- high mobility, roughly 43-79 km/h in the stress case
- URLLC traffic burst
- weak cell-edge SINR via edge placement, NLOS, and shadowing
- same-band interference stress through co-channel sub-6 small cells
- blockage-style stress through stronger NLOS and O2I loss
- stale or delayed controller observations through the stress template control profile

## Modeling Boundary

These are system-level O-RAN scenarios. They are not full 3GPP conformance
tests. The current simulator uses NR-inspired abstractions for numerology,
PRBs, OFDM symbols, HARQ, MCS/TBS, mobility, and inter-cell interference.
That is appropriate for orchestration experiments, but the paper should state
that exact L1 waveform processing, full RRC procedures, and full standard MAC
procedures are outside the evaluation scope.
