# Paper Evaluation Artifacts

This project now separates full simulator rollouts from lightweight mechanism
benchmarks.

## Full simulator rollout

- Runner: `experiments/run_cross_scenario_generalization.py`
- Config: `experiments/configs/oran_paper_eval.yaml`
- Output root: `outputs/paper_eval/oran_paper_eval_v1`
- Purpose: Experiment 1, cross-scenario multi-xApp orchestration with the O-RAN
  scenario matrix and DeepSeek-backed Non-RT LLM path.

The current configuration uses CPU `custom_phy` for stable long runs.  It keeps
the LLM outside the Near-RT control path; Near-RT actions remain deterministic
and bounded.

## Sandbox/mechanism benchmarks

- Runner: `experiments/run_paper_sandbox_eval.py`
- Output root: `outputs/paper_eval/paper_sandbox_eval_v1`
- Artifact index:
  `outputs/paper_eval/paper_sandbox_eval_v1/paper_evaluation_artifact_index.csv`

The sandbox runner produces deterministic data and figures for:

- Experiment 2: failed-case memory usefulness
- Experiment 3: event-driven contract-constrained Near-RT adaptation
- Experiment 5: guidance-constrained Near-RT tuning
- Experiment 6: LLM-safe pipeline evaluation
- Experiment 7: latency and scalability
- Experiment 8: sensitivity analysis
- Experiment 9: end-to-end closed-loop learning

Experiment 3 is implemented in:

- Runner: `experiments/run_experiment3_event_adaptation.py`
- Default output root: `outputs/paper_eval/experiment3_event_adaptation_neural_v2`
- Optimized 10-seed paper output:
  `outputs/paper_eval/experiment3_event_adaptation_quad035_paper10`
- Integrated sandbox output: `outputs/paper_eval/paper_sandbox_eval_v1/exp3_event_driven_contract_adaptation`
- Detailed protocol: `docs/experiment3_event_driven_contract_adaptation.md`
- DRL implementation: offline-trained PyTorch neural actor/critic policies on
  CUDA when available, with NumPy fallback and inference-only testing for DDPG,
  TD3, and SAC rows.

Existing legacy outputs from earlier mechanism benchmarks are also plotted when
present:

- Feasibility repair: `outputs/candidate_repair/candidate_repair_eval_v1`
- Experiment 4: `outputs/conflict_taxonomy/conflict_taxonomy_eval_v1`

## Literature manifest

The baseline/platform literature manifest is stored at:

- `docs/references/literature_baselines.csv`

Downloaded open PDFs are stored under:

- `docs/references/pdfs`

These PDFs are used only as primary-source references for baseline naming and
experimental positioning.  The implemented baselines are local approximations
of the published methods, not reimplementations of their full codebases.
