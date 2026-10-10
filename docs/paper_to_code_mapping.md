# Paper-to-Code Mapping

This document provides the explicit thesis symbol to code mapping for the current Phase 1 prototype.

## Symbol Mapping Table

| Paper object | Meaning | Current code object | Key files |
|---|---|---|---|
| `\bar{I}_k` | active intent at Non-RT epoch `k` | `ActiveIntentRecord` | `models/intent_orchestration.py`, `ric/non_rt/active_intent_manager.py`, `models/policy.py` |
| `K_k` | validated KPI template at epoch `k` | `ValidatedIntentTemplate` | `models/intent_orchestration.py`, `ric/non_rt/intent_goal_translator_rapp.py`, `ric/non_rt/intent_translation_validator.py`, `ric/non_rt/intent_translation_projector.py`, `ric/non_rt/intent_translation_fallbacks.py` |
| `g_k = (x, B, eta, chi)` | orchestration plan | `OrchestrationPlanView` derived from `XAppSelectionPlan` | `models/intent_orchestration.py`, `ric/non_rt/xapp_selection_rapp.py`, `ric/non_rt/policy_store.py` |
| `U_k^RL` | Near-RT RL guidance | `RlGuidanceContract` | `models/intent_orchestration.py`, `ric/non_rt/reward_shaping_rapp.py`, `ric/non_rt/policy_store.py` |
| safety shield | legality and safety layer | `SafetyConstraintSet` + `SafetyRuleReference` | `models/intent_orchestration.py`, `ric/non_rt/reward_shaping_rapp.py`, `ric/near_rt/goal_parameter_tuning_xapp.py` |
| memory state | success/failure case memory plus retrieval trace | `OrchestrationMemoryState` + `OrchestrationMemoryCase` | `models/intent_orchestration.py`, `models/policy.py`, `ric/non_rt/xapp_selection_rapp.py` |
| method ablation switches | paper-facing experiment modes | `memory_mode`, `selection_mode`, `repair_mode`, `conflict_mode`, `llm_pipeline_mode`, `near_rt_mode` | `experiments/experiment_variants.py`, `models/scenario.py`, `core/simulator.py`, `ric/controller_manager.py` |

## Object-by-Object Notes

### `\bar{I}_k`: Active intent

Implemented:

- event-triggered update
- piecewise-constant hold across multiple Non-RT epochs
- epoch audit metadata

Current runtime fields:

- `active_intent_id`
- `intent_text`
- `effective_from_epoch`
- `last_update_epoch`
- `last_change_reason`

Important note:

- the active intent lifecycle is slow
- but translation is still refreshed every Non-RT epoch

### `K_k`: Validated KPI template

Implemented:

- raw candidate tracking
- structured validator report
- projector repair stage
- explicit fallback stage

Tracked metadata:

- `raw_llm_candidate`
- `validator_outcome`
- `projector_output`
- `fallback_output`
- `validation_report`
- `projection_report`
- `fallback_report`

Phase 1 constraint:

- projector only repairs and clips
- it does not redesign the goal

### `g_k = (x, B, eta, chi)`: Orchestration plan

Current practical source:

- `XAppSelectionPlan`

Current unified thesis view:

- `OrchestrationPlanView`

Status by component:

- `x`: implemented
- `B`: implemented via hard safety envelope
- `eta`: placeholder carrier for high-level orchestration parameters
- `chi`: explicit `ArbitrationProfileRef`, derived from the selected portfolio and consumed by the Near-RT merge logic for `weighted_merge`, `priority_override`, and `guardrail_first`

### `U_k^RL`: RL guidance contract

Implemented as a fixed guidance schema.

Current fields:

- `guided_action_region`
- `reward_coefficients`
- `parameter_priority_vector`
- `selected_safety_rule_set`
- `source_epoch`
- `derived_from_intent_id`
- `fallback_used`

Phase 1 constraint:

- reward form is fixed
- only coefficients and priorities are instantiated
- safety rules come only from a predefined rule library
- current Near-RT consumer directly uses `guided_action_region`, `reward_coefficients`, `selected_safety_rule_set`, and a normalized `parameter_priority_vector`

### Safety shield

Implemented as serializable references plus concrete bounded payload.

Current rule IDs:

- `rule:hard_safety_envelope`
- `rule:forbidden_combinations`
- `rule:exploration_health_gate`
- `rule:degraded_network_freeze`
- `rule:fallback_template_guard`

Near-RT consumption currently uses:

- hard envelope
- forbidden combinations
- exploration health gate
- safe-baseline fallback

### Memory state

Current status:

- JSON-backed success case memory is implemented
- JSON-backed failure case memory is implemented
- `memory_mode` is implemented for `full`, `none`, `success_only`, `failed_only`, and diagnostic-style retrieval behavior
- an explicit retrieval key `SelectionQueryKey` is implemented for reflective selection
- `SelectionQueryKey` now includes an explicit encoded feature map plus an embedding vector that act as the current `Enc(q_k^{sel})` implementation
- retrieval over recent success/failure cases is implemented inside `xapp_selection_rapp`
- retrieval exports structured `RetrievedMemoryCaseTrace` records, including similarity, recency, and final retrieval score
- retrieval also includes embedding similarity, retriever-backend metadata, and retrieval confidence in the exported trace
- stored memory cases now track retrieval hits, last retrieval slot, and a lightweight retention score for pruning
- stored memory cases now also retain `intent_loss_normalized`, making the persisted memory signal closer to the paper's realized-loss view
- failure-triggered reflection is implemented and stored alongside failure cases
- a small candidate-plan search and surrogate scoring stage is implemented before final reselection
- the surrogate score now exports `predicted_intent_loss`, intent-alignment, memory-support, failure-risk, conflict-cost, switching-cost, stability-risk, and resource-pressure components through `CandidatePlanScore`
- candidate scoring now uses a `minimize` objective style that is closer to the paper's `J_k^{sel}` formulation
- the default selector path now uses a memory-calibrated surrogate predictor that refines heuristic estimates with retrieved success/failure cases
- the retrieval stack now supports pluggable embedding backends, including weighted-hashed local backends loaded from state files
- the surrogate predictor stack now supports pluggable predictors, including a local linear-stateful predictor loaded from a JSON state file

Current limitation:

- persistence is local JSON persistence, not a full database-backed lifecycle store
- retrieval is currently a pluggable local embedding backend with deterministic hashed embeddings, not a learned retriever
- memory maintenance is lightweight scoring-and-pruning, not a richer confidence-managed lifecycle service

## Current Near-RT Consumption Path

1. `PolicyStore` exposes the latest `PolicySnapshot`
2. `goal_parameter_tuning_xapp` reads:
   - `structured_goal`
   - `xapp_selection_plan`
   - `rl_guidance` if present
   - otherwise `reward_spec`
3. the xApp builds the effective action region
4. the xApp evaluates health-gate conditions
5. it emits either:
   - normal epsilon-greedy online tuning
   - or `safe_baseline_fallback`

The current Near-RT runtime path also materializes typed runtime views for:

- `theta_{k,t}` via `ThetaRuntimeState`
- `omega_{k,t}` via `OmegaRuntimeState`
- the active mask via `NearRtRuntimeState`

## Implemented vs Placeholder

### Implemented in Phase 1

- active intent lifecycle
- validated translation pipeline
- xApp selection plan contract view
- fixed RL guidance contract
- near-RT contract-first consumption
- typed `theta / omega / m` runtime state materialization
- pluggable retriever and surrogate-predictor interfaces with local state-file loading
- slot/final-summary evidence export

### Placeholder or partial

- richer success/failure memory lifecycle management beyond the current local JSON store
- learned retrieval beyond the current pluggable hashed-embedding backend
- richer reflection-memory maintenance policies
- orchestration candidate reranking with learned scorers beyond the current memory-calibrated surrogate predictor
- standalone trained arbitration policy for `omega_{k,t}`
- dedicated baseline and ablation runners
- large-scale source tree migration
