# Intent-Driven O-RAN Architecture

This repository implements a prototype of the paper-style intent-driven O-RAN orchestration loop on top of the existing simulator, controller manager, rApp/xApp registry, policy store, and LLM adapter.

## Phase 1 Scope

Phase 1 delivers the minimum thesis-aligned closed loop:

1. Slow-timescale active intent management
2. Validated intent translation
3. Non-RT orchestration plan and RL guidance contracts
4. Near-RT safe RL consumption path
5. Logging and export hooks for smoke-run evidence

The following are intentionally **not** implemented in Phase 1:

- richer reflection-memory lifecycle management beyond the current local JSON store
- fusion/arbitration module refactor
- baseline/ablation runner family
- global `src/oran_sim/` migration

## Top-Level Runtime Modules

### Environment and simulator

- `core/simulator.py`
- `core/state_updater.py`
- `core/metrics.py`
- `traffic/`
- `scenarios/`
- `state/`
- `ran/`
- `phy/`

These modules provide the slot-level environment, KPI aggregation, PHY execution, mobility, traffic, and output export.

### Non-RT orchestration

- `ric/non_rt/active_intent_manager.py`
- `ric/non_rt/intent_goal_translator_rapp.py`
- `ric/non_rt/intent_translation_validator.py`
- `ric/non_rt/intent_translation_projector.py`
- `ric/non_rt/intent_translation_fallbacks.py`
- `ric/non_rt/xapp_selection_rapp.py`
- `ric/non_rt/reward_shaping_rapp.py`
- `ric/non_rt/policy_store.py`

These modules implement the slow-to-medium timescale orchestration layer.

### Near-RT control

- `ric/near_rt/goal_parameter_tuning_xapp.py`
- `ric/near_rt/xapp_variant_catalog.py`
- existing heuristic/parameter xApps under `ric/near_rt/`

These modules consume the current Non-RT policy snapshot and generate executable RAN control actions.

### Shared contracts

- `models/intent_orchestration.py`
- `models/policy.py`
- `models/action.py`

These files define the paper-facing contracts and the runtime policy snapshot used by the controller manager and xApps.

## Paper Module to Code Mapping

### Active intent `\bar{I}_k`

- Contract: `ActiveIntentRecord`
- Lifecycle manager: `ric/non_rt/active_intent_manager.py`
- Persisted in: `PolicySnapshot.active_intent`
- Audit fields:
  - `last_update_epoch`
  - `last_change_reason`
  - `active_intent_id`
  - `intent_changed_this_epoch`

### KPI template `K_k`

- Canonical contract: `ValidatedIntentTemplate`
- KPI entry contract: `ValidatedKpiTemplateEntry`
- Translation pipeline:
  - generator: `ric/non_rt/intent_goal_translator_rapp.py`
  - validator: `ric/non_rt/intent_translation_validator.py`
  - projector: `ric/non_rt/intent_translation_projector.py`
  - fallback: `ric/non_rt/intent_translation_fallbacks.py`
- Persisted in: `PolicySnapshot.validated_intent_template`

### Orchestration plan `g_k = (x, B, eta, chi)`

- Unified view: `OrchestrationPlanView`
- Current source plan: `XAppSelectionPlan`
- Current field mapping:
  - `x`: selected xApps
  - `B`: hard safety envelope / parameter bounds
  - `eta`: high-level plan payload placeholder
  - `chi`: `ArbitrationProfileRef`, derived from the selected xApp portfolio and consumed by the Near-RT merge layer

### RL guidance `U_k^RL`

- Contract: `RlGuidanceContract`
- Producer: `ric/non_rt/reward_shaping_rapp.py`
- Fixed fields:
  - `guided_action_region`
  - `reward_coefficients`
  - `parameter_priority_vector`
  - `selected_safety_rule_set`
  - `source_epoch`
  - `derived_from_intent_id`
  - `fallback_used`

Current Phase 1 consumption status:

- `guided_action_region`: consumed by the Near-RT tuning xApp
- `reward_coefficients`: consumed by the Near-RT tuning xApp
- `selected_safety_rule_set`: consumed by the Near-RT tuning xApp
- `parameter_priority_vector`: normalized and consumed by the Near-RT tuning xApp as a priority-weight signal during joint-action construction
- `chi`: consumed by the Near-RT tuning xApp to switch among `weighted_merge`, `priority_override`, and `guardrail_first`

### Safety shield

- Contract: `SafetyConstraintSet`
- Serializable rules: `SafetyRuleReference`
- Current rule library IDs:
  - `rule:hard_safety_envelope`
  - `rule:forbidden_combinations`
  - `rule:exploration_health_gate`
  - `rule:degraded_network_freeze`
  - `rule:fallback_template_guard`
- Consumed in: `ric/near_rt/goal_parameter_tuning_xapp.py`

### Memory state

- Contracts:
  - `OrchestrationMemoryState`
  - `OrchestrationMemoryCase`
- Producer: `ric/non_rt/xapp_selection_rapp.py`
- Persisted in: `PolicySnapshot.memory_state`

Current behavior:

- success cases can be stored in a local JSON-backed memory when the active plan reaches a sufficiently low goal gap
- failure cases can be stored in the same local memory when reselection is triggered
- retrieval returns the top recent similar success and failure cases and feeds them into reselection/reflection prompts
- each reselection builds an explicit `SelectionQueryKey` (`q_k^{sel}`) and records a small scored candidate set through `CandidatePlanScore`
- the selected candidate now exports intent-alignment, memory-support, conflict-cost, switching-cost, and stability-risk surrogates for later analysis
- retrieval now also exports explicit `TopK` traces with similarity score, recency score, retrieval score, and source slot information
- `SelectionQueryKey` now carries a sparse encoded feature map (`Enc(q_k^{sel})`) and each memory case stores a matching sparse encoding for retrieval
- a lightweight recency-aging term is used during retrieval, while memory persistence still remains local JSON
- stored memory cases now maintain retrieval-hit counters, last-retrieved slot metadata, and a retention score used for lightweight pruning
- candidate selection now exports a `J_k^{sel}`-style surrogate objective with explicit `predicted_intent_loss`, conflict, switching, and stability terms instead of only an alignment-style score

Current limitation:

- persistence is currently a lightweight local JSON store, not a richer lifecycle-managed memory backend
- retrieval is heuristic similarity retrieval, not a learned retriever

## Time-Scale Separation

### Intent update epoch

- Managed by `ActiveIntentManager`
- Event-triggered
- Piecewise constant across multiple Non-RT epochs

### Non-RT orchestration epoch

- Triggered by `ControllerManager` using `rapp_interval_slots`
- Uses the current active intent together with fresh aggregated observation
- Re-runs translation each epoch even if the active intent is unchanged

### Near-RT slot loop

- Triggered by `ControllerManager` using `xapp_interval_slots`
- Near-RT RL xApp consumes the latest policy snapshot every slot
- Uses safety gating and safe-baseline fallback if exploration becomes unsafe

## Current Contract Flow

1. `ActiveIntentManager` exposes the current active intent
2. `intent_goal_translator_rapp` refreshes `StructuredIntentGoal` and `ValidatedIntentTemplate`
3. `xapp_selection_rapp` builds `q_k^{sel}`, retrieves recent success/failure cases, scores a small candidate portfolio set, and produces `XAppSelectionPlan` plus an explicit `chi` profile in `OrchestrationPlanView`
4. `reward_shaping_rapp` instantiates:
   - `RewardSpec` for backward compatibility
   - `RlGuidanceContract` as the thesis-aligned Near-RT contract
5. `PolicyStore` aggregates all updates into one `PolicySnapshot`
6. `goal_parameter_tuning_xapp` prefers `RlGuidanceContract`, then falls back to legacy `RewardSpec`

## Current Limitations

The following thesis concepts are only partially implemented or remain placeholders:

- richer reflection-memory lifecycle management beyond the current local JSON store
- explicit candidate-plan reranking using learned scorers
- standalone fusion/arbitration engine beyond current controller merge rules
- richer baseline and ablation runner suite
- full project reorganization under `src/oran_sim/`
