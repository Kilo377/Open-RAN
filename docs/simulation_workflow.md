# Simulation Workflow

This document describes the current Phase 1 execution workflow from operator intent to near-RT slot control and exported KPI evidence.

## Three Time Scales

### 1. Intent update epoch

The operator intent is modeled as a slow, event-triggered, piecewise-constant process.

- Manager: `ric/non_rt/active_intent_manager.py`
- Runtime object: `ActiveIntentRecord`
- Audit fields:
  - `last_update_epoch`
  - `last_change_reason`
  - `active_intent_id`
  - `intent_changed_this_epoch`

If no new event arrives, the active intent remains unchanged across multiple Non-RT epochs.

### 2. Non-RT orchestration epoch

The Non-RT orchestration layer runs every `rapp_interval_slots`.

- Orchestrator: `ric/controller_manager.py`
- State carrier: `ric/non_rt/policy_store.py`

At each Non-RT epoch:

1. the current active intent is injected into the policy snapshot
2. translation is refreshed using the new aggregated observation
3. xApp selection uses the translated goal
4. reward shaping instantiates an RL guidance contract

Important modeling choice:

- **active intent may stay unchanged**
- **translation may still refresh every Non-RT epoch**

This keeps `\bar{I}_k` slow while allowing `K_k` to adapt to new observations.

### 3. Near-RT slot control

The Near-RT control loop runs every `xapp_interval_slots`.

- Main consumer: `ric/near_rt/goal_parameter_tuning_xapp.py`

The xApp:

1. reads the latest `PolicySnapshot`
2. prefers `RlGuidanceContract`
3. falls back to `RewardSpec` if the explicit guidance contract is absent
4. applies safety constraints and health gating
5. executes epsilon-greedy or safe-baseline fallback

## End-to-End Data Flow

### Step A. Active intent lifecycle

Entry point:

- `experiments/run_intent_orchestration.py`

Runtime flow:

1. CLI input initializes `ActiveIntentManager`
2. `ControllerManager` queries the active intent at each Non-RT epoch
3. `PolicyStore` records:
   - `active_intent_id`
   - `intent_changed_this_epoch`
   - `translation_refreshed_this_epoch`

### Step B. Validated translation pipeline

Files:

- `ric/non_rt/intent_goal_translator_rapp.py`
- `ric/non_rt/intent_translation_validator.py`
- `ric/non_rt/intent_translation_projector.py`
- `ric/non_rt/intent_translation_fallbacks.py`

Pipeline:

1. raw LLM candidate is produced
2. validator checks schema/admissibility/direction/range
3. projector repairs and clips into the admissible set
4. fallback is used only when repair cannot produce a usable template
5. result is stored as:
   - `StructuredIntentGoal`
   - `ValidatedIntentTemplate`

Tracked evidence:

- raw candidate
- validator outcome
- projector output
- fallback output

### Step C. Orchestration plan

Files:

- `ric/non_rt/xapp_selection_rapp.py`
- `models/intent_orchestration.py`

Current Phase 1 representation:

- source plan: `XAppSelectionPlan`
- unified thesis view: `OrchestrationPlanView`

Current mapping:

- `x`: selected xApps
- `B`: hard safety envelope
- `eta`: placeholder high-level parameters
- `chi`: explicit arbitration profile reference consumed by the Near-RT merge layer

Current retrieval and memory behavior:

1. build an explicit `SelectionQueryKey` from the current goal, context signature, and previous portfolio
2. encode the retrieval query into a sparse feature map
3. retrieve recent similar success cases
4. retrieve recent similar failure cases
5. export `TopK` retrieval traces with semantic similarity, sparse-feature similarity, recency, and combined retrieval score
6. score a small candidate portfolio set with an explicit `J_k^{sel}`-style surrogate objective
7. use retrieved cases to enrich selection and reflection context
8. on good outcomes, write a success case into memory
9. update retrieval-hit counters, retention scores, and lightweight pruning metadata
10. on reselection-triggering outcomes, write a failure case and reflection into memory
11. if persistent memory is enabled, sync the current memory state to a local JSON store

### Step D. RL guidance instantiation

Files:

- `ric/non_rt/reward_shaping_rapp.py`
- `models/intent_orchestration.py`

Phase 1 rule:

- the reward form is fixed
- reward shaping only instantiates coefficients and priorities
- safety only activates predefined rule references

Produced object:

- `RlGuidanceContract`

Fields:

- `guided_action_region`
- `reward_coefficients`
- `parameter_priority_vector`
- `selected_safety_rule_set`
- `source_epoch`
- `derived_from_intent_id`
- `fallback_used`

Current Phase 1 note:

- `guided_action_region`, `reward_coefficients`, and `selected_safety_rule_set` are consumed by `goal_parameter_tuning_xapp`
- `parameter_priority_vector` is normalized at the Non-RT layer and consumed by the Near-RT xApp as a priority-weight signal during joint-action construction
- `chi` is consumed at the Near-RT layer to switch among `weighted_merge`, `priority_override`, and `guardrail_first`

### Step E. Near-RT safe RL consumption

File:

- `ric/near_rt/goal_parameter_tuning_xapp.py`

Consumption order:

1. read `RlGuidanceContract`
2. if missing, read legacy `RewardSpec`
3. build head library from the guided action region
4. merge overlapping head outputs using `chi`
5. intersect with safety envelope and forbidden combinations
6. check exploration health gate
7. choose:
   - `epsilon_greedy`
   - or `safe_baseline_fallback`

### Step F. KPI aggregation and export

Files:

- `core/state_updater.py`
- `core/simulator.py`
- `core/result_writer.py`

Exported evidence now includes:

- slot-level KPI trajectory
- control trace
- near-RT guidance mode and selection mode
- safe-baseline fallback indicator
- final policy snapshot summary

## Logging Surface

The current orchestration logs emit:

- `[active_intent]`
- `[rapp_goal]`
- `[rapp_goal_validation]`
- `[rapp_selection]`
- `[rapp_reflection]`
- `[rapp_reward]`
- `[rapp_guidance]`
- concise slot metrics

These are intended to support smoke-run inspection and paper screenshots.

## Current Phase 1 Gaps

Still not implemented in this workflow:

- richer memory lifecycle management beyond the current local JSON persistence
- learned retrieval or embedding retrieval
- stronger reflection-memory maintenance policies such as aging and de-duplication
- learned orchestration candidate reranking beyond the current heuristic surrogate scorer
- standalone fusion/arbitration refactor
- baseline/ablation orchestration runners
