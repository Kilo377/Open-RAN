# Experiment 1 Baseline Alignment

## Positioning

Experiment 1 is repositioned as **Cross-Scenario Multi-xApp Orchestration: Comparison with Existing Orchestration and Intent-Driven Control Methods**. It evaluates whether reflective outcome memory improves multi-xApp orchestration under unseen scenario shifts, whether the proposed method is stronger than literature-inspired orchestration/control baselines, and whether failed-case memory reduces repeated failures.

## Main Methods

| Method | Type | Implementation in this repo | Memory use |
| --- | --- | --- | --- |
| `fixed_plan` | Simple baseline | Always selects load steering plus QoS scheduling (`traffic_steering_xapp`, `latency_scheduler_xapp`) and keeps the portfolio fixed. | None |
| `no_memory_planner` | Internal ablation | Uses current intent, scenario state, xApp registry, feasibility projection, and conflict metadata. | None |
| `success_only_memory` | Internal ablation | Uses the reflective selector but only stores/retrieves successful analogues. | Success cases only |
| `orchestran_like_resource_aware` | Literature-aligned | Builds a network-intelligence catalog from the intent, solves near-RT/edge/non-RT execution placement, tracks lifecycle/migration state, and scores intent loss, resource pressure, timing deadline violation, lifecycle cost, and switching cost. | None |
| `intent_hrl_like` | Literature-aligned | Uses a bi-level HRL abstraction: a meta-controller maps intent plus quantized KPI state to target classes, and a low-level policy selects xApp actions. Warm-up epochs update a persistent policy table and convergence curve; transfer epochs load that frozen policy. | None |
| `oreo_like_service_graph` | Literature-aligned | Builds a multi-layer service graph from service demands, enumerates service configurations, runs admission/configuration scoring, rewards xApp sharing, and penalizes resource, dependency, and interface mismatch costs. | None |
| `full_method` | Proposed | Uses schema-bound planning, deterministic repair, typed conflict handling, success memory, failed-case reflection, typed guidance, and bounded Near-RT tuning. | Success and failure cases |

## Service-Graph Literature Baseline

`oreo_like_service_graph` is included as the seventh Experiment 1 method. Its objective remains different from the proposed method: it is deployment/service-coverage oriented and does not use reflective success/failure memory.

## Literature Mapping

- OrchestRAN baseline: inspired by OrchestRAN's Open RAN network-intelligence orchestration by modeling function selection, execution-location placement, timing-deadline feasibility, resource pressure, and lifecycle/switching cost.
- Intent-HRL baseline: inspired by hierarchical intent-driven O-RAN control by separating a meta-controller target-class decision from a low-level controller action objective over xApp composition and state-conditioned reward, then evaluating transfer with a frozen policy learned during warm-up.
- OREO baseline: inspired by service-graph xApp orchestration by modeling service admission, semantic xApp sharing groups, dependency mismatch, interface mismatch, resource-aware deployment cost, and an explicit configuration/admission solver.

## Faithful-Abstraction Policy

The three literature baselines are implemented as faithful abstractions rather than full system reproductions. This is intentional: Experiment 1 compares cross-scenario multi-xApp orchestration behavior under the same simulator, xApp registry, intent suite, and KPI logger. Therefore, each literature-inspired method keeps the mechanism that is most relevant to the paper's contribution while omitting implementation details that are orthogonal to this benchmark.

| Literature baseline | Mechanism preserved | Controlled difference in this repo |
| --- | --- | --- |
| OrchestRAN | Network-intelligence selection, execution-location placement (`near_rt`, `edge`, `non_rt`), deadline violation, lifecycle/migration cost, resource pressure, and switching cost. The placement/lifecycle state is persisted per intent/seed during warm-up. | It runs over this paper's xApp registry and simulator instead of importing the original OrchestRAN runtime platform. It does not use reflective memory or failed-case avoid rules. |
| Intent-HRL | Bi-level intent control: meta-controller target classes from intent/KPI state, low-level xApp composition action, reward terms for target coverage, state-action suitability, conflict, resource, stability, and switching, plus a warm-up training history and frozen train/test transfer policy. | It uses a lightweight one-hidden-layer MLP Q-policy with a tabular stabilizer over the simulator's state/action abstraction instead of importing the original training stack, so it is reported as an inspired baseline rather than an exact reproduction. |
| OREO | Multi-layer service-graph abstraction: service demands, service configuration options, xApp semantic sharing groups, service admission, dependency/interface mismatch, and resource-aware deployment cost. | It is evaluated as an xApp service-graph optimizer inside the same simulator, because OREO's native deployment/admission platform is not directly importable into this RAN simulator. |

Use "OrchestRAN", "Intent-HRL", and "OREO" as concise method labels in tables and figures, with the text stating that these are simulator-aligned inspired baselines rather than exact reimplementations of the original systems.

## Primary Configuration

The 16-intent Experiment 1 configuration is:

`experiments/configs/oran_paper_eval_multi_intent.yaml`

It currently runs seven primary methods:

1. `fixed_plan`
2. `no_memory_planner`
3. `success_only_memory`
4. `orchestran_like_resource_aware`
5. `intent_hrl_like`
6. `oreo_like_service_graph`
7. `full_method`

With the current manifest and three seeds, the planned matrix is 147 runs per intent and 2352 total runs across 16 intents.
