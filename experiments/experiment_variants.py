"""Experiment variant definitions for minimal paper-ready benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    description: str
    validator_enabled: bool = True
    rl_guidance_enabled: bool = True
    safety_shield_enabled: bool = True
    static_xapp_selection_enabled: bool = False
    reflection_enabled: bool = True
    memory_mode: str = "full"
    selection_mode: str = "reflective"
    repair_mode: str = "deterministic"
    conflict_mode: str = "typed_taxonomy"
    llm_pipeline_mode: str = "schema_bound"
    near_rt_mode: str = "projected_tuning"
    extra_ric_overrides: dict[str, Any] = field(default_factory=dict)
    extra_sim_overrides: dict[str, Any] = field(default_factory=dict)

    def ric_overrides(self) -> dict[str, Any]:
        overrides = {
            "experiment_variant_name": str(self.name),
            "validator_enabled": bool(self.validator_enabled),
            "rl_guidance_enabled": bool(self.rl_guidance_enabled),
            "safety_shield_enabled": bool(self.safety_shield_enabled),
            "static_xapp_selection_enabled": bool(self.static_xapp_selection_enabled),
            "reflection_enabled": bool(self.reflection_enabled),
            "memory_mode": str(self.memory_mode),
            "selection_mode": str(self.selection_mode),
            "repair_mode": str(self.repair_mode),
            "conflict_mode": str(self.conflict_mode),
            "llm_pipeline_mode": str(self.llm_pipeline_mode),
            "near_rt_mode": str(self.near_rt_mode),
        }
        overrides.update(dict(self.extra_ric_overrides))
        return overrides

    def sim_overrides(self) -> dict[str, Any]:
        return dict(self.extra_sim_overrides)


def build_minimal_paper_variants() -> dict[str, ExperimentVariant]:
    variants = {
        "full_method": ExperimentVariant(
            name="full_method",
            description="Full Phase 1 intent-driven orchestration pipeline.",
            extra_ric_overrides={
                "max_selected_xapps": 5,
                "non_rt_reselection_enabled": True,
                "non_rt_reselection_fallback_threshold": 1,
                "non_rt_reselection_min_gap_slots": 8,
            },
        ),
        "fixed_plan": ExperimentVariant(
            name="fixed_plan",
            description=(
                "Static baseline that always starts from the same load-steering plus QoS-scheduler portfolio."
            ),
            static_xapp_selection_enabled=True,
            reflection_enabled=False,
            memory_mode="none",
            selection_mode="fixed_plan",
            extra_ric_overrides={
                "max_selected_xapps": 2,
            },
        ),
        "no_memory_planner": ExperimentVariant(
            name="no_memory_planner",
            description=(
                "Use the current intent, scenario descriptor, registry, and conflict metadata, but disable "
                "reflective success/failure memory retrieval and storage."
            ),
            reflection_enabled=False,
            memory_mode="none",
            selection_mode="no_memory_planner",
        ),
        "orchestran_like_resource_aware": ExperimentVariant(
            name="orchestran_like_resource_aware",
            description=(
                "OrchestRAN-inspired network-intelligence orchestration baseline: select xApp/algorithm portfolios "
                "under resource and execution-time costs without outcome memory or failed-case repair."
            ),
            reflection_enabled=False,
            memory_mode="none",
            selection_mode="resource_aware",
            extra_ric_overrides={
                "selection_objective_reference": "OrchestRAN-like resource-aware orchestration",
            },
        ),
        "intent_hrl_like": ExperimentVariant(
            name="intent_hrl_like",
            description=(
                "Intent-HRL-inspired two-level baseline: a meta-controller maps the intent to KPI target classes and "
                "a low-level controller selects xApp/parameter candidates with a reward-style objective."
            ),
            reflection_enabled=False,
            memory_mode="none",
            selection_mode="intent_hrl",
            extra_ric_overrides={
                "selection_objective_reference": "Intent-HRL-like hierarchical RL orchestration",
            },
        ),
        "oreo_like_service_graph": ExperimentVariant(
            name="oreo_like_service_graph",
            description=(
                "OREO-inspired service-graph baseline optimizing service coverage, xApp sharing, and resource cost "
                "without reflective memory or execution-outcome repair."
            ),
            reflection_enabled=False,
            memory_mode="none",
            selection_mode="service_graph",
            extra_ric_overrides={
                "selection_objective_reference": "OREO-like service-graph orchestration",
            },
        ),
        "static_xapp_selection": ExperimentVariant(
            name="static_xapp_selection",
            description=(
                "Freeze the selected xApp set after the initial orchestration epoch while keeping translation, "
                "guidance refresh, and Near-RT tuning active."
            ),
            static_xapp_selection_enabled=True,
            selection_mode="static",
        ),
        "no_rl_guidance": ExperimentVariant(
            name="no_rl_guidance",
            description=(
                "Disable the explicit RL guidance contract and let Near-RT tuning consume the legacy RewardSpec path."
            ),
            rl_guidance_enabled=False,
            near_rt_mode="legacy_reward_spec",
        ),
        "no_safety_shield": ExperimentVariant(
            name="no_safety_shield",
            description=(
                "Disable the Near-RT hard envelope, forbidden-combination filter, health gate, and safe-baseline fallback "
                "while retaining basic action legality from the bounded parameter library."
            ),
            safety_shield_enabled=False,
            near_rt_mode="unshielded_tuning",
        ),
        "no_validator": ExperimentVariant(
            name="no_validator",
            description=(
                "Disable the formal validator/projector/fallback chain while retaining minimal JSON parsing and required-field checks."
            ),
            validator_enabled=False,
            llm_pipeline_mode="minimal_parse",
            repair_mode="disabled",
        ),
        "no_reflection": ExperimentVariant(
            name="no_reflection",
            description="Disable Reflexion-style reselection context while keeping xApp reselection itself enabled.",
            reflection_enabled=False,
        ),
        "no_memory": ExperimentVariant(
            name="no_memory",
            description="Disable reflective success/failure memory retrieval and storage.",
            memory_mode="none",
        ),
        "success_only_memory": ExperimentVariant(
            name="success_only_memory",
            description="Use retrieved and stored successful analogues but suppress failed-case memory and reflection.",
            reflection_enabled=False,
            memory_mode="success_only",
        ),
        "no_transfer_guard": ExperimentVariant(
            name="no_transfer_guard",
            description=(
                "Keep reflective success/failure memory but disable post-selection transfer guard and coverage repair."
            ),
            extra_ric_overrides={
                "reflective_transfer_guard_enabled": False,
            },
        ),
        "shuffled_memory": ExperimentVariant(
            name="shuffled_memory",
            description=(
                "Keep full memory but shuffle retrieved cases deterministically instead of ranking by relevance."
            ),
            extra_ric_overrides={
                "memory_retrieval_mode": "shuffled",
            },
        ),
        "no_age_decay": ExperimentVariant(
            name="no_age_decay",
            description="Keep full memory retrieval and retention but remove slot-age decay from memory scores.",
            extra_ric_overrides={
                "memory_age_decay_enabled": False,
            },
        ),
        "stale_memory_no_decay": ExperimentVariant(
            name="stale_memory_no_decay",
            description="Keep full memory retrieval and retention but remove slot-age decay from memory scores.",
            extra_ric_overrides={
                "memory_age_decay_enabled": False,
            },
        ),
        "failed_only_memory": ExperimentVariant(
            name="failed_only_memory",
            description="Use failed-case memory and avoid traces without success analogue reuse.",
            memory_mode="failed_only",
        ),
    }
    return variants
