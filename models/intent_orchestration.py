"""Intent-driven orchestration data models.

These models support a multi-stage Non-RT -> Near-RT orchestration flow:
1. User intent + current RAN state -> structured goal/KPI targets
2. Structured goal -> xApp selection plan
3. Goal + selected xApps -> reward/tuning specification
4. Near-RT tuning xApp -> control-surface fine tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _objective_to_semantic_type(objective: str) -> str:
    normalized = str(objective or "").strip().lower()
    if normalized in {"maximize", "max"}:
        return "max"
    if normalized in {"minimize", "min"}:
        return "min"
    return "target"


@dataclass
class GoalKpiTarget:
    """One KPI objective derived from a user intent."""

    kpi_name: str
    objective: str = "maximize"
    target_value: float | None = None
    tolerance: float | None = None
    weight: float = 1.0
    unit: str = ""
    priority: int = 0
    hard: bool = False


@dataclass
class KpiNormalizationMetadata:
    """Normalization metadata attached to one validated KPI template entry."""

    unit: str = ""
    normalization_coefficient: float = 1.0
    reference_value: float | None = None
    lower_clip: float | None = None
    upper_clip: float | None = None


@dataclass
class ValidatedKpiTemplateEntry:
    """One validated KPI entry in the canonical template K_k."""

    kpi_name: str
    semantic_type: str = "max"
    target_value: float | None = None
    threshold_value: float | None = None
    weight: float = 1.0
    tolerance: float | None = None
    priority: int = 0
    hard: bool = False
    normalization: KpiNormalizationMetadata = field(default_factory=KpiNormalizationMetadata)


@dataclass
class TranslationQueryKey:
    """Structured retrieval key q_k^{ctx} for translation-context augmentation."""

    query_id: str = ""
    intent_id: str = ""
    encoding_version: str = "translation_context_v2_embedding"
    retriever_backend_name: str = "hashed_feature_embedding_v1"
    intent_tokens: list[str] = field(default_factory=list)
    context_signature: dict[str, object] = field(default_factory=dict)
    encoded_features: dict[str, float] = field(default_factory=dict)
    embedding_vector: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class TranslationContextTrace:
    """One retrieved context item used to augment validated intent translation."""

    source_id: str = ""
    source_type: str = ""
    retrieval_rank: int = 0
    relevance_score: float = 0.0
    encoding_version: str = "translation_context_v2_embedding"
    retriever_backend_name: str = "hashed_feature_embedding_v1"
    encoded_features: dict[str, float] = field(default_factory=dict)
    feature_similarity: float = 0.0
    embedding_vector: list[float] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    summary: str = ""
    payload: dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Validator outcome for the translated KPI template."""

    schema_ok: bool = True
    admissibility_ok: bool = True
    direction_ok: bool = True
    range_ok: bool = True
    validator_name: str = "legacy_passthrough"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_fields: list[str] = field(default_factory=list)
    duplicate_kpis: list[str] = field(default_factory=list)
    repaired_fields: list[str] = field(default_factory=list)
    rejected_kpis: list[str] = field(default_factory=list)


@dataclass
class ProjectionReport:
    """Projector/repair outcome for mapping to an admissible template set."""

    projected: bool = False
    projector_name: str = "legacy_passthrough"
    projection_target: str = ""
    repaired_fields: list[str] = field(default_factory=list)
    dropped_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class FallbackReport:
    """Fallback metadata for validated template generation."""

    fallback_used: bool = False
    fallback_strategy: str = ""
    fallback_reason: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class StructuredIntentGoal:
    """Structured goal translated from raw user intent."""

    goal_id: str = ""
    intent_id: str = ""
    intent_text: str = ""
    summary: str = ""
    horizon_slots: int = 10
    risk_posture: str = "balanced"
    target_entities: list[str] = field(default_factory=lambda: ["cluster"])
    kpi_targets: list[GoalKpiTarget] = field(default_factory=list)
    constraints: dict[str, object] = field(default_factory=dict)
    generated_by: str = "heuristic"


@dataclass
class ValidatedIntentTemplate:
    """Canonical KPI template K_k with validation/projection/fallback metadata."""

    template_id: str = ""
    goal_id: str = ""
    intent_id: str = ""
    intent_text: str = ""
    summary: str = ""
    horizon_slots: int = 10
    risk_posture: str = "balanced"
    target_entities: list[str] = field(default_factory=lambda: ["cluster"])
    kpi_template: list[ValidatedKpiTemplateEntry] = field(default_factory=list)
    constraints: dict[str, object] = field(default_factory=dict)
    translation_query_key: TranslationQueryKey | None = None
    retrieved_context_summary: dict[str, object] = field(default_factory=dict)
    retrieved_context_traces: list[TranslationContextTrace] = field(default_factory=list)
    raw_llm_candidate: dict[str, object] = field(default_factory=dict)
    validator_outcome: dict[str, object] = field(default_factory=dict)
    projector_output: dict[str, object] = field(default_factory=dict)
    fallback_output: dict[str, object] = field(default_factory=dict)
    validation_report: ValidationReport = field(default_factory=ValidationReport)
    projection_report: ProjectionReport = field(default_factory=ProjectionReport)
    fallback_report: FallbackReport = field(default_factory=FallbackReport)
    generated_by: str = "heuristic"


@dataclass
class ActiveIntentRecord:
    """Slow-timescale active intent record I_bar_k with audit metadata."""

    intent_id: str = ""
    intent_text: str = ""
    summary: str = ""
    effective_from_epoch: int | None = None
    last_update_epoch: int | None = None
    last_change_reason: str = ""
    source_name: str = "operator"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ReflectionNote:
    """One Reflexion-style self-critique record for xApp selection."""

    reflection_round: int = 0
    outcome_summary: str = ""
    self_critique: str = ""
    improvement_hypothesis: str = ""
    recommended_changes: list[str] = field(default_factory=list)
    generated_by: str = "heuristic"


@dataclass
class OrchestrationMemoryCase:
    """One success/failure orchestration case stored for retrieval."""

    case_id: str = ""
    memory_type: str = "success"
    source_slot_id: int | None = None
    last_retrieved_slot_id: int | None = None
    encoding_version: str = "structured_retrieval_v2_embedding"
    retriever_backend_name: str = "hashed_feature_embedding_v1"
    goal_id: str = ""
    intent_id: str = ""
    risk_posture: str = "balanced"
    kpi_names: list[str] = field(default_factory=list)
    selected_xapps: list[str] = field(default_factory=list)
    goal_gap_score: float | None = None
    intent_loss_normalized: float | None = None
    queue_bytes: int | None = None
    throughput_mbps: float | None = None
    avg_bler: float | None = None
    context_signature: dict[str, object] = field(default_factory=dict)
    encoded_features: dict[str, float] = field(default_factory=dict)
    embedding_vector: list[float] = field(default_factory=list)
    retrieval_hits: int = 0
    retention_score: float = 0.0
    summary: str = ""
    failure_reason: str = ""
    reflection: ReflectionNote | None = None
    generated_by: str = "heuristic"


@dataclass
class RetrievedMemoryCaseTrace:
    """Structured TopK retrieval trace for one success/failure memory case."""

    case_id: str = ""
    memory_type: str = "success"
    retrieval_rank: int = 0
    similarity_score: float = 0.0
    feature_similarity: float = 0.0
    retriever_backend_name: str = "hashed_feature_embedding_v1"
    embedding_vector: list[float] = field(default_factory=list)
    recency_score: float = 0.0
    retrieval_score: float = 0.0
    retrieval_confidence: float = 0.0
    source_slot_id: int | None = None
    memory_retention_score: float = 0.0
    selected_xapps: list[str] = field(default_factory=list)
    kpi_names: list[str] = field(default_factory=list)
    context_signature: dict[str, object] = field(default_factory=dict)
    failure_reason: str = ""
    summary: str = ""


@dataclass
class SelectionQueryKey:
    """Structured retrieval key q_k^{sel} for reflective xApp selection."""

    query_id: str = ""
    query_slot_id: int | None = None
    encoding_version: str = "structured_retrieval_v2_embedding"
    retriever_backend_name: str = "hashed_feature_embedding_v1"
    goal_id: str = ""
    intent_id: str = ""
    risk_posture: str = "balanced"
    kpi_names: list[str] = field(default_factory=list)
    context_signature: dict[str, object] = field(default_factory=dict)
    encoded_features: dict[str, float] = field(default_factory=dict)
    embedding_vector: list[float] = field(default_factory=list)
    previous_selected_xapps: list[str] = field(default_factory=list)
    previous_xapp_categories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class CandidatePlanScore:
    """Structured surrogate score entry for one candidate orchestration plan."""

    candidate_id: str = ""
    score_version: str = "j_sel_surrogate_v3_predictor_objective"
    score_direction: str = "minimize"
    predictor_name: str = "heuristic_structured_surrogate_predictor_v1"
    requested_categories: list[str] = field(default_factory=list)
    selected_xapps: list[str] = field(default_factory=list)
    predicted_intent_alignment: float = 0.0
    predicted_intent_loss: float = 0.0
    predicted_memory_support: float = 0.0
    predicted_failure_risk: float = 0.0
    predicted_conflict_cost: float = 0.0
    switching_cost: float = 0.0
    stability_risk: float = 0.0
    resource_pressure: float = 0.0
    eta_profile_name: str = ""
    chi_profile_name: str = ""
    parameter_bound_count: int = 0
    feasible: bool = True
    projection_applied: bool = False
    total_cpu_cost: float = 0.0
    total_mem_cost: float = 0.0
    total_score: float = 0.0
    selected: bool = False
    rationale: str = ""


@dataclass
class XAppCandidateScore:
    """One xApp candidate evaluation against the current goal."""

    xapp_name: str
    rationale: str = ""
    expected_kpi_impact: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    selected: bool = False


@dataclass
class A1GuidanceProfile:
    """Formalized eta_k profile carried from Non-RT to Near-RT."""

    profile_name: str = ""
    goal_focus: str = "balanced"
    reward_term_bias: dict[str, float] = field(default_factory=dict)
    field_priority_hints: dict[str, float] = field(default_factory=dict)
    safety_rule_bias: list[str] = field(default_factory=list)
    action_region_bias: str = "balanced"
    selected_categories: list[str] = field(default_factory=list)
    risk_posture: str = "balanced"
    defaulted: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class QoSConstraintSpec:
    """One outage-style QoS constraint used in admissibility checks."""

    metric_name: str
    semantic_type: str = "max"
    threshold_value: float | None = None
    outage_budget: float = 0.05
    scope: str = "cluster"
    target_slice_id: str = ""
    target_traffic_type: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class QoSProtectionProfile:
    """Typed QoS protection summary aligned with outage-style constraints."""

    profile_id: str = ""
    strictness: str = "balanced"
    constraints: list[QoSConstraintSpec] = field(default_factory=list)
    required_categories: list[str] = field(default_factory=list)
    generated_by: str = "heuristic"


@dataclass
class AmbientParameterDimension:
    """One active dimension in the global ambient parameter vector theta."""

    field_name: str
    target_xapp_name: str = ""
    tuning_mode: str = "guided_action_region"
    active: bool = True


@dataclass
class AmbientParameterSpace:
    """Explicit ambient parameter-space view (theta, m)."""

    space_id: str = ""
    dimension_version: str = "ambient_parameter_space_v1"
    dimensions: list[AmbientParameterDimension] = field(default_factory=list)
    active_parameter_mask: dict[str, int] = field(default_factory=dict)
    active_parameter_count: int = 0


@dataclass
class ArbitrationVariableSpec:
    """Explicit arbitration-variable specification for omega_{k,t}."""

    field_name: str
    arbitration_mode: str = "weighted_merge"
    merge_scope: str = "field"
    candidate_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ThetaRuntimeState:
    """Explicit theta_{k,t} runtime snapshot for Near-RT execution."""

    state_id: str = ""
    values: dict[str, object] = field(default_factory=dict)
    dimension_count: int = 0


@dataclass
class OmegaRuntimeFieldState:
    """One omega_{k,t} field-level arbitration state."""

    field_name: str = ""
    arbitration_mode: str = "weighted_merge"
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class OmegaRuntimeState:
    """Explicit omega_{k,t} runtime snapshot for Near-RT execution."""

    state_id: str = ""
    field_states: list[OmegaRuntimeFieldState] = field(default_factory=list)
    field_count: int = 0


@dataclass
class NearRtRuntimeState:
    """Typed runtime state carrying explicit theta, omega, and mask views."""

    runtime_id: str = ""
    active_parameter_mask: dict[str, int] = field(default_factory=dict)
    active_parameter_count: int = 0
    theta_state: ThetaRuntimeState | None = None
    omega_state: OmegaRuntimeState | None = None


@dataclass
class XAppSelectionPlan:
    """Selected xApp portfolio for the current structured goal."""

    plan_id: str = ""
    goal_id: str = ""
    selected_xapps: list[str] = field(default_factory=list)
    selection_query_key: SelectionQueryKey | None = None
    candidate_plan_projections: list["OrchestrationCandidatePlan"] = field(default_factory=list)
    candidate_plan_scores: list[CandidatePlanScore] = field(default_factory=list)
    candidate_scores: list[XAppCandidateScore] = field(default_factory=list)
    eta: A1GuidanceProfile | None = None
    chi: ArbitrationProfileRef | None = None
    orchestration_mode: str = "parallel"
    status: str = "pending"
    selection_locked: bool = False
    selection_generation: int = 0
    last_selection_slot: int | None = None
    last_reflection_slot: int | None = None
    last_goal_gap_score: float | None = None
    best_goal_gap_score: float | None = None
    consecutive_gap_violation_count: int = 0
    stagnation_count: int = 0
    failure_count: int = 0
    last_failure_reason: str = ""
    last_failure_gap_score: float | None = None
    hard_safety_envelope: list["RewardParameterRange"] = field(default_factory=list)
    forbidden_combinations: list["ForbiddenFieldCombination"] = field(default_factory=list)
    exploration_health_gate: "ExplorationHealthGate | None" = None
    qos_protection_profile: QoSProtectionProfile | None = None
    feasibility_summary: FeasibilityConstraintSummary | None = None
    reflection: ReflectionNote | None = None
    generated_by: str = "heuristic"


@dataclass
class RewardParameterRange:
    """One tunable field range produced for the Near-RT tuning xApp."""

    field_name: str
    target_xapp_name: str | None = None
    scope: str = "cluster"
    lower_bound: float | None = None
    upper_bound: float | None = None
    step: float | None = None
    discrete_values: list[object] = field(default_factory=list)
    blocked_discrete_values: list[object] = field(default_factory=list)
    max_delta: float | None = None
    tuning_mode: str = "heuristic_parameter"


@dataclass
class ForbiddenFieldCombination:
    """One directly disallowed field combination for online tuning."""

    field_values: dict[str, object] = field(default_factory=dict)
    scope: str = "any"
    reason: str = ""


@dataclass
class ExplorationHealthGate:
    """Health thresholds used to decide whether online exploration is allowed."""

    max_avg_bler: float | None = None
    max_total_queue_bytes: int | None = None
    min_total_throughput_mbps: float | None = None
    max_handover_pending_ue_count: int | None = None
    freeze_slots_on_violation: int = 4


@dataclass
class FeasibilityConstraintSummary:
    """Unified feasibility-region summary for one projected orchestration plan."""

    feasible: bool = True
    projected: bool = False
    max_selected_xapps: int = 0
    max_cpu_cost: float | None = None
    max_mem_cost: float | None = None
    total_cpu_cost: float = 0.0
    total_mem_cost: float = 0.0
    dependency_pairs: list[str] = field(default_factory=list)
    mutex_pairs: list[str] = field(default_factory=list)
    qos_constraint_count: int = 0
    qos_projection_applied: bool = False
    projection_notes: list[str] = field(default_factory=list)


@dataclass
class ArbitrationProfileRef:
    """Reference to the effective fusion/arbitration profile chi."""

    profile_name: str | None = None
    composition_mode: str = "parameter_tuning"
    arbitration_mode: str = "weighted_merge"
    merge_scope: str = "field"
    override_order: list[str] = field(default_factory=list)
    defaulted: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class OrchestrationCandidatePlan:
    """Projected full-form orchestration candidate (x, B, eta, chi)."""

    candidate_id: str = ""
    requested_xapps: list[str] = field(default_factory=list)
    x: list[str] = field(default_factory=list)
    B: list["RewardParameterRange"] = field(default_factory=list)
    eta: A1GuidanceProfile | None = None
    chi: ArbitrationProfileRef | None = None
    forbidden_combinations: list["ForbiddenFieldCombination"] = field(default_factory=list)
    exploration_health_gate: "ExplorationHealthGate | None" = None
    qos_protection_profile: QoSProtectionProfile | None = None
    feasibility_summary: FeasibilityConstraintSummary | None = None


@dataclass
class SafetyRuleReference:
    """Serializable reference to one selected safety rule."""

    rule_id: str
    config: dict[str, object] = field(default_factory=dict)
    description: str = ""


@dataclass
class SafetyConstraintSet:
    """Typed safety shield inputs carried from Non-RT to Near-RT."""

    rule_set_id: str = ""
    goal_id: str = ""
    selection_plan_id: str = ""
    selected_rule_refs: list[SafetyRuleReference] = field(default_factory=list)
    safety_envelope: list["RewardParameterRange"] = field(default_factory=list)
    forbidden_combinations: list["ForbiddenFieldCombination"] = field(default_factory=list)
    exploration_health_gate: "ExplorationHealthGate | None" = None
    generated_by: str = "heuristic"


@dataclass
class OrchestrationPlanView:
    """Unified view of g_k = (x, B, eta, chi) for orchestration analysis."""

    plan_id: str = ""
    goal_id: str = ""
    x: list[str] = field(default_factory=list)
    B: list["RewardParameterRange"] = field(default_factory=list)
    eta: A1GuidanceProfile | None = None
    chi: ArbitrationProfileRef | None = None
    forbidden_combinations: list["ForbiddenFieldCombination"] = field(default_factory=list)
    exploration_health_gate: "ExplorationHealthGate | None" = None
    qos_protection_profile: QoSProtectionProfile | None = None
    feasibility_summary: FeasibilityConstraintSummary | None = None
    generated_by: str = "heuristic"


@dataclass
class RewardSpec:
    """Reward/action specification produced for intent-aligned tuning."""

    reward_id: str = ""
    goal_id: str = ""
    selection_plan_id: str = ""
    reward_terms: dict[str, float] = field(default_factory=dict)
    kpi_weights: dict[str, float] = field(default_factory=dict)
    parameter_ranges: list[RewardParameterRange] = field(default_factory=list)
    black_box_xapps: list[str] = field(default_factory=list)
    generated_by: str = "heuristic"


@dataclass
class RlGuidanceContract:
    """Explicit Near-RT guidance U_k^RL derived from Non-RT orchestration."""

    guidance_id: str = ""
    goal_id: str = ""
    selection_plan_id: str = ""
    guided_action_region: list[RewardParameterRange] = field(default_factory=list)
    reward_coefficients: dict[str, float] = field(default_factory=dict)
    parameter_priority_vector: dict[str, float] = field(default_factory=dict)
    ambient_parameter_space: AmbientParameterSpace | None = None
    arbitration_variable_specs: list[ArbitrationVariableSpec] = field(default_factory=list)
    selected_safety_rule_set: SafetyConstraintSet | None = None
    black_box_xapps: list[str] = field(default_factory=list)
    source_epoch: int | None = None
    derived_from_intent_id: str = ""
    fallback_used: bool = False
    generated_by: str = "heuristic"


@dataclass
class OrchestrationMemoryState:
    """Memory state exposed by reflective Non-RT orchestration."""

    memory_enabled: bool = False
    memory_mode: str = "full"
    success_cases: list[OrchestrationMemoryCase] = field(default_factory=list)
    failure_cases: list[OrchestrationMemoryCase] = field(default_factory=list)
    success_case_ids: list[str] = field(default_factory=list)
    reflection_note_ids: list[str] = field(default_factory=list)
    last_selection_query_key: SelectionQueryKey | None = None
    last_retrieved_success_case_ids: list[str] = field(default_factory=list)
    last_retrieved_failure_case_ids: list[str] = field(default_factory=list)
    last_retrieved_success_traces: list[RetrievedMemoryCaseTrace] = field(default_factory=list)
    last_retrieved_failure_traces: list[RetrievedMemoryCaseTrace] = field(default_factory=list)
    last_reflection: ReflectionNote | None = None
    notes: list[str] = field(default_factory=list)


def validated_intent_template_from_goal(
    goal: StructuredIntentGoal,
    *,
    validator_name: str = "legacy_goal_adapter",
    projector_name: str = "legacy_goal_adapter",
    fallback_used: bool = False,
    fallback_reason: str = "",
    translation_query_key: TranslationQueryKey | None = None,
    retrieved_context_summary: dict[str, object] | None = None,
    retrieved_context_traces: list[TranslationContextTrace] | None = None,
    raw_llm_candidate: dict[str, object] | None = None,
    validator_outcome: dict[str, object] | None = None,
    projector_output: dict[str, object] | None = None,
    fallback_output: dict[str, object] | None = None,
    validation_report: ValidationReport | None = None,
    projection_report: ProjectionReport | None = None,
    fallback_report: FallbackReport | None = None,
) -> ValidatedIntentTemplate:
    """Build a canonical KPI template view from the legacy structured goal."""

    kpi_template: list[ValidatedKpiTemplateEntry] = []
    for item in list(getattr(goal, "kpi_targets", []) or []):
        semantic_type = _objective_to_semantic_type(getattr(item, "objective", "maximize"))
        target_value = getattr(item, "target_value", None) if semantic_type == "target" else None
        threshold_value = getattr(item, "target_value", None) if semantic_type != "target" else None
        reference_value = getattr(item, "target_value", None)
        tolerance_value = getattr(item, "tolerance", None)
        normalization_coefficient = 1.0
        if tolerance_value not in (None, 0, 0.0):
            normalization_coefficient = float(tolerance_value)
        elif reference_value not in (None, 0, 0.0):
            normalization_coefficient = max(abs(float(reference_value)), 1.0)
        kpi_template.append(
            ValidatedKpiTemplateEntry(
                kpi_name=str(getattr(item, "kpi_name", "") or ""),
                semantic_type=semantic_type,
                target_value=target_value,
                threshold_value=threshold_value,
                weight=float(getattr(item, "weight", 1.0) or 1.0),
                tolerance=getattr(item, "tolerance", None),
                priority=int(getattr(item, "priority", 0) or 0),
                hard=bool(getattr(item, "hard", False)),
                normalization=KpiNormalizationMetadata(
                    unit=str(getattr(item, "unit", "") or ""),
                    normalization_coefficient=float(normalization_coefficient),
                    reference_value=reference_value,
                ),
            )
        )

    return ValidatedIntentTemplate(
        template_id=str(getattr(goal, "goal_id", "") or ""),
        goal_id=str(getattr(goal, "goal_id", "") or ""),
        intent_id=str(getattr(goal, "intent_id", "") or ""),
        intent_text=str(getattr(goal, "intent_text", "") or ""),
        summary=str(getattr(goal, "summary", "") or ""),
        horizon_slots=int(getattr(goal, "horizon_slots", 10) or 10),
        risk_posture=str(getattr(goal, "risk_posture", "balanced") or "balanced"),
        target_entities=list(getattr(goal, "target_entities", []) or ["cluster"]),
        kpi_template=kpi_template,
        constraints=dict(getattr(goal, "constraints", {}) or {}),
        translation_query_key=translation_query_key,
        retrieved_context_summary=dict(retrieved_context_summary or {}),
        retrieved_context_traces=list(retrieved_context_traces or []),
        raw_llm_candidate=dict(raw_llm_candidate or {}),
        validator_outcome=dict(validator_outcome or {}),
        projector_output=dict(projector_output or {}),
        fallback_output=dict(fallback_output or {}),
        validation_report=validation_report
        if validation_report is not None
        else ValidationReport(
            schema_ok=True,
            admissibility_ok=True,
            direction_ok=True,
            range_ok=True,
            validator_name=validator_name,
            normalized_fields=["kpi_targets"],
        ),
        projection_report=projection_report
        if projection_report is not None
        else ProjectionReport(
            projected=False,
            projector_name=projector_name,
            projection_target="legacy_structured_goal",
        ),
        fallback_report=fallback_report
        if fallback_report is not None
        else FallbackReport(
            fallback_used=bool(fallback_used),
            fallback_strategy="legacy_goal_adapter" if fallback_used else "",
            fallback_reason=str(fallback_reason or ""),
        ),
        generated_by=str(getattr(goal, "generated_by", "heuristic") or "heuristic"),
    )


def active_intent_record_from_goal(
    goal: StructuredIntentGoal | None,
    *,
    current_epoch: int | None = None,
    intent_text: str = "",
    change_reason: str = "legacy_policy_update",
    source_name: str = "legacy_policy_store",
) -> ActiveIntentRecord | None:
    """Build a slow-timescale active intent record from the current goal/text."""

    goal_intent_id = str(getattr(goal, "intent_id", "") or "")
    goal_intent_text = str(getattr(goal, "intent_text", "") or "")
    effective_text = str(intent_text or goal_intent_text).strip()
    if goal is None and effective_text == "":
        return None
    return ActiveIntentRecord(
        intent_id=goal_intent_id,
        intent_text=effective_text,
        summary=str(getattr(goal, "summary", "") or ""),
        effective_from_epoch=current_epoch,
        last_update_epoch=current_epoch,
        last_change_reason=str(change_reason or ""),
        source_name=str(source_name or "legacy_policy_store"),
        metadata={
            "goal_id": str(getattr(goal, "goal_id", "") or ""),
            "risk_posture": str(getattr(goal, "risk_posture", "") or ""),
        },
    )


def safety_constraint_set_from_plan(
    plan: XAppSelectionPlan | None,
    *,
    generated_by: str = "legacy_plan_adapter",
) -> SafetyConstraintSet | None:
    """Build a typed safety shield view from the current xApp selection plan."""

    if plan is None:
        return None
    selected_rule_refs: list[SafetyRuleReference] = []
    hard_safety_envelope = list(getattr(plan, "hard_safety_envelope", []) or [])
    forbidden_combinations = list(getattr(plan, "forbidden_combinations", []) or [])
    exploration_health_gate = getattr(plan, "exploration_health_gate", None)
    if len(hard_safety_envelope) > 0:
        selected_rule_refs.append(
            SafetyRuleReference(
                rule_id="rule:hard_safety_envelope",
                config={
                    "range_count": int(len(hard_safety_envelope)),
                    "scopes": sorted(
                        {
                            str(getattr(item, "scope", "cluster") or "cluster")
                            for item in hard_safety_envelope
                        }
                    ),
                },
                description="Bound near-RT tuning to the Non-RT hard safety envelope.",
            )
        )
    if len(forbidden_combinations) > 0:
        selected_rule_refs.append(
            SafetyRuleReference(
                rule_id="rule:forbidden_combinations",
                config={
                    "combination_count": int(len(forbidden_combinations)),
                },
                description="Reject explicitly forbidden field combinations during tuning.",
            )
        )
    if exploration_health_gate is not None:
        selected_rule_refs.append(
            SafetyRuleReference(
                rule_id="rule:exploration_health_gate",
                config={
                    "max_avg_bler": getattr(exploration_health_gate, "max_avg_bler", None),
                    "max_total_queue_bytes": getattr(
                        exploration_health_gate, "max_total_queue_bytes", None
                    ),
                    "min_total_throughput_mbps": getattr(
                        exploration_health_gate,
                        "min_total_throughput_mbps",
                        None,
                    ),
                    "max_handover_pending_ue_count": getattr(
                        exploration_health_gate,
                        "max_handover_pending_ue_count",
                        None,
                    ),
                    "freeze_slots_on_violation": int(
                        getattr(exploration_health_gate, "freeze_slots_on_violation", 4) or 4
                    ),
                },
                description="Freeze or suppress exploration when health-gate thresholds are violated.",
            )
        )
    return SafetyConstraintSet(
        rule_set_id=f"safety:{str(getattr(plan, 'plan_id', '') or '')}",
        goal_id=str(getattr(plan, "goal_id", "") or ""),
        selection_plan_id=str(getattr(plan, "plan_id", "") or ""),
        selected_rule_refs=selected_rule_refs,
        safety_envelope=hard_safety_envelope,
        forbidden_combinations=forbidden_combinations,
        exploration_health_gate=exploration_health_gate,
        generated_by=str(getattr(plan, "generated_by", generated_by) or generated_by),
    )


def orchestration_plan_view_from_plan(
    plan: XAppSelectionPlan | None,
    *,
    eta: A1GuidanceProfile | None = None,
    chi: ArbitrationProfileRef | None = None,
) -> OrchestrationPlanView | None:
    """Build a unified g_k view from the legacy selection plan."""

    if plan is None:
        return None
    effective_eta = eta if eta is not None else getattr(plan, "eta", None)
    if chi is None:
        chi = getattr(plan, "chi", None) or derive_arbitration_profile_from_plan(plan)
    return OrchestrationPlanView(
        plan_id=str(getattr(plan, "plan_id", "") or ""),
        goal_id=str(getattr(plan, "goal_id", "") or ""),
        x=list(getattr(plan, "selected_xapps", []) or []),
        B=list(getattr(plan, "hard_safety_envelope", []) or []),
        eta=effective_eta,
        chi=chi,
        forbidden_combinations=list(getattr(plan, "forbidden_combinations", []) or []),
        exploration_health_gate=getattr(plan, "exploration_health_gate", None),
        qos_protection_profile=getattr(plan, "qos_protection_profile", None),
        feasibility_summary=getattr(plan, "feasibility_summary", None),
        generated_by=str(getattr(plan, "generated_by", "heuristic") or "heuristic"),
    )


def a1_guidance_profile_payload(profile: A1GuidanceProfile | dict[str, object] | None) -> dict[str, object]:
    """Convert eta_k into a stable dict payload for logging and consumers."""

    if profile is None:
        return {}
    if isinstance(profile, dict):
        return dict(profile)
    return {
        "profile_name": str(getattr(profile, "profile_name", "") or ""),
        "goal_focus": str(getattr(profile, "goal_focus", "balanced") or "balanced"),
        "reward_term_bias": dict(getattr(profile, "reward_term_bias", {}) or {}),
        "field_priority_hints": dict(getattr(profile, "field_priority_hints", {}) or {}),
        "safety_rule_bias": list(getattr(profile, "safety_rule_bias", []) or []),
        "action_region_bias": str(getattr(profile, "action_region_bias", "balanced") or "balanced"),
        "selected_categories": list(getattr(profile, "selected_categories", []) or []),
        "risk_posture": str(getattr(profile, "risk_posture", "balanced") or "balanced"),
        "defaulted": bool(getattr(profile, "defaulted", False)),
        "notes": list(getattr(profile, "notes", []) or []),
    }


def qos_protection_profile_payload(
    profile: QoSProtectionProfile | dict[str, object] | None,
) -> dict[str, object]:
    """Convert QoS protection profile into a stable dict payload."""

    if profile is None:
        return {}
    if isinstance(profile, dict):
        return dict(profile)
    return {
        "profile_id": str(getattr(profile, "profile_id", "") or ""),
        "strictness": str(getattr(profile, "strictness", "balanced") or "balanced"),
        "constraints": [
            {
                "metric_name": str(getattr(item, "metric_name", "") or ""),
                "semantic_type": str(getattr(item, "semantic_type", "max") or "max"),
                "threshold_value": getattr(item, "threshold_value", None),
                "outage_budget": float(getattr(item, "outage_budget", 0.05) or 0.05),
                "scope": str(getattr(item, "scope", "cluster") or "cluster"),
                "target_slice_id": str(getattr(item, "target_slice_id", "") or ""),
                "target_traffic_type": str(getattr(item, "target_traffic_type", "") or ""),
                "notes": list(getattr(item, "notes", []) or []),
            }
            for item in list(getattr(profile, "constraints", []) or [])
        ],
        "required_categories": list(getattr(profile, "required_categories", []) or []),
        "generated_by": str(getattr(profile, "generated_by", "heuristic") or "heuristic"),
    }


def _canonical_xapp_category_name(xapp_name: str) -> str:
    normalized_name = str(xapp_name or "").strip()
    known_categories = (
        "mobility_predictor_xapp",
        "traffic_steering_xapp",
        "power_saving_xapp",
        "spectrum_optimizer_xapp",
        "latency_scheduler_xapp",
        "massive_mimo_beamformer_xapp",
        "ran_slicing_manager_xapp",
        "urllc_guard_xapp",
        "cell_sleep_xapp",
        "goal_parameter_tuning_xapp",
    )
    for category_name in known_categories:
        if normalized_name == category_name or normalized_name.startswith(f"{category_name}_"):
            return str(category_name)
    return normalized_name


def derive_arbitration_profile_from_plan(plan: XAppSelectionPlan | None) -> ArbitrationProfileRef | None:
    """Derive a default but explicit chi profile from the selected xApp portfolio."""

    if plan is None:
        return None
    selected_xapps = [
        str(item).strip()
        for item in list(getattr(plan, "selected_xapps", []) or [])
        if str(item).strip() != ""
    ]
    selected_categories = list(dict.fromkeys(_canonical_xapp_category_name(item) for item in selected_xapps))

    field_owners: dict[str, set[str]] = {}
    for parameter_range in list(getattr(plan, "hard_safety_envelope", []) or []):
        field_name = str(getattr(parameter_range, "field_name", "") or "").strip()
        target_xapp_name = str(getattr(parameter_range, "target_xapp_name", "") or "").strip()
        if field_name == "" or target_xapp_name == "":
            continue
        field_owners.setdefault(field_name, set()).add(_canonical_xapp_category_name(target_xapp_name))

    overlapping_field_categories = {
        field_name: owners
        for field_name, owners in field_owners.items()
        if len(owners) > 1
    }
    recommendation_categories = {
        "traffic_steering_xapp",
        "mobility_predictor_xapp",
        "cell_sleep_xapp",
    }
    guard_categories = {
        "urllc_guard_xapp",
        "latency_scheduler_xapp",
        "ran_slicing_manager_xapp",
    }
    scheduler_dominant_categories = {
        "latency_scheduler_xapp",
        "spectrum_optimizer_xapp",
        "ran_slicing_manager_xapp",
        "urllc_guard_xapp",
    }

    composition_mode = "parameter_tuning"
    arbitration_mode = "weighted_merge"
    notes: list[str] = []
    if any(category_name in recommendation_categories for category_name in selected_categories):
        composition_mode = "mixed"
        arbitration_mode = "guardrail_first"
        notes.append("recommendation_style_categories_present")
    if len(overlapping_field_categories) > 0:
        composition_mode = "mixed"
        arbitration_mode = "priority_override"
        notes.append("overlapping_field_owners_detected")
    if len([category for category in selected_categories if category in scheduler_dominant_categories]) >= 3:
        arbitration_mode = "priority_override"
        notes.append("scheduler_dominant_portfolio")
    if composition_mode == "parameter_tuning" and any(category_name in guard_categories for category_name in selected_categories):
        notes.append("guardrail_categories_present")

    default_override_order = [
        category_name
        for category_name in (
            "urllc_guard_xapp",
            "latency_scheduler_xapp",
            "ran_slicing_manager_xapp",
            "spectrum_optimizer_xapp",
            "massive_mimo_beamformer_xapp",
            "traffic_steering_xapp",
            "mobility_predictor_xapp",
            "power_saving_xapp",
            "cell_sleep_xapp",
        )
        if category_name in selected_categories
    ]

    return ArbitrationProfileRef(
        profile_name=f"chi:{str(getattr(plan, 'plan_id', '') or 'default')}",
        composition_mode=composition_mode,
        arbitration_mode=arbitration_mode,
        merge_scope="field",
        override_order=default_override_order,
        defaulted=True,
        notes=notes,
    )


def rl_guidance_contract_from_reward_spec(
    reward_spec: RewardSpec | None,
    *,
    selected_safety_rule_set: SafetyConstraintSet | None = None,
    source_epoch: int | None = None,
    derived_from_intent_id: str = "",
    fallback_used: bool = False,
) -> RlGuidanceContract | None:
    """Build an explicit U_k^RL view from the legacy reward specification."""

    if reward_spec is None:
        return None
    parameter_priority_vector = {
        str(getattr(item, "field_name", "") or ""): 1.0
        for item in list(getattr(reward_spec, "parameter_ranges", []) or [])
        if str(getattr(item, "field_name", "") or "").strip() != ""
    }
    return RlGuidanceContract(
        guidance_id=str(getattr(reward_spec, "reward_id", "") or ""),
        goal_id=str(getattr(reward_spec, "goal_id", "") or ""),
        selection_plan_id=str(getattr(reward_spec, "selection_plan_id", "") or ""),
        guided_action_region=list(getattr(reward_spec, "parameter_ranges", []) or []),
        reward_coefficients={
            str(key): float(value)
            for key, value in dict(getattr(reward_spec, "reward_terms", {}) or {}).items()
        },
        parameter_priority_vector=parameter_priority_vector,
        selected_safety_rule_set=selected_safety_rule_set,
        black_box_xapps=list(getattr(reward_spec, "black_box_xapps", []) or []),
        source_epoch=source_epoch,
        derived_from_intent_id=str(derived_from_intent_id or ""),
        fallback_used=bool(fallback_used),
        generated_by=str(getattr(reward_spec, "generated_by", "heuristic") or "heuristic"),
    )


def theta_runtime_state_payload(runtime_state: ThetaRuntimeState | None) -> dict[str, object]:
    """Serialize the explicit theta runtime state for logging/export."""

    if runtime_state is None:
        return {}
    return {
        "state_id": str(getattr(runtime_state, "state_id", "") or ""),
        "dimension_count": int(getattr(runtime_state, "dimension_count", 0) or 0),
        "values": dict(getattr(runtime_state, "values", {}) or {}),
    }


def omega_runtime_state_payload(runtime_state: OmegaRuntimeState | None) -> dict[str, object]:
    """Serialize the explicit omega runtime state for logging/export."""

    if runtime_state is None:
        return {}
    return {
        "state_id": str(getattr(runtime_state, "state_id", "") or ""),
        "field_count": int(getattr(runtime_state, "field_count", 0) or 0),
        "field_states": [
            {
                "field_name": str(getattr(field_state, "field_name", "") or ""),
                "arbitration_mode": str(
                    getattr(field_state, "arbitration_mode", "weighted_merge") or "weighted_merge"
                ),
                "weights": {
                    str(option_id): float(weight)
                    for option_id, weight in dict(getattr(field_state, "weights", {}) or {}).items()
                },
            }
            for field_state in list(getattr(runtime_state, "field_states", []) or [])
        ],
    }


def near_rt_runtime_state_payload(runtime_state: NearRtRuntimeState | None) -> dict[str, object]:
    """Serialize the typed Near-RT runtime state for logging/export."""

    if runtime_state is None:
        return {}
    return {
        "runtime_id": str(getattr(runtime_state, "runtime_id", "") or ""),
        "active_parameter_mask": {
            str(field_name): int(flag)
            for field_name, flag in dict(getattr(runtime_state, "active_parameter_mask", {}) or {}).items()
        },
        "active_parameter_count": int(getattr(runtime_state, "active_parameter_count", 0) or 0),
        "theta_state": theta_runtime_state_payload(getattr(runtime_state, "theta_state", None)),
        "omega_state": omega_runtime_state_payload(getattr(runtime_state, "omega_state", None)),
    }
