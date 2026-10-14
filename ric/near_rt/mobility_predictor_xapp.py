"""Mobility robustness xApp with prediction-aware handover tuning."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    beam_misalignment_risk_score,
    candidate_cell_ranking,
    clamp,
    future_cqi_estimate,
    goal_prefers_latency,
    goal_prefers_mobility,
    iter_ue_observations,
    make_analysis_constraint,
    mobility_risk_score,
    safe_mean,
    ue_channel_quality_confidence,
)


class MobilityPredictorXApp(XAppBase):
    def __init__(
        self,
        name: str = "mobility_predictor_xapp",
        priority: int = 18,
        risk_threshold: float = 0.55,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "mobility_bias_db",
                "steering_hysteresis_db",
                "time_to_trigger_ms",
                "mobility_margin_db",
                "measurement_event_offset_db",
            ],
        )
        self.risk_threshold = float(risk_threshold)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        ues = iter_ue_observations(observation)
        empty_action = ConfigAction(source_name=self.name, apply_scope="cluster")
        if len(ues) == 0:
            return ControlDecision(config_action=empty_action)

        scored_ues = [
            (
                ue,
                mobility_risk_score(ue),
                beam_misalignment_risk_score(ue),
                future_cqi_estimate(ue),
            )
            for ue in ues
        ]
        focus_ue, max_risk, max_beam_risk, future_cqi = max(
            scored_ues,
            key=lambda item: (float(item[1]), float(item[2])),
        )
        mean_confidence = safe_mean(
            [ue_channel_quality_confidence(ue) for ue, _, _, _ in scored_ues],
            default=0.5,
        )

        base_mobility_bias_db = float(
            getattr(policy_snapshot, "preferred_mobility_bias_db", 0.0) or 0.0
        )
        base_hysteresis_db = float(
            getattr(policy_snapshot, "preferred_steering_hysteresis_db", 3.0) or 3.0
        )
        base_event_offset_db = float(
            getattr(policy_snapshot, "preferred_measurement_event_offset_db", 2.0) or 2.0
        )
        base_ttt_ms = float(getattr(policy_snapshot, "preferred_time_to_trigger_ms", 120.0) or 120.0)
        base_margin_db = float(getattr(policy_snapshot, "preferred_mobility_margin_db", 2.0) or 2.0)
        latency_sensitive = goal_prefers_latency(policy_snapshot)
        mobility_intent = goal_prefers_mobility(policy_snapshot)
        robust_guard = clamp(
            0.40 * max_risk + 0.25 * max_beam_risk + 0.20 * (1.0 - mean_confidence)
            + (0.10 if latency_sensitive else 0.0)
            + (0.10 if mobility_intent else 0.0),
            0.0,
            1.0,
        )

        action = empty_action
        if max_risk >= self.risk_threshold:
            mobility_bias_db = clamp(base_mobility_bias_db + 4.0 * max_risk * mean_confidence, -6.0, 8.0)
            if mean_confidence >= 0.45:
                steering_hysteresis_db = clamp(
                    base_hysteresis_db - 0.7 * max_risk + 0.5 * robust_guard,
                    0.5,
                    12.0,
                )
                measurement_event_offset_db = clamp(
                    base_event_offset_db - 0.7 * max_risk + 0.35 * robust_guard,
                    0.5,
                    8.0,
                )
                time_to_trigger_ms = clamp(base_ttt_ms - 40.0 * max_risk + 60.0 * robust_guard, 40.0, 480.0)
            else:
                steering_hysteresis_db = clamp(
                    base_hysteresis_db + 1.5 * (1.0 - mean_confidence),
                    0.5,
                    12.0,
                )
                measurement_event_offset_db = clamp(
                    base_event_offset_db + 0.8 * (1.0 - mean_confidence),
                    0.5,
                    8.0,
                )
                time_to_trigger_ms = clamp(base_ttt_ms + 120.0 * (1.0 - mean_confidence), 80.0, 640.0)
            action = ConfigAction(
                mobility_bias_db=mobility_bias_db,
                steering_hysteresis_db=steering_hysteresis_db,
                time_to_trigger_ms=(time_to_trigger_ms if not getattr(observation, 'rt_public_projection', False) else None),
                mobility_margin_db=(clamp(base_margin_db + 2.5 * robust_guard - 1.2 * max_risk, 0.5, 8.0) if not getattr(observation, 'rt_public_projection', False) else None),
                measurement_event_offset_db=measurement_event_offset_db,
                apply_scope="cluster",
                source_name=self.name,
            )

        analysis_constraint = make_analysis_constraint(
            constraint_type="mobility_prediction",
            source_name=self.name,
            params={
                "focus_ue_id": str(getattr(focus_ue, "ue_id", "")),
                "handover_risk_score": float(max_risk),
                "beam_misalignment_risk": float(max_beam_risk),
                "future_cqi_estimate": float(future_cqi),
                "candidate_cell_ranking": candidate_cell_ranking(focus_ue),
                "mean_channel_quality_confidence": float(mean_confidence),
                "robust_guard_score": float(robust_guard),
                "time_to_trigger_ms": float(getattr(action, "time_to_trigger_ms", 0.0) or 0.0),
                "mobility_margin_db": float(getattr(action, "mobility_margin_db", 0.0) or 0.0),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
