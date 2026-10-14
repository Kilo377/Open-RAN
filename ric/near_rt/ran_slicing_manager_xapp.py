"""Intent-aware RAN slicing xApp."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    clamp,
    dominant_slice_id,
    goal_prefers_latency,
    goal_prefers_reliability,
    make_analysis_constraint,
    safe_mean,
    slice_pressures,
)


class RANSlicingManagerXApp(XAppBase):
    def __init__(
        self,
        name: str = "ran_slicing_manager_xapp",
        priority: int = 12,
        slice_dominance_threshold: float = 0.45,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "scheduler_policy_hint",
                "slice_weight_adjustment",
                "prb_weight",
                "priority_coefficient",
            ],
        )
        self.slice_dominance_threshold = float(slice_dominance_threshold)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        dominant_slice, dominant_share = dominant_slice_id(observation)
        pressures = slice_pressures(observation)
        total_pressure = max(1.0, float(sum(pressures.values())))
        base_weights = dict(getattr(policy_snapshot, "preferred_slice_weight_adjustment", {}) or {})
        if len(base_weights) == 0:
            base_weights = {"urllc": 1.0, "embb": 1.0, "mmtc": 1.0}

        latency_guard = goal_prefers_latency(policy_snapshot)
        reliability_guard = goal_prefers_reliability(policy_snapshot)
        pressure_ratios = {
            str(slice_id): float(value) / total_pressure
            for slice_id, value in pressures.items()
        }
        for slice_id in list(base_weights.keys()):
            ratio = float(pressure_ratios.get(slice_id, 0.0))
            guard_bonus = 0.0
            if str(slice_id).lower() == "urllc":
                guard_bonus += 0.18 if latency_guard else 0.0
                guard_bonus += 0.12 if reliability_guard else 0.0
            base_weights[slice_id] = clamp(
                float(base_weights[slice_id]) + 0.55 * ratio + guard_bonus,
                0.60,
                1.80,
            )

        if dominant_slice == "urllc":
            base_weights["embb"] = clamp(float(base_weights.get("embb", 1.0)) - 0.10, 0.70, 1.50)
            base_weights["mmtc"] = clamp(float(base_weights.get("mmtc", 1.0)) - 0.05, 0.70, 1.50)
        elif dominant_slice == "embb":
            base_weights["urllc"] = clamp(float(base_weights.get("urllc", 1.0)) - 0.05, 0.75, 1.50)
            base_weights["mmtc"] = clamp(float(base_weights.get("mmtc", 1.0)) - 0.10, 0.70, 1.50)

        action = ConfigAction(source_name=self.name)
        if dominant_share >= self.slice_dominance_threshold:
            action = ConfigAction(
                scheduler_policy_hint="slice_bias",
                slice_weight_adjustment=base_weights,
                prb_weight=(clamp(0.85 + 0.35 * dominant_share, 0.5, 1.8) if not getattr(observation, 'rt_public_projection', False) else None),
                priority_coefficient=(clamp(1.0 + 0.75 * dominant_share, 1.0, 3.0) if not getattr(observation, 'rt_public_projection', False) else None),
                apply_scope="cluster",
                source_name=self.name,
            )

        analysis_constraint = make_analysis_constraint(
            constraint_type="ran_slicing_analysis",
            source_name=self.name,
            params={
                "dominant_slice_id": dominant_slice,
                "dominant_slice_share": float(dominant_share),
                "slice_pressures": {key: float(value) for key, value in pressures.items()},
                "slice_pressure_ratios": dict(pressure_ratios),
                "proposed_slice_weight_adjustment": dict(base_weights),
                "mean_slice_pressure": float(safe_mean(pressures.values(), default=0.0)),
                "prb_weight": float(getattr(action, "prb_weight", 0.0) or 0.0),
                "priority_coefficient": float(getattr(action, "priority_coefficient", 0.0) or 0.0),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
