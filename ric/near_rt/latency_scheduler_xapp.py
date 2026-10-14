"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: latency_scheduler_xapp.py
Description: Near-RT latency-aware scheduler xApp. The xApp combines queue,
HOL delay, HARQ urgency, and slice mix to emit bounded QoS tuning controls.
"""

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    clamp,
    goal_prefers_latency,
    goal_prefers_reliability,
    is_urllc_ue,
    iter_ue_observations,
    make_analysis_constraint,
    safe_mean,
    slice_delay_budget_ms,
    ue_latency_pressure_score,
)


class LatencySchedulerXApp(XAppBase):
    """
    Implements an intent-aware latency scheduler xApp.
    """

    def __init__(
        self,
        name: str = "latency_scheduler_xapp",
        priority: int = 10,
        queue_threshold_bytes: int = 4000,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "scheduler_policy_hint",
                "latency_bias",
                "slice_weight_adjustment",
                "prb_weight",
                "priority_coefficient",
                "delay_threshold_ms",
            ],
        )
        self.queue_threshold_bytes = queue_threshold_bytes

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        ues = iter_ue_observations(observation)
        urgent_ues = [ue for ue in ues if is_urllc_ue(ue)]
        latency_pressures = [ue_latency_pressure_score(ue) for ue in urgent_ues]
        max_latency_pressure = max(latency_pressures, default=0.0)
        mean_latency_pressure = safe_mean(latency_pressures, default=0.0)
        latency_intent = goal_prefers_latency(policy_snapshot)
        reliability_intent = goal_prefers_reliability(policy_snapshot)
        pressure = clamp(
            0.45 * max_latency_pressure
            + 0.25 * mean_latency_pressure
            + 0.20 * (float(observation.total_queue_bytes) / max(1.0, float(self.queue_threshold_bytes)))
            + (0.12 if latency_intent else 0.0)
            + (0.08 if reliability_intent else 0.0),
            0.0,
            2.0,
        )
        preferred_scheduler_policy_hint = getattr(policy_snapshot, "preferred_scheduler_policy_hint", None)
        preferred_latency_bias = getattr(policy_snapshot, "preferred_latency_bias", None)
        preferred_slice_weight_adjustment = dict(
            getattr(policy_snapshot, "preferred_slice_weight_adjustment", {}) or {}
        )
        if len(preferred_slice_weight_adjustment) == 0:
            preferred_slice_weight_adjustment = {"urllc": 1.0, "embb": 1.0, "mmtc": 1.0}

        if (
            observation.total_queue_bytes > self.queue_threshold_bytes
            and preferred_scheduler_policy_hint == "slice_bias"
            and len(urgent_ues) == 0
        ):
            action = ConfigAction(
                scheduler_policy_hint="slice_bias",
                slice_weight_adjustment=preferred_slice_weight_adjustment,
                source_name=self.name,
            )
            return ControlDecision(
                config_action=action,
                policy_constraints=[
                    make_analysis_constraint(
                        constraint_type="latency_scheduler_analysis",
                        source_name=self.name,
                        params={
                            "total_queue_bytes": int(observation.total_queue_bytes),
                            "urgent_ue_count": int(len(urgent_ues)),
                            "max_latency_pressure": float(max_latency_pressure),
                            "mean_latency_pressure": float(mean_latency_pressure),
                            "qos_pressure": float(pressure),
                            "selected_policy_hint": "slice_bias",
                            "preserved_non_rt_slice_weights": True,
                        },
                    )
                ],
            )

        for slice_id, base_weight in list(preferred_slice_weight_adjustment.items()):
            delay_budget = slice_delay_budget_ms(slice_id)
            slice_multiplier = 1.0
            if str(slice_id).lower() == "urllc":
                slice_multiplier += 0.45 * clamp(pressure, 0.0, 1.5)
            elif str(slice_id).lower() == "embb":
                slice_multiplier -= 0.12 * clamp(pressure, 0.0, 1.0)
            preferred_slice_weight_adjustment[slice_id] = clamp(
                float(base_weight) * slice_multiplier,
                0.65,
                1.85,
            )

        if observation.total_queue_bytes > self.queue_threshold_bytes or pressure >= 0.65:
            policy_hint = (
                str(preferred_scheduler_policy_hint)
                if preferred_scheduler_policy_hint is not None
                else ("slice_bias" if len(urgent_ues) > 0 else "latency_bias")
            )
            action = ConfigAction(
                scheduler_policy_hint=policy_hint,
                latency_bias=clamp(
                    float(preferred_latency_bias) if preferred_latency_bias is not None else 0.85 + 0.55 * pressure,
                    0.5,
                    2.5,
                ),
                slice_weight_adjustment=preferred_slice_weight_adjustment,
                prb_weight=(clamp(0.70 + 0.45 * pressure, 0.4, 1.8) if not getattr(observation, 'rt_public_projection', False) else None),
                priority_coefficient=(clamp(1.0 + 0.65 * pressure, 1.0, 3.0) if not getattr(observation, 'rt_public_projection', False) else None),
                delay_threshold_ms=(clamp(
                    float(getattr(policy_snapshot, "preferred_delay_threshold_ms", 20.0) or 20.0)
                    - 5.0 * clamp(pressure, 0.0, 1.0),
                    5.0,
                    80.0,
                ) if not getattr(observation, 'rt_public_projection', False) else None),
                source_name=self.name,
            )
            return ControlDecision(
                config_action=action,
                policy_constraints=[
                    make_analysis_constraint(
                        constraint_type="latency_scheduler_analysis",
                        source_name=self.name,
                        params={
                            "total_queue_bytes": int(observation.total_queue_bytes),
                            "urgent_ue_count": int(len(urgent_ues)),
                            "max_latency_pressure": float(max_latency_pressure),
                            "mean_latency_pressure": float(mean_latency_pressure),
                            "qos_pressure": float(pressure),
                            "selected_policy_hint": str(policy_hint),
                            "prb_weight": float(action.prb_weight or 0.0),
                            "priority_coefficient": float(action.priority_coefficient or 0.0),
                            "delay_threshold_ms": float(action.delay_threshold_ms or 0.0),
                        },
                    )
                ],
            )

        return ControlDecision(
            config_action=ConfigAction(source_name=self.name),
            policy_constraints=[
                make_analysis_constraint(
                    constraint_type="latency_scheduler_analysis",
                    source_name=self.name,
                    params={
                        "total_queue_bytes": int(observation.total_queue_bytes),
                        "urgent_ue_count": int(len(urgent_ues)),
                        "max_latency_pressure": float(max_latency_pressure),
                        "mean_latency_pressure": float(mean_latency_pressure),
                        "qos_pressure": float(pressure),
                        "selected_policy_hint": "idle",
                    },
                )
            ],
        )
