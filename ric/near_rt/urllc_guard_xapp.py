"""URLLC guardrail xApp for bounded reliability and latency protection."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    clamp,
    is_urllc_ue,
    iter_ue_observations,
    make_analysis_constraint,
    select_urlcc_focus_cell,
    ue_latency_pressure_score,
)


class URLLCGuardXApp(XAppBase):
    def __init__(
        self,
        name: str = "urllc_guard_xapp",
        priority: int = 25,
        queue_threshold_bytes: int = 1200,
        bler_threshold: float = 0.12,
        latency_budget_threshold_ms: float = 20.0,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "scheduler_policy_hint",
                "latency_bias",
                "power_cap_dbm",
                "prb_weight",
                "priority_coefficient",
                "delay_threshold_ms",
            ],
        )
        self.queue_threshold_bytes = int(queue_threshold_bytes)
        self.bler_threshold = float(bler_threshold)
        self.latency_budget_threshold_ms = float(latency_budget_threshold_ms)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        urllc_ues = [ue for ue in iter_ue_observations(observation) if is_urllc_ue(ue)]
        empty_action = ConfigAction(source_name=self.name)
        if len(urllc_ues) == 0:
            return ControlDecision(config_action=empty_action)

        target_cell = select_urlcc_focus_cell(observation)
        total_urllc_queue_bytes = sum(int(getattr(ue, "queue_bytes", 0) or 0) for ue in urllc_ues)
        max_guard_pressure = max(ue_latency_pressure_score(ue) for ue in urllc_ues)
        worst_urllc_bler = max(float(getattr(ue, "bler", 0.0) or 0.0) for ue in urllc_ues)
        requires_guard = (
            total_urllc_queue_bytes >= self.queue_threshold_bytes
            or worst_urllc_bler >= self.bler_threshold
            or max_guard_pressure >= 0.85
        )

        action = empty_action
        if requires_guard and target_cell is not None:
            base_latency_bias = float(
                getattr(policy_snapshot, "preferred_latency_bias", 1.0) or 1.0
            )
            base_power_cap_dbm = float(
                getattr(policy_snapshot, "preferred_power_cap_dbm", 30.0) or 30.0
            )
            action = ConfigAction(
                target_cell_id=str(target_cell.cell_id),
                scheduler_policy_hint="latency_bias",
                latency_bias=clamp(base_latency_bias + 0.60 * max_guard_pressure, 0.9, 2.0),
                power_cap_dbm=max(30.0, base_power_cap_dbm),
                prb_weight=clamp(1.0 + 0.35 * max_guard_pressure, 1.0, 1.8),
                priority_coefficient=clamp(1.5 + 0.75 * max_guard_pressure, 1.2, 3.0),
                delay_threshold_ms=clamp(
                    min(
                        self.latency_budget_threshold_ms,
                        float(getattr(policy_snapshot, "preferred_delay_threshold_ms", self.latency_budget_threshold_ms) or self.latency_budget_threshold_ms),
                    )
                    - 3.0 * clamp(max_guard_pressure, 0.0, 1.0),
                    5.0,
                    self.latency_budget_threshold_ms,
                ),
                source_name=self.name,
            )

        analysis_constraint = make_analysis_constraint(
            constraint_type="urllc_guard",
            source_name=self.name,
            scope="cell" if target_cell is not None else "cluster",
            target_id=(str(target_cell.cell_id) if target_cell is not None else None),
            params={
                "guard_required": bool(requires_guard),
                "urllc_queue_bytes": int(total_urllc_queue_bytes),
                "worst_urllc_bler": float(worst_urllc_bler),
                "max_guard_pressure": float(max_guard_pressure),
                "guard_resource_reservation": float(clamp(0.10 + 0.15 * max_guard_pressure, 0.05, 0.35)),
                "retx_priority_floor": 1.0,
                "reject_other_actions": bool(max_guard_pressure >= 1.10),
                "prb_weight": float(getattr(action, "prb_weight", 0.0) or 0.0),
                "priority_coefficient": float(getattr(action, "priority_coefficient", 0.0) or 0.0),
                "delay_threshold_ms": float(getattr(action, "delay_threshold_ms", 0.0) or 0.0),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
