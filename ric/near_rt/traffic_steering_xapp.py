"""Traffic-steering xApp with intent-aware load and edge-user scoring."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    cell_interference_score,
    cell_load_score,
    clamp,
    goal_prefers_latency,
    goal_prefers_mobility,
    goal_prefers_throughput,
    iter_cell_observations,
    iter_ue_observations,
    make_analysis_constraint,
    mobility_risk_score,
    safe_mean,
)


class TrafficSteeringXApp(XAppBase):
    def __init__(
        self,
        name: str = "traffic_steering_xapp",
        priority: int = 19,
        load_threshold: float = 0.70,
        edge_sinr_threshold_db: float = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "traffic_steering_preference",
                "steering_aggressiveness",
                "load_threshold",
                "cio_bias_db",
                "mobility_bias_db",
                "measurement_event_offset_db",
            ],
        )
        self.load_threshold = float(load_threshold)
        self.edge_sinr_threshold_db = float(edge_sinr_threshold_db)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        cells = iter_cell_observations(observation)
        ues = iter_ue_observations(observation)
        empty_action = ConfigAction(source_name=self.name, apply_scope="cluster")
        if len(cells) == 0:
            return ControlDecision(config_action=empty_action)

        busiest_cell = max(cells, key=cell_load_score)
        highest_load = cell_load_score(busiest_cell)
        mean_interference = safe_mean(
            [cell_interference_score(cell) for cell in cells],
            default=0.0,
        )
        edge_ues = [
            ue for ue in ues
            if float(getattr(ue, "sinr_db", 0.0) or 0.0) <= self.edge_sinr_threshold_db
            or mobility_risk_score(ue) >= 0.60
        ]

        preferred_steering = str(
            getattr(policy_snapshot, "preferred_traffic_steering_preference", "balanced")
            or "balanced"
        )
        base_mobility_bias_db = float(
            getattr(policy_snapshot, "preferred_mobility_bias_db", 0.0) or 0.0
        )
        base_event_offset_db = float(
            getattr(policy_snapshot, "preferred_measurement_event_offset_db", 2.0) or 2.0
        )
        base_aggressiveness = float(
            getattr(policy_snapshot, "preferred_steering_aggressiveness", 0.55) or 0.55
        )
        base_cio_bias_db = float(
            getattr(policy_snapshot, "preferred_cio_bias_db", 0.0) or 0.0
        )
        configured_load_threshold = float(
            getattr(policy_snapshot, "preferred_load_threshold", self.load_threshold)
            or self.load_threshold
        )
        edge_ratio = float(len(edge_ues)) / max(1.0, float(len(ues)))
        throughput_weight = 1.0 if goal_prefers_throughput(policy_snapshot) else 0.55
        latency_guard = 0.65 if goal_prefers_latency(policy_snapshot) else 0.0
        mobility_guard = 0.55 if goal_prefers_mobility(policy_snapshot) else 0.0
        interference_guard = clamp(mean_interference / 6.0, 0.0, 1.0)
        steering_pressure = clamp(
            0.45 * max(0.0, highest_load - configured_load_threshold) / max(0.05, 1.0 - configured_load_threshold)
            + 0.35 * edge_ratio
            + 0.20 * throughput_weight
            - 0.18 * latency_guard * edge_ratio
            - 0.18 * mobility_guard * edge_ratio
            - 0.16 * interference_guard,
            0.0,
            1.0,
        )
        steering_aggressiveness = clamp(
            0.25 + 0.60 * base_aggressiveness + 0.45 * steering_pressure,
            0.05,
            1.0,
        )

        action = empty_action
        if highest_load >= configured_load_threshold or len(edge_ues) > 0:
            action = ConfigAction(
                traffic_steering_preference=(
                    preferred_steering if preferred_steering != "balanced" else "edge_offload"
                ),
                steering_aggressiveness=steering_aggressiveness,
                load_threshold=configured_load_threshold,
                cio_bias_db=clamp(base_cio_bias_db + 4.5 * steering_pressure, -6.0, 6.0),
                mobility_bias_db=clamp(
                    base_mobility_bias_db + 3.5 * steering_aggressiveness * max(highest_load, edge_ratio),
                    -8.0,
                    8.0,
                ),
                measurement_event_offset_db=clamp(
                    base_event_offset_db - 0.8 * steering_aggressiveness * max(highest_load, edge_ratio),
                    0.5,
                    8.0,
                ),
                apply_scope="cluster",
                source_name=self.name,
            )
        elif mean_interference > 3.0:
            action = ConfigAction(
                traffic_steering_preference="stay_put",
                steering_aggressiveness=clamp(0.20 + 0.20 * base_aggressiveness, 0.05, 0.45),
                load_threshold=configured_load_threshold,
                cio_bias_db=clamp(base_cio_bias_db - 1.0, -6.0, 6.0),
                mobility_bias_db=min(0.0, base_mobility_bias_db - 1.0),
                measurement_event_offset_db=clamp(base_event_offset_db + 0.5, 0.5, 8.0),
                apply_scope="cluster",
                source_name=self.name,
            )

        analysis_constraint = make_analysis_constraint(
            constraint_type="traffic_steering_analysis",
            source_name=self.name,
            params={
                "busiest_cell_id": str(getattr(busiest_cell, "cell_id", "cell_0")),
                "highest_load_score": float(highest_load),
                "mean_interference_penalty_db": float(mean_interference),
                "edge_ue_count": int(len(edge_ues)),
                "edge_ue_ratio": float(edge_ratio),
                "steering_pressure": float(steering_pressure),
                "steering_aggressiveness": float(
                    getattr(action, "steering_aggressiveness", 0.0) or 0.0
                ),
                "cio_bias_db": float(getattr(action, "cio_bias_db", 0.0) or 0.0),
                "load_threshold": float(configured_load_threshold),
                "preference": str(
                    getattr(action, "traffic_steering_preference", "balanced") or "balanced"
                ),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
