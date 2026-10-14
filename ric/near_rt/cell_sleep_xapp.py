"""Near-RT xApp that recommends guarded cell sleep modes."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    cell_load_score,
    clamp,
    goal_prefers_energy,
    goal_prefers_latency,
    iter_cell_observations,
    make_analysis_constraint,
)


class CellSleepXApp(XAppBase):
    def __init__(
        self,
        *,
        name: str = "cell_sleep_xapp",
        priority: int = 9,
        micro_sleep_load_threshold: float = 0.28,
        light_sleep_load_threshold: float = 0.18,
        deep_sleep_load_threshold: float = 0.08,
        max_queue_bytes_for_sleep: int = 4000,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=["cell_sleep_mode", "power_cap_dbm"],
        )
        self.micro_sleep_load_threshold = max(0.0, float(micro_sleep_load_threshold))
        self.light_sleep_load_threshold = max(0.0, float(light_sleep_load_threshold))
        self.deep_sleep_load_threshold = max(0.0, float(deep_sleep_load_threshold))
        self.max_queue_bytes_for_sleep = max(0, int(max_queue_bytes_for_sleep))

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        candidate_cells = [
            cell
            for cell in iter_cell_observations(observation)
            if not bool(getattr(cell, "is_primary", False))
        ]
        if len(candidate_cells) == 0:
            return ControlDecision(config_action=ConfigAction(source_name=self.name))

        target_cell = min(
            candidate_cells,
            key=lambda cell: (
                cell_load_score(cell),
                float(getattr(cell, "total_queue_bytes", 0) or 0.0),
            ),
        )
        load_score = float(cell_load_score(target_cell))
        queue_bytes = int(getattr(target_cell, "total_queue_bytes", 0) or 0)
        inter_cell_penalty = float(
            getattr(target_cell, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
        )
        energy_intent = goal_prefers_energy(policy_snapshot)
        latency_guard = goal_prefers_latency(policy_snapshot)
        sleep_threshold = clamp(
            float(getattr(policy_snapshot, "preferred_sleep_threshold", self.micro_sleep_load_threshold) or self.micro_sleep_load_threshold),
            0.02,
            0.80,
        )
        wake_up_threshold = clamp(
            float(getattr(policy_snapshot, "preferred_wake_up_threshold", max(self.micro_sleep_load_threshold + 0.08, 0.35)) or max(self.micro_sleep_load_threshold + 0.08, 0.35)),
            sleep_threshold + 0.05,
            1.25,
        )
        dwell_time_slots = int(
            clamp(float(getattr(policy_snapshot, "preferred_dwell_time_slots", 30) or 30), 1.0, 240.0)
        )
        sleep_score = clamp(
            0.45 * max(0.0, sleep_threshold - load_score) / max(0.05, sleep_threshold)
            + 0.25 * max(0.0, self.max_queue_bytes_for_sleep - queue_bytes) / max(1.0, float(self.max_queue_bytes_for_sleep))
            + 0.15 * max(0.0, 1.0 - inter_cell_penalty)
            + (0.15 if energy_intent else 0.0)
            - (0.25 if latency_guard else 0.0),
            0.0,
            1.0,
        )

        sleep_mode = "active"
        power_cap_dbm = None
        if queue_bytes <= self.max_queue_bytes_for_sleep and sleep_score >= 0.25:
            if load_score <= self.deep_sleep_load_threshold and inter_cell_penalty <= 1.0:
                sleep_mode = "deep_sleep"
                power_cap_dbm = 0.0
            elif load_score <= self.light_sleep_load_threshold:
                sleep_mode = "light_sleep"
                power_cap_dbm = 12.0
            elif load_score <= self.micro_sleep_load_threshold:
                sleep_mode = "micro_sleep"
                power_cap_dbm = 18.0

        action = ConfigAction(
            target_cell_id=str(getattr(target_cell, "cell_id", getattr(observation, "cell_id", "cell_0"))),
            cell_sleep_mode=sleep_mode,
            power_cap_dbm=power_cap_dbm,
            sleep_threshold=(sleep_threshold if not getattr(observation, 'rt_public_projection', False) else None),
            wake_up_threshold=(wake_up_threshold if not getattr(observation, 'rt_public_projection', False) else None),
            dwell_time_slots=(dwell_time_slots if not getattr(observation, 'rt_public_projection', False) else None),
            source_name=self.name,
        )
        analysis_constraint = make_analysis_constraint(
            constraint_type="cell_sleep_analysis",
            source_name=self.name,
            scope="cell",
            target_id=str(getattr(target_cell, "cell_id", getattr(observation, "cell_id", "cell_0"))),
            params={
                "load_score": load_score,
                "queue_bytes": queue_bytes,
                "inter_cell_interference_penalty_db_mean": inter_cell_penalty,
                "sleep_score": float(sleep_score),
                "sleep_threshold": float(sleep_threshold),
                "wake_up_threshold": float(wake_up_threshold),
                "dwell_time_slots": int(dwell_time_slots),
                "recommended_sleep_mode": sleep_mode,
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
