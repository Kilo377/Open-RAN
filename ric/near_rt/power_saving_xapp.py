"""Energy-saving xApp with bounded power and wake-up tuning."""

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
    make_analysis_constraint,
    select_least_loaded_cell,
)


class PowerSavingXApp(XAppBase):
    def __init__(
        self,
        name: str = "power_saving_xapp",
        priority: int = 8,
        low_load_threshold: float = 0.35,
        moderate_load_threshold: float = 0.72,
        queue_threshold_bytes: int = 3000,
        moderate_queue_threshold_bytes: int = 18000,
        min_power_cap_dbm: float = 24.0,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "power_cap_dbm",
                "beam_candidate_preference",
                "sleep_threshold",
                "wake_up_threshold",
                "dwell_time_slots",
            ],
        )
        self.low_load_threshold = float(low_load_threshold)
        self.moderate_load_threshold = max(
            float(self.low_load_threshold),
            float(moderate_load_threshold),
        )
        self.queue_threshold_bytes = int(queue_threshold_bytes)
        self.moderate_queue_threshold_bytes = max(
            int(self.queue_threshold_bytes),
            int(moderate_queue_threshold_bytes),
        )
        self.min_power_cap_dbm = float(min_power_cap_dbm)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        target_cell = select_least_loaded_cell(observation)
        empty_action = ConfigAction(source_name=self.name)
        if target_cell is None:
            return ControlDecision(config_action=empty_action)

        target_load = cell_load_score(target_cell)
        base_power_cap_dbm = float(
            getattr(policy_snapshot, "preferred_power_cap_dbm", 29.5) or 29.5
        )
        preferred_beam = str(
            getattr(policy_snapshot, "preferred_beam_candidate_preference", "balanced")
            or "balanced"
        )
        energy_intent = goal_prefers_energy(policy_snapshot)
        latency_guard = goal_prefers_latency(policy_snapshot)
        queue_bytes = int(getattr(target_cell, "total_queue_bytes", 0) or 0)
        energy_headroom = clamp(
            0.55 * max(0.0, self.moderate_load_threshold - target_load)
            + 0.25 * max(0.0, self.moderate_queue_threshold_bytes - queue_bytes)
            / max(1.0, float(self.moderate_queue_threshold_bytes))
            + (0.15 if energy_intent else 0.0)
            - (0.20 if latency_guard else 0.0),
            0.0,
            1.0,
        )
        sleep_threshold = clamp(
            float(getattr(policy_snapshot, "preferred_sleep_threshold", self.low_load_threshold) or self.low_load_threshold),
            0.02,
            0.80,
        )
        wake_up_threshold = clamp(
            float(getattr(policy_snapshot, "preferred_wake_up_threshold", self.moderate_load_threshold) or self.moderate_load_threshold),
            sleep_threshold + 0.05,
            1.25,
        )
        dwell_time_slots = int(
            clamp(float(getattr(policy_snapshot, "preferred_dwell_time_slots", 20) or 20), 1.0, 200.0)
        )

        action = empty_action
        energy_mode = "normal"
        if (
            target_load <= sleep_threshold
            and queue_bytes <= self.queue_threshold_bytes
        ):
            action = ConfigAction(
                target_cell_id=str(target_cell.cell_id),
                power_cap_dbm=clamp(
                    base_power_cap_dbm - 7.5 * energy_headroom,
                    self.min_power_cap_dbm,
                    base_power_cap_dbm,
                ),
                beam_candidate_preference=(
                    preferred_beam if preferred_beam != "coverage" else "balanced"
                ),
                sleep_threshold=(sleep_threshold if not getattr(observation, 'rt_public_projection', False) else None),
                wake_up_threshold=(wake_up_threshold if not getattr(observation, 'rt_public_projection', False) else None),
                dwell_time_slots=(dwell_time_slots if not getattr(observation, 'rt_public_projection', False) else None),
                source_name=self.name,
            )
            energy_mode = "saving"
        elif (
            target_load <= self.moderate_load_threshold
            and int(getattr(target_cell, "total_queue_bytes", 0) or 0)
            <= self.moderate_queue_threshold_bytes
        ):
            moderate_headroom = max(0.0, self.moderate_load_threshold - target_load)
            action = ConfigAction(
                target_cell_id=str(target_cell.cell_id),
                power_cap_dbm=clamp(
                    base_power_cap_dbm - (1.0 + 3.0 * max(moderate_headroom, energy_headroom)),
                    min(self.min_power_cap_dbm + 1.0, base_power_cap_dbm),
                    base_power_cap_dbm,
                ),
                beam_candidate_preference="balanced",
                sleep_threshold=(sleep_threshold if not getattr(observation, 'rt_public_projection', False) else None),
                wake_up_threshold=(wake_up_threshold if not getattr(observation, 'rt_public_projection', False) else None),
                dwell_time_slots=(dwell_time_slots if not getattr(observation, 'rt_public_projection', False) else None),
                source_name=self.name,
            )
            energy_mode = "mild_saving"

        analysis_constraint = make_analysis_constraint(
            constraint_type="power_saving_analysis",
            source_name=self.name,
            scope="cell",
            target_id=str(getattr(target_cell, "cell_id", "cell_0")),
            params={
                "target_cell_id": str(getattr(target_cell, "cell_id", "cell_0")),
                "target_load_score": float(target_load),
                "energy_mode": str(energy_mode),
                "energy_headroom": float(energy_headroom),
                "sleep_threshold": float(sleep_threshold),
                "wake_up_threshold": float(wake_up_threshold),
                "dwell_time_slots": int(dwell_time_slots),
                "recommended_power_cap_dbm": (
                    float(action.power_cap_dbm) if action.power_cap_dbm is not None else None
                ),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
