"""Spectrum and interference-coordination xApp."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    cell_interference_score,
    clamp,
    dominant_slice_id,
    goal_prefers_interference_control,
    make_analysis_constraint,
    select_busiest_cell,
)


class SpectrumOptimizerXApp(XAppBase):
    def __init__(
        self,
        name: str = "spectrum_optimizer_xapp",
        priority: int = 14,
        utilization_threshold: float = 0.65,
        moderate_utilization_threshold: float = 0.45,
        interference_threshold_db: float = 2.5,
        slice_pressure_threshold: float = 0.40,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "scheduler_policy_hint",
                "slice_weight_adjustment",
                "power_offset_db",
                "muting_ratio",
                "coordination_threshold",
            ],
        )
        self.utilization_threshold = float(utilization_threshold)
        self.moderate_utilization_threshold = min(
            float(self.utilization_threshold),
            float(moderate_utilization_threshold),
        )
        self.interference_threshold_db = float(interference_threshold_db)
        self.slice_pressure_threshold = float(slice_pressure_threshold)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        target_cell = select_busiest_cell(observation)
        empty_action = ConfigAction(source_name=self.name)
        if target_cell is None:
            return ControlDecision(config_action=empty_action)

        utilization = float(getattr(target_cell, "resource_grid_utilization", 0.0) or 0.0)
        interference_penalty_db = cell_interference_score(target_cell)
        dominant_slice, dominant_share = dominant_slice_id(
            observation,
            target_cell_id=str(target_cell.cell_id),
        )
        slice_weights = dict(getattr(policy_snapshot, "preferred_slice_weight_adjustment", {}) or {})
        if len(slice_weights) == 0:
            slice_weights = {"urllc": 1.0, "embb": 1.0, "mmtc": 1.0}

        if dominant_slice in slice_weights:
            slice_weights[dominant_slice] = min(
                1.45,
                float(slice_weights[dominant_slice]) + 0.35 * dominant_share,
            )
        interference_intent = goal_prefers_interference_control(policy_snapshot)
        coordination_pressure = clamp(
            0.42 * (interference_penalty_db / max(0.1, self.interference_threshold_db))
            + 0.28 * utilization
            + 0.20 * dominant_share
            + (0.10 if interference_intent else 0.0),
            0.0,
            1.5,
        )

        should_emit_moderate_action = (
            utilization >= self.moderate_utilization_threshold
            or dominant_share >= self.slice_pressure_threshold
            or interference_penalty_db >= max(1.5, self.interference_threshold_db * 0.7)
        )

        if utilization < self.utilization_threshold and not should_emit_moderate_action:
            return ControlDecision(
                config_action=empty_action,
                policy_constraints=[
                    make_analysis_constraint(
                        constraint_type="spectrum_optimizer_analysis",
                        source_name=self.name,
                        scope="cell",
                        target_id=str(target_cell.cell_id),
                        params={
                            "target_cell_id": str(target_cell.cell_id),
                            "resource_packing_bias": "relaxed",
                            "mu_grouping_bias": "off",
                            "dominant_slice_id": dominant_slice,
                            "dominant_slice_share": float(dominant_share),
                            "optimizer_mode": "idle",
                            "coordination_pressure": float(coordination_pressure),
                        },
                    )
                ],
            )

        optimizer_mode = "dense"
        if interference_penalty_db >= self.interference_threshold_db:
            scheduler_policy_hint = "balanced"
            resource_packing_bias = "interference_aware"
            mu_grouping_bias = "conservative"
            optimizer_mode = "interference_aware"
            power_offset_db = -2.5 * clamp(coordination_pressure, 0.0, 1.0)
            muting_ratio = clamp(0.08 + 0.22 * coordination_pressure, 0.0, 0.35)
        elif utilization < self.utilization_threshold:
            scheduler_policy_hint = "slice_bias" if dominant_share >= self.slice_pressure_threshold else "balanced"
            if dominant_slice in slice_weights:
                slice_weights[dominant_slice] = min(
                    1.25,
                    float(slice_weights[dominant_slice]) + 0.15 * dominant_share,
                )
            resource_packing_bias = "moderate_dense"
            mu_grouping_bias = "guarded"
            optimizer_mode = "moderate_pressure"
            power_offset_db = -0.8 * clamp(coordination_pressure, 0.0, 1.0)
            muting_ratio = clamp(0.03 + 0.08 * coordination_pressure, 0.0, 0.18)
        else:
            scheduler_policy_hint = (
                str(getattr(policy_snapshot, "preferred_scheduler_policy_hint", "slice_bias") or "slice_bias")
            )
            if scheduler_policy_hint not in {"slice_bias", "balanced"}:
                scheduler_policy_hint = "slice_bias"
            resource_packing_bias = "throughput_dense"
            mu_grouping_bias = "aggressive"
            power_offset_db = 0.5 if interference_penalty_db < 1.0 else -1.0 * clamp(coordination_pressure, 0.0, 1.0)
            muting_ratio = clamp(0.02 * coordination_pressure, 0.0, 0.12)

        action = ConfigAction(
            target_cell_id=str(target_cell.cell_id),
            scheduler_policy_hint=scheduler_policy_hint,
            slice_weight_adjustment=slice_weights,
            power_offset_db=(power_offset_db if not getattr(observation, 'rt_public_projection', False) else None),
            muting_ratio=(muting_ratio if not getattr(observation, 'rt_public_projection', False) else None),
            coordination_threshold=(clamp(
                float(getattr(policy_snapshot, "preferred_coordination_threshold", self.interference_threshold_db) or self.interference_threshold_db),
                0.5,
                8.0,
            ) if not getattr(observation, 'rt_public_projection', False) else None),
            source_name=self.name,
        )
        analysis_constraint = make_analysis_constraint(
            constraint_type="spectrum_optimizer_analysis",
            source_name=self.name,
            scope="cell",
            target_id=str(target_cell.cell_id),
            params={
                "target_cell_id": str(target_cell.cell_id),
                "resource_packing_bias": resource_packing_bias,
                "mu_grouping_bias": mu_grouping_bias,
                "dominant_slice_id": dominant_slice,
                "dominant_slice_share": float(dominant_share),
                "utilization": float(utilization),
                "interference_penalty_db": float(interference_penalty_db),
                "optimizer_mode": str(optimizer_mode),
                "coordination_pressure": float(coordination_pressure),
                "power_offset_db": float(power_offset_db),
                "muting_ratio": float(muting_ratio),
                "coordination_threshold": float(action.coordination_threshold or 0.0),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
