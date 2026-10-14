"""Massive-MIMO beamforming xApp with coverage/capacity tuning."""

from __future__ import annotations

from models.action import ConfigAction, ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    beam_misalignment_risk_score,
    clamp,
    iter_ue_observations,
    make_analysis_constraint,
    safe_mean,
    select_beam_stressed_cell,
)


class MassiveMIMOBeamformerXApp(XAppBase):
    def __init__(
        self,
        name: str = "massive_mimo_beamformer_xapp",
        priority: int = 17,
        beam_risk_threshold: float = 0.50,
        coverage_sinr_threshold_db: float = 7.0,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "beam_candidate_preference",
                "power_cap_dbm",
                "power_offset_db",
                "coordination_threshold",
            ],
        )
        self.beam_risk_threshold = float(beam_risk_threshold)
        self.coverage_sinr_threshold_db = float(coverage_sinr_threshold_db)

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        target_cell = select_beam_stressed_cell(observation)
        empty_action = ConfigAction(source_name=self.name)
        if target_cell is None:
            return ControlDecision(config_action=empty_action)

        serving_ues = [
            ue
            for ue in iter_ue_observations(observation)
            if str(getattr(ue, "serving_cell_id", "")) == str(target_cell.cell_id)
        ]
        mean_beam_risk = safe_mean(
            [beam_misalignment_risk_score(ue) for ue in serving_ues],
            default=0.0,
        )
        mean_post_eq_sinr_db = safe_mean(
            [
                float(
                    getattr(ue, "post_eq_sinr_db", getattr(ue, "sinr_db", 0.0))
                    or 0.0
                )
                for ue in serving_ues
            ],
            default=float(getattr(observation, "avg_sinr_db", 0.0) or 0.0),
        )
        mean_precoder_match = safe_mean(
            [float(getattr(ue, "precoder_match_score", 0.5) or 0.5) for ue in serving_ues],
            default=0.5,
        )
        preferred_beam = str(
            getattr(policy_snapshot, "preferred_beam_candidate_preference", "balanced")
            or "balanced"
        )
        preferred_power_cap_dbm = float(
            getattr(policy_snapshot, "preferred_power_cap_dbm", 30.0) or 30.0
        )

        if mean_beam_risk >= self.beam_risk_threshold or mean_post_eq_sinr_db <= self.coverage_sinr_threshold_db:
            beam_preference = preferred_beam if preferred_beam != "capacity" else "coverage"
            if beam_preference == "balanced":
                beam_preference = "coverage"
            power_cap_dbm = clamp(preferred_power_cap_dbm + 0.8 * mean_beam_risk, 28.0, 32.0)
            rank_cap = 1
            power_offset_db = clamp(1.5 * mean_beam_risk, 0.0, 2.0)
            coordination_threshold = clamp(self.coverage_sinr_threshold_db + 1.0, 4.0, 14.0)
        elif mean_precoder_match >= 0.70 and mean_post_eq_sinr_db >= 11.0:
            beam_preference = "capacity"
            power_cap_dbm = clamp(preferred_power_cap_dbm, 27.0, 30.0)
            rank_cap = 2
            power_offset_db = clamp(-0.5 * mean_precoder_match, -1.5, 0.5)
            coordination_threshold = clamp(self.coverage_sinr_threshold_db - 0.5, 3.0, 12.0)
        else:
            beam_preference = "balanced"
            power_cap_dbm = clamp(preferred_power_cap_dbm, 27.5, 31.0)
            rank_cap = 1
            power_offset_db = 0.0
            coordination_threshold = clamp(self.coverage_sinr_threshold_db, 3.0, 12.0)

        action = ConfigAction(
            target_cell_id=str(target_cell.cell_id),
            beam_candidate_preference=beam_preference,
            power_cap_dbm=power_cap_dbm,
            power_offset_db=(power_offset_db if not getattr(observation, 'rt_public_projection', False) else None),
            coordination_threshold=(coordination_threshold if not getattr(observation, 'rt_public_projection', False) else None),
            source_name=self.name,
        )
        analysis_constraint = make_analysis_constraint(
            constraint_type="massive_mimo_beamforming_analysis",
            source_name=self.name,
            scope="cell",
            target_id=str(target_cell.cell_id),
            params={
                "target_cell_id": str(target_cell.cell_id),
                "mean_beam_misalignment_risk": float(mean_beam_risk),
                "mean_post_eq_sinr_db": float(mean_post_eq_sinr_db),
                "mean_precoder_match_score": float(mean_precoder_match),
                "rank_cap": int(rank_cap),
                "mu_pairing_bias": "aggressive" if rank_cap > 1 else "conservative",
                "power_offset_db": float(power_offset_db),
                "coordination_threshold": float(coordination_threshold),
            },
        )
        return ControlDecision(
            config_action=action,
            policy_constraints=[analysis_constraint],
        )
