"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: beam_xapp.py
Description: This file defines the beam management xApp. The xApp reads
current observation and policy context, then decides whether to update
beam preference and transmit power cap through a limited control surface.
"""

from models.action import ConfigAction, ControlDecision
from ric.near_rt.xapp_base import XAppBase


class BeamXApp(XAppBase):
    """
    Implements one beam control xApp.
    """

    def __init__(
        self,
        name: str,
        priority: int,
        sinr_threshold_db: float,
        queue_threshold_bytes: int,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "beam_candidate_preference",
                "power_cap_dbm",
            ],
        )
        self.sinr_threshold_db = sinr_threshold_db
        self.queue_threshold_bytes = queue_threshold_bytes

    def _build_empty_action(self) -> ConfigAction:
        return ConfigAction(
            beam_candidate_preference=None,
            power_cap_dbm=None,
        )

    def run(
        self,
        observation,
        policy_snapshot,
    ) -> ControlDecision:
        if observation is None:
            return ControlDecision(config_action=self._build_empty_action())

        avg_sinr_db = getattr(observation, "avg_sinr_db", None)
        total_queue_bytes = getattr(observation, "total_queue_bytes", None)

        if avg_sinr_db is None or total_queue_bytes is None:
            return ControlDecision(config_action=self._build_empty_action())

        preferred_beam_candidate_preference = getattr(
            policy_snapshot,
            "preferred_beam_candidate_preference",
            None,
        )
        preferred_power_cap_dbm = getattr(
            policy_snapshot,
            "preferred_power_cap_dbm",
            None,
        )

        if (
            avg_sinr_db < self.sinr_threshold_db
            or total_queue_bytes > self.queue_threshold_bytes
        ):
            return ControlDecision(
                config_action=ConfigAction(
                    beam_candidate_preference=(
                        preferred_beam_candidate_preference
                        if preferred_beam_candidate_preference is not None
                        else "coverage"
                    ),
                    power_cap_dbm=(
                        preferred_power_cap_dbm
                        if preferred_power_cap_dbm is not None
                        else 30.5
                    ),
                    source_name=self.name,
                )
            )

        return ControlDecision(config_action=self._build_empty_action())
