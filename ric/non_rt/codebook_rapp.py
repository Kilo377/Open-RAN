"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: codebook_rapp.py
Description: This file implements a simple Non-RT codebook rApp. The rApp
analyzes long-timescale system metrics and generates recommended beam,
power, and codebook policies for Near-RT xApps.
"""

from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
from ric.non_rt.rapp_base import RAppBase


class CodebookRApp(RAppBase):
    """
    Implements a simple Non-RT codebook recommendation rApp.
    """

    def __init__(
        self,
        name: str = "codebook_rapp",
        bler_threshold: float = 0.12,
        throughput_threshold_mbps: float = 7.0,
    ) -> None:
        super().__init__(name=name)
        self.bler_threshold = bler_threshold
        self.throughput_threshold_mbps = throughput_threshold_mbps

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        avg_bler = getattr(observation, "avg_bler", None)
        has_tx = bool(getattr(observation, "num_scheduled_ues", 0) > 0)
        has_traffic = bool(getattr(observation, "total_queue_bytes", 0) > 0)

        bler_bad = (avg_bler is not None) and (avg_bler > self.bler_threshold)
        throughput_low = (
            has_traffic
            and (observation.total_throughput_mbps < self.throughput_threshold_mbps)
        )

        if has_tx and (bler_bad or throughput_low):
            return PolicyUpdate(
                preferred_beam_candidate_preference="coverage",
                preferred_power_cap_dbm=31.0,
                source_name=self.name,
            )

        return PolicyUpdate(source_name=self.name)
