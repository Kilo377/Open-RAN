"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: slicing_rapp.py
Description: This file implements a simple Non-RT slicing rApp. The rApp
analyzes system-level load and generates scheduler-related policy
recommendations that can later be interpreted by Near-RT xApps.
"""

from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
from ric.non_rt.rapp_base import RAppBase


class SlicingRApp(RAppBase):
    """
    Implements a simple slicing-aware Non-RT rApp.
    """

    def __init__(
        self,
        name: str = "slicing_rapp",
        queue_threshold_bytes: int = 5000,
    ) -> None:
        super().__init__(name=name)
        self.queue_threshold_bytes = queue_threshold_bytes

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        if observation.total_queue_bytes > self.queue_threshold_bytes:
            return PolicyUpdate(
                preferred_scheduler_policy_hint="slice_bias",
                preferred_slice_weight_adjustment={
                    "urllc": 1.35,
                    "embb": 1.05,
                    "mmtc": 0.85,
                },
                source_name=self.name,
            )

        return PolicyUpdate(source_name=self.name)
