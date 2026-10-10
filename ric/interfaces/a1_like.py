"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: a1_like.py
Description: This file defines an A1-like interface adapter. It exposes
long-timescale observations to Non-RT rApps and retrieves policy updates
instead of executable control actions.
"""

from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate
from ric.non_rt.rapp_base import RAppBase


class A1LikeInterface:
    """
    Provides a simple A1-like interaction path for Non-RT rApps.
    INPUT
    rapp: The Non-RT rApp instance.
    OUTPUT
    An interface adapter that converts observation to policy updates.
    EXAMPLE OUTPUT
    interface = A1LikeInterface(rapp)
    policy_update = interface.send_observation(observation)
    """

    def __init__(self, rapp: RAppBase) -> None:
        """
        Initializes the A1-like interface with one rApp.
        INPUT
        rapp: The Non-RT rApp instance.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.rapp = rapp
        """
        self.rapp = rapp

    def send_observation(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        """
        Sends one observation to the rApp and gets one policy update back.
        INPUT
        observation: The aggregated current observation.
        OUTPUT
        One PolicyUpdate object from the rApp.
        EXAMPLE OUTPUT
        PolicyUpdate(
            preferred_codebook_id=2,
            preferred_scheduler_mode="slice_aware",
            source_name="slicing_rapp"
        )
        """
        return self.rapp.run(observation, policy_snapshot=policy_snapshot)
