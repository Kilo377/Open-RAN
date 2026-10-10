"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: e2_like.py
Description: This file defines an E2-like interface adapter. It exposes
current observations together with Non-RT policy snapshots to Near-RT
xApps and retrieves executable control actions.
"""

from models.action import ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.xapp_base import XAppBase


class E2LikeInterface:
    """
    Provides a simple E2-like interaction path for Near-RT xApps.
    INPUT
    xapp: The Near-RT xApp instance.
    OUTPUT
    An interface adapter that converts observation and policy to action.
    EXAMPLE OUTPUT
    interface = E2LikeInterface(xapp)
    action = interface.send_context(observation, policy_snapshot)
    """

    def __init__(self, xapp: XAppBase) -> None:
        """
        Initializes the E2-like interface with one xApp.
        INPUT
        xapp: The Near-RT xApp instance.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.xapp = xapp
        """
        self.xapp = xapp

    def send_context(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        """
        Sends one observation and one policy snapshot to the xApp and gets
        one executable action back.
        INPUT
        observation: The aggregated current observation.
        policy_snapshot: The current Non-RT policy snapshot.
        OUTPUT
        One ControlDecision object from the xApp.
        EXAMPLE OUTPUT
        ControlDecision(...)
        """
        return self.xapp.run(observation, policy_snapshot)
