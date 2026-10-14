"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: xapp_base.py
Description: This file defines the abstract base class of Near-RT xApps.
An xApp reads real-time observations together with Non-RT policy snapshots
and outputs executable RAN control actions. Each xApp must also declare
its priority and the action fields it is allowed to control.
"""

from abc import ABC, abstractmethod
from typing import List

from models.action import ControlDecision
from models.observation import Observation
from models.policy import PolicySnapshot


class XAppBase(ABC):
    """
    Defines the abstract interface of one Near-RT xApp.
    INPUT
    name: The xApp name.
    priority: The xApp priority. A larger value means higher priority.
    controlled_fields: The list of action fields this xApp is allowed to control.
    OUTPUT
    A reusable xApp object that can generate executable actions.
    EXAMPLE OUTPUT
    xapp = SomeXApp(
        name="beam_xapp",
        priority=20,
        controlled_fields=["codebook_id", "tx_power_dbm", "beam_profile_id"]
    )
    action = xapp.run(observation, policy_snapshot)
    """

    def __init__(
        self,
        name: str,
        priority: int,
        controlled_fields: List[str],
        contract=None,
    ) -> None:
        """
        Initializes the xApp with a given name, priority, and field ownership.
        INPUT
        name: The xApp name.
        priority: The xApp priority. A larger value means higher priority.
        controlled_fields: The list of action fields the xApp is allowed to control.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.name = "beam_xapp"
        self.priority = 20
        self.controlled_fields = ["codebook_id", "tx_power_dbm", "beam_profile_id"]
        """
        self.name = name
        self.priority = priority
        self.controlled_fields = controlled_fields
        from ric.xapp_contract import contract_for
        self.contract = contract or contract_for(controlled_fields,name)

    @abstractmethod
    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        """
        Generates one executable action from the current observation and
        policy snapshot.
        INPUT
        observation: The aggregated current observation.
        policy_snapshot: The current Non-RT policy snapshot.
        OUTPUT
        One ControlDecision object.
        EXAMPLE OUTPUT
        ControlDecision(...)
        """
        raise NotImplementedError
