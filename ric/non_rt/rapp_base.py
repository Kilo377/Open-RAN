"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: rapp_base.py
Description: This file defines the abstract base class of Non-RT rApps.
A rApp reads long-timescale observations and outputs policy updates or
recommendations instead of executable RAN control actions.
"""

from abc import ABC, abstractmethod

from models.observation import Observation
from models.policy import PolicySnapshot, PolicyUpdate


class RAppBase(ABC):
    """
    Defines the abstract interface of one Non-RT rApp.
    INPUT
    name: The rApp name.
    OUTPUT
    A reusable rApp object that can generate policy updates.
    EXAMPLE OUTPUT
    rapp = SomeRApp(name="codebook_rapp")
    policy_update = rapp.run(observation)
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the rApp with a given name.
        INPUT
        name: The rApp name.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.name = "codebook_rapp"
        """
        self.name = name

    @abstractmethod
    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot | None = None,
    ) -> PolicyUpdate:
        """
        Generates one policy update from the current observation.
        INPUT
        observation: The aggregated current observation.
        policy_snapshot: The current aggregated Non-RT policy snapshot that may
        include outputs from earlier rApps within the same epoch.
        OUTPUT
        One PolicyUpdate object.
        EXAMPLE OUTPUT
        PolicyUpdate(
            preferred_codebook_id=2,
            preferred_tx_power_dbm=31.0,
            source_name="codebook_rapp"
        )
        """
        raise NotImplementedError
