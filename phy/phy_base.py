"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: phy_base.py
Description: This file defines the abstract base class of PHY adapters.
All PHY implementations in this project should inherit from this base
class and implement the run_slot method.
"""

from abc import ABC, abstractmethod

from models.phy_input import PhyInput
from models.phy_result import PhyResult


class PhyBase(ABC):
    """
    Defines the abstract interface of one PHY adapter.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A reusable PHY adapter object.
    EXAMPLE OUTPUT
    phy = SomePhyAdapter()
    phy_result = phy.run_slot(phy_input)
    """

    @abstractmethod
    def run_slot(self, phy_input: PhyInput) -> PhyResult:
        """
        Executes one PHY slot update using the given PHY input.
        INPUT
        phy_input: The PHY input object for the current slot.
        OUTPUT
        One PhyResult object for the current slot.
        EXAMPLE OUTPUT
        PhyResult(slot_id=1, ue_results=[...])
        """
        raise NotImplementedError