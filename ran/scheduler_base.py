"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: scheduler_base.py
Description: This file defines the abstract base class of MAC-layer
schedulers. All scheduler implementations in this project should inherit
from this base class and implement the run method.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class SchedulerBase(ABC):
    """
    Defines the abstract interface of one scheduler.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A reusable scheduler object.
    EXAMPLE OUTPUT
    scheduler = SomeScheduler()
    scheduled_ue_ids = scheduler.run(state)
    """

    @abstractmethod
    def run(self, state: Any) -> List[str]:
        """
        Runs one scheduling decision for the current slot.
        INPUT
        state: The global simulation state.
        OUTPUT
        A list containing the scheduled UE identifiers for the current slot.
        EXAMPLE OUTPUT
        ["ue_1"]
        """
        raise NotImplementedError
