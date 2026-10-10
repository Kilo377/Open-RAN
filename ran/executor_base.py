"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: executor_base.py
Description: This file defines the abstract base class of RAN action
executors. All executor implementations in this project should inherit
from this base class and implement the apply_action method.
"""

from abc import ABC, abstractmethod

from models.action import Action
from state.global_state import GlobalState


class ExecutorBase(ABC):
    """
    Defines the abstract interface of one RAN action executor.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A reusable executor object.
    EXAMPLE OUTPUT
    executor = SomeExecutor()
    executor.apply_action(state, action)
    """

    @abstractmethod
    def apply_action(self, state: GlobalState, action: Action) -> list[str]:
        """
        Applies one executable action to the current global state.
        INPUT
        state: The global simulation state.
        action: The executable action object.
        OUTPUT
        A list of resolved target cell identifiers. The state is updated in place.
        EXAMPLE OUTPUT
        The cell configuration or scheduler mode is updated.
        """
        raise NotImplementedError
