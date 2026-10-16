"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: controller_registry.py
Description: This file defines the controller registry used by the
simulator. It stores multiple Near-RT xApps and multiple Non-RT rApps and
provides a clean way to inject different controller implementations into
the simulator.
"""

from typing import List

from ric.near_rt.xapp_base import XAppBase
from ric.non_rt.rapp_base import RAppBase


class ControllerRegistry:
    """
    Stores controller instances used by the simulator.
    INPUT
    No explicit input during initialization.
    OUTPUT
    A controller registry object holding xApp and rApp lists.
    EXAMPLE OUTPUT
    registry = ControllerRegistry()
    registry.register_xapp(xapp)
    registry.register_rapp(rapp)
    """

    def __init__(self) -> None:
        """
        Initializes the empty controller registry.
        INPUT
        No explicit input.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.xapps = []
        self.rapps = []
        """
        self.xapps: List[XAppBase] = []
        self.rapps: List[RAppBase] = []
        self.active_intent_manager = None

    def register_xapp(self, xapp: XAppBase) -> None:
        """
        Registers one Near-RT xApp instance.
        INPUT
        xapp: The xApp instance to register.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        registry.register_xapp(BeamXApp())
        """
        if any(app.name==xapp.name for app in self.xapps):raise ValueError('Duplicate xApp instance name: '+xapp.name)
        self.xapps.append(xapp)

    def register_rapp(self, rapp: RAppBase) -> None:
        """
        Registers one Non-RT rApp instance.
        INPUT
        rapp: The rApp instance to register.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        registry.register_rapp(CodebookRApp())
        """
        self.rapps.append(rapp)
