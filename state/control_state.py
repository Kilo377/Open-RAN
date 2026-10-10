"""state/control_state.py

Defines the control-plane runtime state stored inside GlobalState.

This separates persistent configuration, one-shot commands, and policy
constraints so that the simulator can model control semantics more explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.action import (
    CommandAction,
    ConfigAction,
    ControlArbitrationTrace,
    ControlMetadata,
    PolicyConstraint,
)
from models.oran_lifecycle import OranLifecycleResources


@dataclass
class ControlState:
    """Stores control-plane state for the current slot."""

    active_config: ConfigAction = field(default_factory=ConfigAction)
    active_config_actions: list[ConfigAction] = field(default_factory=list)
    pending_commands: list[CommandAction] = field(default_factory=list)
    active_policy_constraints: list[PolicyConstraint] = field(default_factory=list)

    decision_metadata: ControlMetadata = field(default_factory=ControlMetadata)
    arbitration_trace: ControlArbitrationTrace = field(default_factory=ControlArbitrationTrace)
    input_observation_trace: dict[str, object] = field(default_factory=dict)
    oran_lifecycle_resources: OranLifecycleResources = field(default_factory=OranLifecycleResources)

    # Actuation pipeline outcomes (delay/reject) for this slot.
    actuation_outcomes: list[dict[str, object]] = field(default_factory=list)
