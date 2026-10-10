"""models/action.py

High-level control-plane data models.

This project separates control semantics into three categories:
- ConfigAction: persistent configuration (target-state) that may take effect
  starting from a given slot.
- CommandAction: one-shot command intended to be consumed once by the RAN loop.
- PolicyConstraint: non-RT constraints/hints that can be consumed by scheduler,
  executor, and link adaptation.

ControlDecision bundles these three outputs for each slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from models.oran_lifecycle import OranLifecycleResources


@dataclass
class ControlMetadata:
    """Common metadata attached to all control-plane items."""

    source: str = "unknown"
    issued_at_slot: Optional[int] = None
    reason: str = ""
    priority: Optional[int] = None
    confidence: Optional[float] = None
    decision_id: Optional[str] = None


@dataclass
class FieldArbitrationResult:
    field_name: str
    value: Any
    source: str
    priority: int


@dataclass
class DroppedControlItem:
    kind: str
    source: str
    reason: str
    field_name: Optional[str] = None
    value: Any = None


@dataclass
class ControlArbitrationTrace:
    received_config_actions: list["ConfigAction"] = field(default_factory=list)
    selected_config_action_by_field: dict[str, FieldArbitrationResult] = field(default_factory=dict)
    merged_config_actions: list["ConfigAction"] = field(default_factory=list)
    accepted_commands: list["CommandAction"] = field(default_factory=list)
    active_constraints: list["PolicyConstraint"] = field(default_factory=list)
    dropped_items_with_reason: list[DroppedControlItem] = field(default_factory=list)


@dataclass
class ConfigAction:
    """Persistent configuration action (target-state).

    Fields are applied by the executor to the simulator state. By default, the
    configuration is effective immediately. If effective_from_slot is set, the
    executor may delay applying the config until state.slot_id reaches it.
    """

    # Backward-compatible direct controls.
    codebook_id: Optional[int] = None
    tx_power_dbm: Optional[float] = None
    beam_profile_id: Optional[int] = None
    scheduler_mode: Optional[str] = None

    # Capability-limited control surface.
    scheduler_policy_hint: Optional[str] = None
    latency_bias: Optional[float] = None
    slice_weight_adjustment: dict[str, float] = field(default_factory=dict)
    beam_candidate_preference: Optional[str] = None
    power_cap_dbm: Optional[float] = None
    cell_sleep_mode: Optional[str] = None
    traffic_steering_preference: Optional[str] = None
    steering_aggressiveness: Optional[float] = None
    load_threshold: Optional[float] = None
    cio_bias_db: Optional[float] = None
    mobility_bias_db: Optional[float] = None
    handover_cost_db: Optional[float] = None
    handover_interruption_slots: Optional[int] = None
    steering_hysteresis_db: Optional[float] = None
    time_to_trigger_ms: Optional[float] = None
    mobility_margin_db: Optional[float] = None
    measurement_event_trigger: Optional[str] = None
    measurement_event_offset_db: Optional[float] = None
    prb_weight: Optional[float] = None
    priority_coefficient: Optional[float] = None
    delay_threshold_ms: Optional[float] = None
    power_offset_db: Optional[float] = None
    muting_ratio: Optional[float] = None
    coordination_threshold: Optional[float] = None
    sleep_threshold: Optional[float] = None
    wake_up_threshold: Optional[float] = None
    dwell_time_slots: Optional[int] = None

    apply_scope: str = "cell"
    target_cell_id: Optional[str] = None
    control_request_id: Optional[str] = None
    effective_from_slot: Optional[int] = None
    effective_until_slot: Optional[int] = None
    source_name: str = "unknown"

    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def __post_init__(self) -> None:
        if self.metadata.source == "unknown" and self.source_name != "unknown":
            self.metadata.source = self.source_name
        if self.source_name == "unknown" and self.metadata.source != "unknown":
            self.source_name = self.metadata.source


@dataclass
class CommandAction:
    """One-shot command action.

    Commands are intended to be consumed once by a command-consumption stage
    within the simulator slot loop.
    """

    command_type: str
    target_ue_id: Optional[str] = None
    delta_priority: float = 0.0

    valid_from_slot: Optional[int] = None
    valid_until_slot: Optional[int] = None
    control_request_id: Optional[str] = None
    source_name: str = "unknown"

    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def __post_init__(self) -> None:
        if self.metadata.source == "unknown" and self.source_name != "unknown":
            self.metadata.source = self.source_name
        if self.source_name == "unknown" and self.metadata.source != "unknown":
            self.source_name = self.metadata.source


@dataclass
class PolicyConstraint:
    """Policy-level constraint/hint (non-RT semantics).

    This structure is intentionally generic. Consumers interpret params based
    on constraint_type.
    """

    constraint_type: str
    scope: str = "cell"
    target_id: Optional[str] = None
    params: dict[str, object] = field(default_factory=dict)

    valid_from_slot: Optional[int] = None
    valid_until_slot: Optional[int] = None
    control_request_id: Optional[str] = None
    source_name: str = "unknown"

    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def __post_init__(self) -> None:
        if self.metadata.source == "unknown" and self.source_name != "unknown":
            self.metadata.source = self.source_name
        if self.source_name == "unknown" and self.metadata.source != "unknown":
            self.source_name = self.metadata.source


@dataclass
class ControlDecision:
    """Full control-plane output for one slot."""

    config_action: ConfigAction = field(default_factory=ConfigAction)
    config_actions: list[ConfigAction] = field(default_factory=list)
    commands: list[CommandAction] = field(default_factory=list)
    policy_constraints: list[PolicyConstraint] = field(default_factory=list)

    metadata: ControlMetadata = field(default_factory=ControlMetadata)
    trace: ControlArbitrationTrace = field(default_factory=ControlArbitrationTrace)
    oran_lifecycle_resources: OranLifecycleResources = field(default_factory=OranLifecycleResources)

    def __post_init__(self) -> None:
        if len(self.config_actions) == 0:
            self.config_actions = [self.config_action]
        else:
            self.config_action = self.config_actions[0]


# Backward-compatible alias: existing code uses Action as the config action.
Action = ConfigAction
