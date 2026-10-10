"""models/oran_lifecycle.py

Lightweight A1/E2-like lifecycle resource models.

These structures intentionally model lifecycle semantics without attempting to
implement the real A1AP/E2AP protocols. They are designed to support
controller orchestration, traceability, and experiment analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class A1PolicyType:
    policy_type_id: str
    name: str
    description: str = ""
    scope: str = "cluster"


@dataclass
class A1PolicyInstance:
    instance_id: str
    policy_type_id: str
    source_name: str
    created_at_slot: Optional[int] = None
    updated_at_slot: Optional[int] = None
    status: str = "active"
    target_xapps: list[str] = field(default_factory=list)
    payload_summary: dict[str, object] = field(default_factory=dict)


@dataclass
class A1PolicyStatus:
    instance_id: str
    status: str = "active"
    reason: str = ""
    updated_at_slot: Optional[int] = None


@dataclass
class A1EIType:
    ei_type_id: str
    name: str
    description: str = ""


@dataclass
class A1EIJob:
    job_id: str
    ei_type_id: str
    consumer_name: str
    status: str = "active"
    created_at_slot: Optional[int] = None
    updated_at_slot: Optional[int] = None
    last_delivery_slot: Optional[int] = None
    payload_summary: dict[str, object] = field(default_factory=dict)


@dataclass
class E2Subscription:
    subscription_id: str
    subscriber_name: str
    scope: str = "cluster"
    metric_filter: str = "full_observation"
    status: str = "active"
    created_at_slot: Optional[int] = None
    updated_at_slot: Optional[int] = None
    target_cell_ids: list[str] = field(default_factory=list)


@dataclass
class E2Indication:
    indication_id: str
    subscription_id: str
    producer_name: str
    slot_id: Optional[int] = None
    produced_at_slot: Optional[int] = None
    delivered_at_slot: Optional[int] = None
    payload_summary: dict[str, object] = field(default_factory=dict)


@dataclass
class E2ControlRequest:
    request_id: str
    source_name: str
    kind: str = "config"
    target_cell_id: Optional[str] = None
    target_cell_ids: list[str] = field(default_factory=list)
    issued_at_slot: Optional[int] = None
    updated_at_slot: Optional[int] = None
    status: str = "issued"
    reason: str = ""
    payload_summary: dict[str, object] = field(default_factory=dict)


@dataclass
class E2ControlOutcome:
    outcome_id: str
    request_id: str
    kind: str = "config"
    source_name: str = "unknown"
    status: str = "pending"
    applied_at_slot: Optional[int] = None
    target_cell_ids: list[str] = field(default_factory=list)
    reason: str = ""
    decision_id: Optional[str] = None


@dataclass
class OranLifecycleResources:
    a1_policy_types: list[A1PolicyType] = field(default_factory=list)
    a1_policy_instances: list[A1PolicyInstance] = field(default_factory=list)
    a1_policy_statuses: list[A1PolicyStatus] = field(default_factory=list)
    a1_ei_types: list[A1EIType] = field(default_factory=list)
    a1_ei_jobs: list[A1EIJob] = field(default_factory=list)
    e2_subscriptions: list[E2Subscription] = field(default_factory=list)
    e2_indications: list[E2Indication] = field(default_factory=list)
    e2_control_requests: list[E2ControlRequest] = field(default_factory=list)
    e2_control_outcomes: list[E2ControlOutcome] = field(default_factory=list)
