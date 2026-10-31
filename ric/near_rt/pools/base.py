"""Shared lifecycle/contract adapter; algorithms belong in their owner's pool."""
from collections import Counter
from copy import deepcopy
import math

from ric.near_rt.xapp_base import XAppBase
from ric.xapp_contract import XAppContract


class PoolXApp(XAppBase):
    library_instance = rule_based_v1 = True
    selection_mode = 'exclusive'
    version = '1.0.0'
    required_kpis = ()
    writable_actions = ()
    commands = ()
    priority = 20
    min_action_ms = 20
    description = ''

    def __init__(self, cell_ids, name=None, slot_duration_ms=.5, parameters=None):
        if not cell_ids or len(set(cell_ids)) != len(cell_ids):
            raise ValueError('Explicit unique allowed cells required')
        if not math.isfinite(slot_duration_ms) or slot_duration_ms <= 0:
            raise ValueError('Positive slot duration required')
        self.allowed_cells = tuple(cell_ids)
        self.dt = slot_duration_ms
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError('xApp parameters must be a mapping')
        self.parameters = deepcopy(parameters or {})
        self.validate_parameters(self.parameters)
        self.kind = self.category_id
        self.initial_bindings = {}
        self.active_cells, self.tokens = set(), {}
        self.last_decisions, self.statistics, self.decision_seconds = [], Counter(), 0.
        contract = XAppContract(tuple(self.required_kpis), frozenset(self.writable_actions),
            commands=frozenset(self.commands), cell_ids=self.allowed_cells,
            max_age_slots=math.ceil(100 / self.dt))
        super().__init__(name or self.implementation_id, self.priority, list(self.writable_actions), contract)

    def validate_parameters(self, parameters):
        if not isinstance(parameters, dict):raise ValueError('xApp parameters must be a mapping')

    def runtime_contract(self, policy):
        return self.contract

    def select(self, cells, tokens):
        self.active_cells, self.tokens = set(cells), dict(tokens)

    def reset(self):
        self.last_decisions = []

    def identity(self, cell_id):
        return dict(instance_id=self.name, implementation_id=self.implementation_id,
            category_id=self.category_id, implementation_version=self.version,
            pool=getattr(self, 'xapp_pool', None), cell_id=cell_id,
            selection_version=self.tokens.get(cell_id, 0))

    def descriptor(self):
        return dict(instance_id=self.name, implementation_id=self.implementation_id,
            category_id=self.category_id, description=self.description,
            pool=getattr(self, 'xapp_pool', None), allowed_cell_ids=list(self.allowed_cells),
            active_cell_ids=sorted(self.active_cells), required_kpis=list(self.required_kpis),
            writable_actions=list(self.controlled_fields), commands=list(self.commands),
            parameters=deepcopy(self.parameters), min_action_ms=self.min_action_ms,
            selection_mode=self.selection_mode)
