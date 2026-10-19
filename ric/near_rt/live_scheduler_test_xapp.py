"""Small registered xApp for live lifecycle experiments, using delivered KPIs."""
from collections import Counter
import time

from models.action import ConfigAction, ControlDecision
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import iter_ue_observations
from ric.xapp_contract import XAppContract


class LiveSchedulerTestXApp(XAppBase):
    library_instance = rule_based_v1 = True
    category_id = kind = 'live_scheduler_test'
    implementation_id = 'live_scheduler_test:queue_hol'
    version = '1.0.0'
    selection_mode = 'exclusive'

    def __init__(self, cell_ids, name='live_scheduler_test'):
        self.allowed_cells = tuple(cell_ids)
        self.initial_bindings = {}
        self.active_cells, self.tokens = set(), {}
        self.last_decisions, self.statistics, self.decision_seconds = [], Counter(), 0.
        actions = ('scheduler_mode', 'scheduler_policy_hint')
        super().__init__(name, 20, list(actions), XAppContract(
            ('queue_bytes', 'hol_delay_ms'), frozenset(actions),
            cell_ids=self.allowed_cells))

    def validate_parameters(self, parameters):
        if parameters:
            raise ValueError('This registered test xApp has fixed parameters')

    def runtime_contract(self, policy):
        return self.contract

    def select(self, cells, tokens):
        self.active_cells, self.tokens = set(cells), dict(tokens)

    def reset(self):
        self.last_decisions = []

    def identity(self, cell_id):
        return dict(instance_id=self.name, implementation_id=self.implementation_id,
            category_id=self.category_id, objective_id='queue_hol_scheduler',
            implementation_version=self.version, cell_id=cell_id,
            selection_version=self.tokens.get(cell_id, 0))

    def descriptor(self):
        return dict(instance_id=self.name, implementation_id=self.implementation_id,
            category_id=self.category_id, description='Queue/HOL scheduler test (local rule)',
            allowed_cell_ids=list(self.allowed_cells), active_cell_ids=sorted(self.active_cells),
            writable_actions=list(self.controlled_fields), parameters=dict(queue_threshold_bytes=4000,
            hol_budget_fraction=.5), selection_mode=self.selection_mode)

    def run(self, observation, policy_snapshot):
        started = time.perf_counter()
        users = iter_ue_observations(observation)
        actions, self.last_decisions = [], []
        for cid in sorted(self.active_cells):
            members = [u for u in users if str(u.serving_cell_id) == cid]
            if not members:
                self.last_decisions.append(dict(**self.identity(cid), slot_id=observation.slot_id,
                    reason='no_ue_reports_hold'))
                continue
            # Missing/stale measurements are excluded by the normal KPI contract.
            queue = sum(u.queue_bytes for u in members)
            pressure = max((u.hol_delay_ms / max(float(u.latency_budget_ms), 1e-6)
                            for u in members), default=0.)
            urgent = queue > 4000 or pressure >= .5
            mode = 'latency_aware' if urgent else 'round_robin'
            action = ConfigAction(target_cell_id=cid, source_name=self.name,
                scheduler_mode=mode, scheduler_policy_hint='latency_bias' if urgent else 'balanced')
            action._library_provenance = self.identity(cid)
            actions.append(action)
            self.last_decisions.append(dict(**self.identity(cid), slot_id=observation.slot_id,
                reason='queue_or_hol_pressure' if urgent else 'low_pressure',
                observed_queue_bytes=queue, observed_hol_fraction=pressure, requested_scheduler=mode))
            self.statistics['decisions'] += 1
        self.decision_seconds += time.perf_counter() - started
        return ControlDecision(config_actions=actions or [ConfigAction(source_name=self.name)])
