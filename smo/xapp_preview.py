"""Shadow decisions from real public E2 inputs. Never executes a control action.

The simulator captures these at its decision boundary, before arbitration. Each
algorithm runs on a disposable copy; an OFF app is previewed as a cold activation.
Reading a replay retrieves that recorded inference, not an invented action.
"""
from copy import deepcopy
import hashlib
import json
import numbers
from pathlib import Path

FIELDS = ('ue_offload_fraction', 'ue_drx_configs', 'scheduler_mode',
          'ue_scheduler_weights', 'cell_sleep_mode')
SCHEMA = 'oran.xapp-decision-preview.v1'


def public_json(value):
    """Strict conversion of the already projected, CPU-only public surface."""
    if value is None or type(value) in (str, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, dict):
        return {str(k): public_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_json(v) for v in value]
    if hasattr(value, '__dict__'):
        return {k: public_json(v) for k, v in vars(value).items() if not k.startswith('_')}
    raise ValueError('Unsupported public preview value: '+type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def current_value(observation, cell_id, field, uid=None):
    cell = next((c for c in observation.cell_observations if c.cell_id == cell_id), None)
    config = getattr(cell, 'executed_config', None) or {}
    if field in ('ue_offload_fraction', 'ue_drx_configs'):
        user = next((u for u in observation.ue_observations if u.ue_id == uid), None)
        policy = getattr(user, 'ue_service_policy', None) or {}
        return deepcopy(policy.get('offload_fraction' if field == 'ue_offload_fraction' else 'drx'))
    value = config.get(field)
    return deepcopy(value.get(uid) if uid is not None and isinstance(value, dict) else value)


def preview_one(app, observation, policy=None, *, failures=()):
    """Clone full learner/lifecycle/RNG state; do not call an E2/executor path."""
    result = dict(app_id='xapp:'+app.implementation_id, status='unavailable',
        inference_invoked=False, actions=[], decisions=[], failures=list(failures),
        mode='state_clone' if app.active_cells else 'cold_activation')
    if failures:
        result['reason'] = 'kpi_contract_rejected'
        return result
    if getattr(app, 'mode', None) != 'frozen':
        result['reason'] = 'frozen_trained_model_required'
        return result
    replica = deepcopy(app)
    replica.log_path = None
    replica.parameters.pop('log_path', None)
    if not app.active_cells:
        # OFF apps have no decision to clone; preview exactly their next
        # activation from a clean lifecycle, retaining the frozen model.
        replica.reset()
        replica.select(replica.allowed_cells, {c: 0 for c in replica.allowed_cells})
    view = deepcopy(observation)
    encoded = public_json(view)
    result.update(input_sha256=digest(encoded), observation_slot=getattr(view, 'slot_id', None),
        report_produced_at_slot=getattr(view, 'rule_report_produced_at_slot', None))
    model = getattr(replica, 'model_path', None)
    before = hashlib.sha256(Path(model).read_bytes()).hexdigest() if model else None
    result['model_sha256'] = before
    if app.implementation_id == 'ue_energy_rf':
        from smo.model_release import catalogue_sha256
        version=getattr(app,'action_set_version','legacy9')
        result.update(action_set_version=version,
            action_catalogue_sha256=getattr(app,'action_catalogue_sha256',catalogue_sha256(version)),
            implementation_variant=getattr(app,'implementation_variant','legacy9'),
            runtime_factory=getattr(app,'runtime_factory',None))
    try:
        result['inference_invoked'] = True
        decision = replica.run(view, deepcopy(policy))
        if model and hashlib.sha256(Path(model).read_bytes()).hexdigest() != before:
            raise RuntimeError('Frozen model changed during preview')
        from ric.xapp_contract import validate_action
        invalid = [issue for action in decision.config_actions
                   for issue in validate_action(action, replica.runtime_contract(policy), observation)]
        rows = []
        for action in decision.config_actions:
            cell = action.target_cell_id
            for field in FIELDS:
                value = getattr(action, field, None)
                if value is None:
                    continue
                items = value.items() if isinstance(value, dict) else [(None, value)]
                for uid, proposed in items:
                    current = current_value(observation, cell, field, uid)
                    rows.append(dict(parameter=field, cell_id=cell, ue_id=uid,
                        current=public_json(current), proposed=public_json(proposed),
                        changed=None if current is None else current != proposed))
        result.update(status='proposed' if rows else 'unavailable' if getattr(replica, 'status', None) == 'not_evaluable' else 'no_action', actions=rows,
            decisions=public_json(getattr(replica, 'last_decisions', [])),
            reason=None if rows else getattr(replica, 'status_reason', 'no_new_action_this_decision'))
        result['action_contract_failures'] = invalid
        if invalid:
            result.update(status='invalid_proposal', reason='action_contract_rejected')
        # A nonempty command/constraint cannot silently disappear in a readout.
        if decision.commands or decision.policy_constraints:
            result.update(status='unavailable', reason='unsupported_nonconfiguration_proposal', actions=[])
    except Exception as error:
        result.update(status='unavailable', reason='model_preview_failed', error=type(error).__name__, actions=[])
    return result


def capture(manager, observation, slot, apps):
    """Use the exact public projection and receipt ownership of normal RIC."""
    from ric.xapp_contract import project
    policy = manager.policy_store.get_snapshot()
    rows = []
    for app in apps:
        contract = app.runtime_contract(policy)
        try:
            view, failures = project(observation, contract, manager.xapp_interval_slots, slot)
            # Share the same receipt builder as deployed algorithms; selection
            # alone must never confirm a still-queued action.
            manager._attach_control_ownership(view, observation)
            row = preview_one(app, view, policy, failures=failures)
        except Exception as error:
            row = dict(app_id='xapp:'+app.implementation_id, status='unavailable',
                inference_invoked=False, actions=[], reason='preview_input_failed',
                error=type(error).__name__, mode='state_clone' if app.active_cells else 'cold_activation')
        rows.append(row)
    result = dict(schema=SCHEMA, slot=slot, observation_slot=observation.slot_id,
        source='public_E2_projection_at_RIC_decision_boundary', apps=rows,
        automatic_selection=False, deployment_enabled=False, arbitration_performed=False)
    json.dumps(result, allow_nan=False)
    return result


class PreviewStore:
    def __init__(self, run=None, live=None):
        self.run, self.live = Path(run) if run else None, Path(live) if live else None

    def read(self, slot):
        from smo.runtime import records
        row = None
        if self.live and self.live.exists():
            row = json.loads(self.live.read_text()).get('decision_preview')
            if row is not None:
                row['requested_slot'] = slot
                row['snapshot_note'] = '最近已采集的决策快照；参数当前值以该快照的报告时刻为准。'
        elif self.run:
            row = next((r for r in records(self.run, 'xapp_decision_previews') if r.get('slot') == slot), None)
        if row is None or (not self.live and row.get('slot') != slot):
            return dict(schema=SCHEMA, slot=slot, status='unavailable', apps=[],
                reason='此快照未记录完整的 xApp 决策预演；旧 KPI 回放不能还原模型动作。',
                automatic_selection=False, deployment_enabled=False)
        if row.get('schema') != SCHEMA or row.get('deployment_enabled') is not False:
            raise ValueError('Invalid decision-preview recording')
        return deepcopy(row)
