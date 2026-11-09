"""A classified handover no-op never substitutes for applied control evidence."""
from copy import deepcopy

import pytest

from experiments import run_urllc_qos_study as study
from experiments.urllc_control_audit import (
    AUDIT_VERSION, classify_stale_rejections, is_urllc_receipt)


def recording():
    old = dict(scheduler_mode='qos_deadline', ue_scheduler_weights={'ue_0': 1., 'ue_6': 2., 'ue_8': 1.})
    current = dict(scheduler_mode='qos_deadline', ue_scheduler_weights={'ue_6': 2., 'ue_8': 1.})
    before = {'cell_0': deepcopy(current)}
    rejected = dict(status='rejected', reason='ue_scheduler_weight_target_outside_cell_scope',
        slot_id=8692, issued_at_slot=8691, apply_scope='cell', target_cell_id='cell_0',
        target_cell_ids=['cell_0'], before=before, after=deepcopy(before),
        fields=[dict(field=k, value=v) for k,v in old.items()],
        provenance=[dict(implementation_id='latency_urllc_rf')])
    applied = dict(status='applied', apply_slot=8702, target_cell_id='cell_0',
        target_cell_ids=['cell_0'], after={'cell_0': deepcopy(current)},
        fields=[dict(field=k, value=v) for k,v in current.items()],
        provenance=[dict(implementation_id='latency_urllc_rf')])
    events = [dict(kind='decision', slot_id=8600, action=2, status='fixed', snapshot={'cell_0': old}),
        dict(kind='decision', slot_id=8700, action=2, status='fixed', snapshot={'cell_0': current}),
        dict(kind='executed_policy_confirmed', produced_slot=8705, action=2,
            executed_snapshot={'cell_0': deepcopy(current)})]
    return events, [rejected, applied]


def test_direct_controller_reports_noop_without_filtering_any_raw_receipt(tmp_path, monkeypatch):
    events, receipts = recording()
    saved = deepcopy(receipts)
    reader = lambda folder,kind: iter(receipts if kind == 'control_receipts' else events)
    monkeypatch.setattr(study, 'jsonl', reader)
    result = study.controller_audit(tmp_path, {}, {}, 'qos2', tmp_path/'unused.json')
    assert study.CONTROLLER_AUDIT_VERSION == AUDIT_VERSION
    assert result['applied_receipts'] == result['confirmed'] == 1
    assert result['rejected_receipts_total'] == result['accepted_stale_scope_noop_rejections'] == 1
    assert result['rejection_classifications'][0]['stale_ue_ids'] == ['ue_0']
    assert result['raw_control_receipts_preserved'] and receipts == saved
    assert study.jsonl is reader  # The production audit installs no reader wrapper.


@pytest.mark.parametrize('change', ['unknown_reason', 'unknown_status', 'non_noop', 'unknown_uid', 'bad_weight', 'no_prior_decision', 'malformed_fields'])
def test_other_rejections_are_fatal_even_when_an_applied_ack_exists(tmp_path, monkeypatch, change):
    events, receipts = recording()
    rejected = receipts[0]
    if change == 'unknown_reason': rejected['reason'] = 'invalid_ue_scheduler_weight'
    elif change == 'unknown_status': rejected['status'] = 'deferred'
    elif change == 'non_noop': rejected['after']['cell_0']['scheduler_mode'] = 'round_robin'
    elif change == 'unknown_uid': rejected['fields'][1]['value']['ghost'] = 1.
    elif change == 'bad_weight': rejected['fields'][1]['value']['ue_6'] = 9.
    elif change == 'no_prior_decision': events.pop(0)
    else: rejected['fields'] = [None, None]
    monkeypatch.setattr(study, 'jsonl', lambda folder,kind: iter(receipts if kind == 'control_receipts' else events))
    with pytest.raises(ValueError):
        study.controller_audit(tmp_path, {}, {}, 'qos2', tmp_path/'unused.json')


@pytest.mark.parametrize('change', ['none', 'late', 'wrong_snapshot'])
def test_classified_rejection_cannot_supply_missing_or_invalid_applied_ack(tmp_path, monkeypatch, change):
    events, receipts = recording()
    if change == 'none': receipts.pop()
    elif change == 'late': receipts[1]['apply_slot'] = 8710
    else: receipts[1]['after']['cell_0']['ue_scheduler_weights']['ue_6'] = 1.
    monkeypatch.setattr(study, 'jsonl', lambda folder,kind: iter(receipts if kind == 'control_receipts' else events))
    with pytest.raises(ValueError, match='applied ACK|earlier matching applied action'):
        study.controller_audit(tmp_path, {}, {}, 'qos2', tmp_path/'unused.json')


def test_classifier_is_read_only_and_ignores_only_unrelated_provenance():
    events, receipts = recording()
    other = dict(status='rejected', reason='some_other_app', provenance=[None, dict(implementation_id='other')])
    original = deepcopy(receipts + [other])
    classified = classify_stale_rejections(receipts + [other], events[:2])
    assert len(classified) == 1 and receipts + [other] == original
    assert is_urllc_receipt(receipts[0]) and not is_urllc_receipt(other)
