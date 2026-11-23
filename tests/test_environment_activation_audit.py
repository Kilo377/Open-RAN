"""Consumer proofs distinguish an accepted write from an actual policy change."""
from copy import deepcopy
from experiments.audit_environment_activation import controlled_value,receipt_evidence,planned_selection


def row():
    state=dict(scheduler_mode='round_robin',ue_scheduler_weights={'u':1.},
        ue_service_policies={'u':dict(offload_fraction=1.,drx=dict(cycle_ms=40,inactivity_ms=8))},
        ap_energy_total_j=10)
    return dict(status='applied',slot_id=2081,issued_at_slot=2080,
        provenance=[dict(implementation_id='test',instance_id='test',fields=['ue_offload_fraction'])],
        before={'cell':deepcopy(state)},after={'cell':deepcopy(state)},
        fields=[dict(field='ue_offload_fraction',value={'u':1.},route='scheduler')])


def test_applied_unchanged_policy_is_noop_even_when_energy_counter_changes():
    r=row();r['after']['cell']['ap_energy_total_j']=11
    p=receipt_evidence(r,'test')
    assert p['status']=='applied'and not p['changed']and not p['mismatches']and p['unresolved']==0


def test_changed_profile_is_verified_against_executed_state_and_owner():
    r=row();r['fields'][0]['value']={'u':.75};r['after']['cell']['ue_service_policies']['u']['offload_fraction']=.75
    p=receipt_evidence(r,'test')
    assert p['changed']and not p['mismatches']and p['resolved']==1
    assert p['owner_provenance'][0]['implementation_id']=='test'
    assert receipt_evidence(r,'another')is None


def test_requested_payload_is_not_assumed_consumed():
    r=row();r['fields'][0]['value']={'u':.75}
    p=receipt_evidence(r,'test');assert p['mismatches']and not p['changed']
    r['fields'][0]['field']='not_recorded'
    p=receipt_evidence(r,'test');assert p['unresolved']==1 and p['mismatches']


def test_drx_subset_and_scheduler_scope_read_actual_consumers():
    state=row()['after']['cell']
    value,ok=controlled_value(state,'ue_drx_configs',{'u':{'inactivity_ms':8}})
    assert ok and value['u']['cycle_ms']==40
    value,ok=controlled_value(state,'ue_scheduler_weights',{'u':1.})
    assert ok and value=={'u':1.}
    _,ok=controlled_value(state,'ue_scheduler_weights',{'missing':1.})
    assert not ok


def test_recorded_hot_timeline_uses_scheduled_slot_not_request_milliseconds():
    hot=dict(initial=[],timeline=[dict(operation='set',apps=['test'],scheduled_slot=2040,
        source='simulation_timeline',request_id='timeline:0')])
    assert planned_selection(hot,'test',2040)
    hot['timeline'][0]['scheduled_slot']=2080
    assert not planned_selection(hot,'test',2040)
