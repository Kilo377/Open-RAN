"""Empty scheduler maps lost by row flattening need exact recorded evidence."""
from copy import deepcopy

import pytest

from test_smo_goal_obstruction import fixture,strength


def empty_bundle_fixture():
    g,f,c,p,_=fixture()
    f['states']['rcp:scheduler_mode']['by_entity']['empty_cell']='round_robin'
    c['groups']=[dict(id='one',app_ids=['a'])]
    g['edges'].append(dict(id='mode:k',kind='influences',source='rcp:scheduler_mode',target='kpi:k',sensitivities=[dict(
        parameter='scheduler_mode',kind='finite_contrast',contrast_id='mode_empty_cell',members=['empty_cell'],
        unit='W / contrast',estimate=2,confidence_interval=[2,2],**{'from':'round_robin','to':'latency_aware'})]))
    app=p['apps'][0]
    app['actions']=[dict(parameter='scheduler_mode',cell_id='empty_cell',ue_id=None,current='round_robin',proposed='latency_aware')]
    app['decisions']=[dict(slot_id=f['slot'],snapshot={'empty_cell':dict(scheduler_mode='latency_aware',ue_scheduler_weights={})},membership={'u':'c'})]
    return g,f,c,p


def test_recorded_empty_map_completes_bundle_without_creating_ue_actions():
    g,f,c,p=empty_bundle_fixture();original=deepcopy(p)
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']==0 and q['reason']is None
    evidence=q['recovered_empty_scheduler_bundles']
    assert len(evidence)==1
    assert evidence[0]['value']=={} and evidence[0]['members_in_cell']==0
    assert evidence[0]['decision_slot']==100 and evidence[0]['flat_ue_actions_added']==0
    assert evidence[0]['membership']=={'u':'c'}
    assert q['arbitration_resources'][0]['resource']==['scheduler_bundle','empty_cell']
    assert len(q['arbitration_resources'][0]['recovered_empty_scheduler_bundles'])==1
    assert p==original and len(p['apps'][0]['actions'])==1


@pytest.mark.parametrize('change',[
    'absent','stale','future','missing_membership','wrong_mode','wrong_cell','nonempty_weights',
    'membership_served','contradictory_snapshot','contradictory_membership','wrong_implementation',
])
def test_incomplete_bundles_stay_strict_without_exact_empty_cell_proof(change):
    g,f,c,p=empty_bundle_fixture();app=p['apps'][0];decision=app['decisions'][0]
    if change=='absent':app.pop('decisions')
    elif change=='stale':decision['slot_id']=99
    elif change=='future':decision['slot_id']=101
    elif change=='missing_membership':decision.pop('membership')
    elif change=='wrong_mode':decision['snapshot']['empty_cell']['scheduler_mode']='qos_deadline'
    elif change=='wrong_cell':decision['snapshot']['another_cell']=decision['snapshot'].pop('empty_cell')
    elif change=='nonempty_weights':decision['snapshot']['empty_cell']['ue_scheduler_weights']={'u':1}
    elif change=='membership_served':decision['membership']['u']='empty_cell'
    elif change=='contradictory_snapshot':
        other=deepcopy(decision);other['snapshot']['empty_cell']['scheduler_mode']='qos_deadline';app['decisions'].append(other)
    elif change=='contradictory_membership':
        other=deepcopy(decision);other['membership']={'v':'c'};app['decisions'].append(other)
    elif change=='wrong_implementation':decision['implementation_id']='unrelated-model'
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='incomplete_scheduler_bundle'


def test_recorded_membership_must_cover_existing_nonempty_flat_weight_actions():
    g,f,c,p=empty_bundle_fixture();app=p['apps'][0]
    app['actions'] += [dict(parameter='scheduler_mode',cell_id='c',ue_id=None,current='round_robin',proposed='latency_aware'),
        dict(parameter='ue_scheduler_weights',cell_id='c',ue_id='u',current=1,proposed=2)]
    app['decisions'][0]['snapshot']['c']=dict(scheduler_mode='latency_aware',ue_scheduler_weights={'u':2})
    valid=strength(g,f,c,p)['groups'][0]
    assert valid['reason']!='incomplete_scheduler_bundle'
    assert len(valid['recovered_empty_scheduler_bundles'])==1
    app['decisions'][0]['membership']={}
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='incomplete_scheduler_bundle'


def test_recorded_nonempty_snapshot_cannot_contradict_flat_weight_values():
    g,f,c,p=empty_bundle_fixture();app=p['apps'][0]
    app['actions'] += [dict(parameter='scheduler_mode',cell_id='c',ue_id=None,current='round_robin',proposed='latency_aware'),
        dict(parameter='ue_scheduler_weights',cell_id='c',ue_id='u',current=1,proposed=2)]
    app['decisions'][0]['snapshot']['c']=dict(scheduler_mode='latency_aware',ue_scheduler_weights={'u':1})
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='incomplete_scheduler_bundle'


def test_nonempty_snapshot_weight_cannot_be_missing_from_flat_actions():
    g,f,c,p=empty_bundle_fixture();app=p['apps'][0]
    app['actions'].append(dict(parameter='scheduler_mode',cell_id='c',ue_id=None,current='round_robin',proposed='latency_aware'))
    app['decisions'][0]['snapshot']['c']=dict(scheduler_mode='latency_aware',ue_scheduler_weights={'u':1})
    q=strength(g,f,c,p)['groups'][0]
    assert q['score']is None and q['reason']=='incomplete_scheduler_bundle'


def test_current_proof_is_not_invalidated_by_a_genuinely_older_membership():
    g,f,c,p=empty_bundle_fixture();old=deepcopy(p['apps'][0]['decisions'][0])
    old['slot_id']=99;old['membership']={'u':'empty_cell'}
    old['snapshot']['empty_cell']['ue_scheduler_weights']={'u':1}
    p['apps'][0]['decisions'].append(old)
    assert strength(g,f,c,p)['groups'][0]['score']==0
