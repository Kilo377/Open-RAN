"""Unified intent-conditioned control dynamics; no outcomes or type-specific scores."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from test_smo_conflict_quantification import fixture

ROOT = Path(__file__).resolve().parents[1]


def review(g, f, c, p, show_dashed=False):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for the production read-only risk engine')
    code = """
const fs=require('fs'),vm=require('vm'),ctx={structuredClone};
vm.createContext(ctx);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),ctx);
const d=JSON.parse(fs.readFileSync(0,'utf8')),original=JSON.stringify(d);
const result=ctx.SmoConflictRisk.review(d.g,d.f,d.c.groups,d.c.ranges,d.p,{showDashed:d.show_dashed});
if(original!==JSON.stringify(d))throw Error('Risk input mutated');
process.stdout.write(JSON.stringify(result));
"""
    return json.loads(subprocess.check_output([node, '-e', code], cwd=ROOT, text=True,
        input=json.dumps(dict(g=g, f=f, c=c, p=p, show_dashed=show_dashed))))


def test_unified_signed_control_dynamics_no_kpi_prediction_or_feedback():
    g, f, c, p, _ = fixture()
    r = review(g, f, c, p);q = r['groups'][0]
    assert r['method'] == 'intent_control_dynamics_v4'
    assert q['score'] == pytest.approx(100 * 2 / 3)
    assert q['score_aggregation'] == 'intent_weighted_control_drive_cancellation'
    assert q['coverage']['comparable_pairs'] == 1
    assert 'direct' not in q and 'indirect' not in q
    assert not r['kpi_forecast_performed'] and not r['observed_after_kpis_used'] and not r['selection_performed']
    assert 'predicted_min' not in json.dumps(r) and 'violation_probability' not in json.dumps(r)
    f['states']['future_actual_kpi'] = dict(value=-999999)
    assert review(g, f, c, p) == r


def test_one_app_multiple_paths_aggregate_before_comparing_without_self_conflict():
    g, f, c, p, _ = fixture()
    p['apps'][0]['actions'] += p['apps'][1]['actions']
    c['groups'] = [dict(id='only-a', app_ids=['a'])]
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == 0 and q['interactions'] == []
    assert q['app_vectors'][0]['drive'] == pytest.approx(1)


def test_shared_parameter_uses_same_gain_dynamics_not_command_distance():
    g, f, c, p, a = fixture()
    p['apps'][1]['actions'] = [a('ue_offload_fraction', .3)]
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == pytest.approx(100)
    assert q['interactions'][0]['opposing']
    p['apps'][1]['actions'] = [a('ue_offload_fraction', .6)]
    assert review(g, f, c, p)['groups'][0]['score'] == 0


def test_partial_drx_fields_keep_known_influence_and_explicit_coverage():
    g, f, c, p, a = fixture()
    p['apps'][1]['actions'] = [a('ue_drx_configs', dict(a('ue_drx_configs', {})['current'], inactivity_ms=7, cycle_ms=80))]
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == pytest.approx(100 * 2 / 3)
    chain = next(x for x in q['chains'] if x['app_id'] == 'b' and x['parameter'] == 'ue_drx_configs')
    assert chain['drive'] == -1 and 'uncovered_structured_components' in chain['notes']
    assert q['score_status'] in ('partial_evidence', 'reference_approximation')


def joint_profile_fixture(uncertain=False):
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_drx_configs',dict(a('ue_drx_configs',{})['current'],cycle_ms=80,inactivity_ms=1))]
    edge=next(e for e in g['edges']if e.get('source')=='rcp:ue_drx_configs')
    edge['sensitivities'].append(dict(kind='finite_contrast',parameter_component='drx_enabled_population',
        contrast_id='measured_joint_profile',estimate=3.,confidence_interval=[-2,8]if uncertain else[2,4],
        **{'from':dict(cycle_ms=40,inactivity_ms=8),'to':dict(cycle_ms=80,inactivity_ms=1)}))
    return g,f,c,p,a


def test_actual_measured_joint_drx_profile_takes_precedence_without_double_counting_scalar_fields():
    g,f,c,p,_=joint_profile_fixture();q=review(g,f,c,p)['groups'][0]
    chain=next(x for x in q['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert chain['drive']==3 and q['score']==0
    assert chain['calibrations']==['measured_joint_profile']
    assert [x['kind']for x in chain['components']]==['structured_profile_reference_weight']
    assert 'uncovered_structured_components'not in chain['notes']


def test_measured_joint_profile_is_used_with_its_uncertainty_in_both_views():
    g,f,c,p,_=joint_profile_fixture(uncertain=True)
    hidden=next(x for x in review(g,f,c,p)['groups'][0]['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    shown=next(x for x in review(g,f,c,p,True)['groups'][0]['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert hidden['drive']==3 and 'uncertain_gain_direction'in hidden['notes']
    assert shown['drive']==3 and 'uncertain_gain_direction'in shown['notes']


def test_joint_profile_does_not_match_an_unmeasured_on_duration_change():
    g,f,c,p,_=joint_profile_fixture();p['apps'][1]['actions'][0]['proposed']['on_duration_ms']=8
    chain=next(x for x in review(g,f,c,p)['groups'][0]['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert chain['drive']==-7 and 'measured_joint_profile'not in chain['calibrations']
    assert 'uncovered_structured_components'in chain['notes']


def test_joint_profile_projects_nonuniform_members_and_supports_reference_reversal():
    g,f,c,p,_=joint_profile_fixture();f['states']['rcp:ue_drx_configs']['by_entity']['v']=deepcopy(f['states']['rcp:ue_drx_configs']['by_entity']['u'])
    chain=next(x for x in review(g,f,c,p)['groups'][0]['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert chain['drive']==1.5 and 'population_direction_projection'in chain['notes']
    g,f,c,p,_=joint_profile_fixture();old=p['apps'][1]['actions'][0]['current'];new=p['apps'][1]['actions'][0]['proposed']
    f['states']['rcp:ue_drx_configs']['by_entity']['u']=deepcopy(new)
    p['apps'][1]['actions'][0].update(current=deepcopy(new),proposed=deepcopy(old))
    chain=next(x for x in review(g,f,c,p)['groups'][0]['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert chain['drive']==-3 and 'common_reference_contrast'in chain['notes']


def test_observed_zero_member_component_is_neutral_on_an_otherwise_visible_edge():
    g,f,c,p,a=fixture();p['apps'][1]['actions']=[a('ue_scheduler_weights',1.5)]
    g['edges'].append(dict(id='weight-gain',kind='influences',source='rcp:ue_scheduler_weights',target='kpi:k',sensitivities=[
        dict(kind='range_linear_fit',contrast_id='measured-zero-u',members=['u'],estimate=0.,confidence_interval=[0.,0.]),
        dict(kind='range_linear_fit',contrast_id='nonzero-v',members=['v'],estimate=1.,confidence_interval=[1.,1.])]))
    f['states']['rcp:ue_scheduler_weights']['by_entity']['v']=1.
    q=review(g,f,c,p)['groups'][0];chain=next(x for x in q['chains']if x['app_id']=='b'and x['parameter']=='ue_scheduler_weights')
    assert chain['drive']==0 and 'observed_zero_reference_weight'in chain['notes']
    assert q['score']==0 and q['coverage']['known_chains']==q['coverage']['total_chains']
    g['edges'][-1]['sensitivities'].pop()
    hidden=review(g,f,c,p)['groups'][0]
    assert next(x for x in hidden['chains']if x['parameter']=='ue_scheduler_weights')['drive']==0


def test_zero_joint_profile_is_not_replaced_by_opposing_scalar_weights():
    g,f,c,p,_=joint_profile_fixture();edge=next(e for e in g['edges']if e.get('source')=='rcp:ue_drx_configs')
    edge['sensitivities'][-1].update(estimate=0.,confidence_interval=[0.,0.])
    q=review(g,f,c,p)['groups'][0];chain=next(x for x in q['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert chain['drive']==0 and chain['calibrations']==['measured_joint_profile'] and q['score']==0


def test_population_projection_is_approximation_not_per_ue_identification():
    g, f, c, p, _ = fixture()
    f['states']['rcp:ue_offload_fraction']['by_entity']['v'] = .5
    q = review(g, f, c, p)['groups'][0]
    chain = next(x for x in q['chains'] if x['app_id'] == 'a' and x['parameter'] == 'ue_offload_fraction')
    assert chain['drive'] == pytest.approx(1)
    assert 'population_direction_projection' in chain['notes']
    assert q['score'] == pytest.approx(100)


def test_no_parameter_family_is_dropped_because_apps_share_a_resource():
    g, f, c, p, a = fixture()
    # Shared offloading plus a distinct DRX action all participate exactly once.
    p['apps'][1]['actions'] += [a('ue_offload_fraction', .3)]
    q = review(g, f, c, p)['groups'][0]
    assert next(v for v in q['app_vectors'] if v['app_id'] == 'b')['drive'] == pytest.approx(-3)
    assert q['score'] == pytest.approx(80)


def test_proven_no_action_is_zero_missing_inference_is_unknown():
    g, f, c, p, _ = fixture()
    p['apps'][0].update(status='no_action', actions=[], inference_invoked=True)
    q = review(g, f, c, p)['groups'][0]
    assert not q['coverage']['missing_proposal_apps'] and q['score'] == 0
    p['apps'][0]['inference_invoked'] = False
    q = review(g, f, c, p)['groups'][0]
    assert q['coverage']['missing_proposal_apps'] == ['a'] and q['score'] is None


def test_missing_action_is_not_neutral_or_measured_zero():
    g, f, c, p, _ = fixture();p['apps'][1]['status'] = 'unavailable'
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] is None and q['coverage']['known_chains'] > 0
    assert q['coverage']['missing_proposal_apps'] == ['b']
    assert all(v['intent_role'] == 'context_unavailable' for v in q['app_vectors'] if v['app_id'] == 'b')


def test_same_direction_can_be_harmful_to_intent_without_inter_app_opposition():
    g, f, c, p, a = fixture()
    p['apps'][0]['actions'] = [a('ue_offload_fraction', .4)]
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == 0 and q['intent_direction']['against'] == 2


def test_measured_uncertain_gains_remain_in_score_and_chain_counts():
    g, f, c, p, _ = fixture();g['edges'][0]['sensitivities'][0]['confidence_interval'] = [-1, 12]
    hidden = review(g, f, c, p)['groups'][0];shown = review(g, f, c, p, True)['groups'][0]
    assert hidden['coverage']['total_chains'] == shown['coverage']['total_chains']
    assert hidden['score'] == shown['score']
    assert shown['score'] == pytest.approx(100 * 2 / 3)
    assert shown['score_status'] == 'partial_evidence'


def test_changed_unmeasured_gain_unknown_unchanged_gain_zero():
    g, f, c, p, _ = fixture();g['edges'][0]['sensitivities'] = []
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] is None
    unknown = next(v for v in q['app_vectors'] if v['app_id'] == 'a')
    assert unknown['drive'] is None


def test_extrapolation_uses_reference_weight_and_marks_approximation():
    g, f, c, p, a = fixture();p['apps'][0]['actions'] = [a('ue_offload_fraction', 1.1)]
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == pytest.approx(200 / 7)
    assert any('outside_reference_domain' in x['notes'] for x in q['chains'])
    assert q['score_status'] in ('partial_evidence', 'reference_approximation')


def test_sleep_family_uses_explicit_binary_reference_weight_not_profile_prediction():
    g, f, c, p, a = fixture();p['apps'][1]['actions'] = [a('cell_sleep_mode', 'light_sleep')]
    g['edges'].append(dict(id='sleep-gain', kind='influences', source='rcp:cell_sleep_mode', target='kpi:k', sensitivities=[dict(kind='finite_contrast', parameter_component='all_cells', estimate=-3, confidence_interval=[-4, -2], **{'from':'active', 'to':'micro_sleep'})]))
    q = review(g, f, c, p)['groups'][0]
    chain = next(x for x in q['chains'] if x['app_id'] == 'b' and x['parameter'] == 'cell_sleep_mode')
    assert chain['drive'] == -3 and chain['direction'] == -1
    assert 'sleep_family_reference_weight' in chain['notes']
    assert 'unmeasured_sleep_profile' in chain['notes']
    assert q['score'] == pytest.approx(80) and q['interactions'][0]['opposing']


def test_common_reference_contrast_composes_before_uncertainty_filter():
    g, f, c, p, a = fixture();f['states']['rcp:scheduler_mode']['by_entity']['c'] = 'latency_aware'
    p['apps'][1]['actions'] = [a('scheduler_mode', 'qos_deadline')]
    p['apps'][1]['actions'][0]['current'] = 'latency_aware'
    g['edges'].append(dict(id='mode-gain', kind='influences', source='rcp:scheduler_mode', target='kpi:k', sensitivities=[
        dict(kind='finite_contrast', parameter_component='all_cells', contrast_id='rr-lat', estimate=-.3, confidence_interval=[-1.1, .5], **{'from':'round_robin', 'to':'latency_aware'}),
        dict(kind='finite_contrast', parameter_component='all_cells', contrast_id='rr-qos', estimate=-6.9, confidence_interval=[-10.4, -3.4], **{'from':'round_robin', 'to':'qos_deadline'})]))
    q = review(g, f, c, p)['groups'][0]
    v = next(v for v in q['app_vectors'] if v['app_id'] == 'b')
    assert v['drive'] == pytest.approx(-6.6)
    assert q['score'] == pytest.approx(100 * 4 / 8.6)
    chain = next(x for x in q['chains'] if x['app_id'] == 'b' and x['parameter'] == 'scheduler_mode')
    assert 'common_reference_contrast' in chain['notes']


def test_recorded_proposals_must_match_actual_parameter_and_decision_slot():
    g, f, c, p, _ = fixture();p['slot'] = 99
    assert review(g, f, c, p)['status'] == 'unavailable'
    p['slot'] = 100;p['apps'][0]['actions'][0]['current'] = .4
    assert review(g, f, c, p)['groups'][0]['coverage']['missing_proposal_apps'] == ['a']


def test_structural_labels_do_not_affect_dynamics_score():
    g, f, c, p, _ = fixture();before = review(g, f, c, p)['groups'][0]['score']
    c['groups'][0]['conflict_summaries'] = [dict(type='direct', apps=['a', 'b']), dict(type='indirect', apps=['b', 'a']), dict(type='indirect', apps=['a', 'a'])]
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == before and q['structural_counts'] == dict(direct=1, indirect=1, implicit=0)


def test_zero_action_is_neutral_but_remains_in_the_specified_all_pair_normalization():
    g, f, c, p, _ = fixture();before = review(g, f, c, p)['groups'][0]['score']
    p['apps'].append(dict(app_id='z', status='no_action', inference_invoked=True, actions=[]))
    g['edges'].append(dict(id='z:drx', kind='controls', source='z', target='rcp:ue_drx_configs'))
    c['groups'][0]['app_ids'].append('z')
    # This is an all-pair average, not the old worst-local maximum. A known
    # zero app creates no opposition but adds known pair strength.
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == pytest.approx(before / 2)
    assert q['intent_direction']['against'] == 1


def test_known_score_preserved_when_third_app_unknown_with_partial_status():
    g, f, c, p, _ = fixture()
    p['apps'].append(dict(app_id='z', status='unavailable', actions=[]))
    g['edges'].append(dict(id='z:drx', kind='controls', source='z', target='rcp:ue_drx_configs'))
    c['groups'][0]['app_ids'].append('z')
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] == pytest.approx(100 * 2 / 3) and q['score_status'] == 'partial_evidence'


def test_fixed_frozen_scale_allows_current_urgency_to_change_not_group_membership():
    g, f, c, p, _ = fixture();c['ranges'][0].update(anchors=[dict(current=10)], lower=12)
    r = review(g, f, c, p);context = r['intent_contexts'][0]
    assert context['scale'] == 2 and context['weight'] == 1
    f['states']['kpi:k']['value'] = 16
    later = review(g, f, c, p)['intent_contexts'][0]
    assert later['scale'] == 2 and later['weight'] == pytest.approx(1/3)


def test_maintain_interval_uses_nearest_boundary_margin_not_forced_direction():
    g, f, c, p, _ = fixture();c['ranges'][0].update(lower=0, upper=20, lower_inclusive=True)
    q = review(g, f, c, p)['groups'][0]
    assert all(v['intent_role'] == 'inside_interval_no_unique_direction' for v in q['app_vectors'])
    f['states']['kpi:k']['value'] = 19
    q = review(g, f, c, p)['groups'][0]
    assert next(v for v in q['app_vectors'] if v['app_id'] == 'a')['intent_role'] == 'consuming_margin'
    assert next(v for v in q['app_vectors'] if v['app_id'] == 'b')['intent_role'] == 'building_margin'


def test_single_sided_zero_baseline_and_exact_hold_are_finite():
    g, f, c, p, _ = fixture();f['states']['kpi:k']['value'] = 0
    c['ranges'][0].update(lower=0, upper=None, current=0, anchors=[dict(current=0)])
    r = review(g, f, c, p);assert r['intent_contexts'][0]['scale'] > 0
    assert r['groups'][0]['score'] == pytest.approx(100*2/3)
    c['ranges'][0].update(lower=0, upper=0, lower_inclusive=True, upper_inclusive=True)
    q = review(g, f, c, p)['groups'][0]
    assert q['intent_direction']['against'] == 2


def test_empty_or_missing_intent_does_not_emit_misleading_score():
    g, f, c, p, _ = fixture();c['ranges'] = []
    assert review(g, f, c, p)['status'] == 'unavailable'
    c['ranges'] = [dict(kpi='k', lower=10, upper=9)]
    assert review(g, f, c, p)['reason'] == 'empty_intent_interval'


def test_cross_kpi_intent_normalization_urgency_and_all_pairs_one_score():
    g, f, c, p, _ = fixture()
    # k: +2,-1; k2: +2,+2. Target distances 1 and 10 respectively.
    for param, gain in [('ue_offload_fraction', 10), ('ue_drx_configs', -2)]:
        e = deepcopy(next(e for e in g['edges'] if e['kind']=='influences' and e['source']=='rcp:'+param))
        e.update(id=e['id']+'2', target='kpi:k2')
        e['sensitivities'][0].update(estimate=gain, confidence_interval=[gain,gain])
        g['edges'].append(e)
    c['ranges'][0].update(lower=11, anchors=[dict(current=10)])
    c['ranges'].append(dict(kpi='k2', lower=20, upper=None, anchors=[dict(current=10)]))
    f['states']['kpi:k2'] = dict(value=10)
    r = review(g, f, c, p);q = r['groups'][0]
    assert q['score'] == pytest.approx(100*2/(3+.4))
    # Only the requested k2 remains: same-direction effects give zero.
    c['ranges'] = c['ranges'][1:]
    assert review(g, f, c, p)['groups'][0]['score'] == 0


def test_unit_conversion_and_group_order_do_not_change_risk():
    g, f, c, p, _ = fixture();before = review(g, f, c, p)['groups'][0]['score']
    f['states']['kpi:k']['value'] *= 1000;c['ranges'][0]['lower'] *= 1000
    for e in g['edges']:
        for r in e.get('sensitivities', []):
            r['estimate'] *= 1000;r['confidence_interval'] = [v*1000 for v in r['confidence_interval']]
    c['groups'][0]['app_ids'].reverse()
    assert review(g, f, c, p)['groups'][0]['score'] == pytest.approx(before)


def test_opposing_zero_or_different_observations_not_silently_equated():
    g, f, c, p, _ = fixture()
    p['apps'][0]['observation_slot'] = 99;p['apps'][1]['observation_slot'] = 98
    assert review(g, f, c, p)['reason'] == 'inconsistent_public_observation'


def test_no_influence_edges_does_not_certify_zero_for_active_controls():
    g, f, c, p, _ = fixture()
    g['edges'] = [e for e in g['edges'] if e['kind'] == 'controls']
    q = review(g, f, c, p)['groups'][0]
    assert q['score'] is None and q['score_status'] == 'partial_evidence'


def test_measured_unchanged_categorical_edge_is_retained_as_neutral():
    g, f, c, p, _ = fixture()
    g['edges'].append(dict(id='mode-gain', kind='influences', source='rcp:scheduler_mode', target='kpi:k', sensitivities=[
        dict(kind='finite_contrast', parameter_component='all_cells', estimate=-.3, confidence_interval=[-1.1, .5], **{'from':'round_robin', 'to':'latency_aware'})]))
    hidden = review(g, f, c, p)['groups'][0]
    shown = review(g, f, c, p, True)['groups'][0]
    assert next(c for c in hidden['chains']if c['parameter']=='scheduler_mode')['drive']==0
    assert any(c['parameter'] == 'scheduler_mode' for c in shown['chains'])


def test_drx_cycle_and_inactivity_weights_compose_without_false_missing_field_warning():
    g, f, c, p, a = fixture()
    e=next(e for e in g['edges']if e['source']=='rcp:ue_drx_configs'and e['kind']=='influences')
    e['sensitivities'].append(dict(kind='range_linear_fit',scalar_component='cycle_ms',parameter_component='drx_enabled_population',fit_domain=[20,160],estimate=-.01,confidence_interval=[-.02,-.005],contrast_id='cycle'))
    p['apps'][1]['actions']=[a('ue_drx_configs',dict(a('ue_drx_configs',{})['current'],inactivity_ms=7,cycle_ms=80))]
    q=review(g,f,c,p)['groups'][0]
    chain=next(x for x in q['chains']if x['app_id']=='b'and x['parameter']=='ue_drx_configs')
    assert chain['drive']==pytest.approx(-1.4)
    assert 'uncovered_structured_components'not in chain['notes']
    assert q['score']==pytest.approx(100*2.8/3.4)
