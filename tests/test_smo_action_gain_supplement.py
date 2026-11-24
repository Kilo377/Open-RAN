"""Fixed-probe provenance, balanced estimation and topology-preserving admission."""
from copy import deepcopy
import json

import pytest

from smo.action_gain_supplement import apply, fit, sha, MEMBERS, SCHEMA


def observation(seed=1, origin_slot=2040, after=9.):
    return dict(worker='fixture', seed=seed, origin_slot=origin_slot,
        input_role='fixed_parameter_probe', arm='FIXED', split='training',
        learned_policy_controls_used=False, combination_evaluation_data_used=False,
        members=MEMBERS, from_weight=1., to_weight=.5, mec_weight=2.,
        scheduler_mode='latency_aware', prefix_equal=True, task_identities_equal=True,
        normal_controls_verified=True, only_competitor_weights_changed=True,
        network_configuration_verified=True,
        before=dict(mec_offloaded_latency_ms=10., ue_power_w=5.),
        after=dict(mec_offloaded_latency_ms=after, ue_power_w=4.9))


def observations():
    return [observation(seed=i) for i in (1, 2, 3)]


def test_repeated_contexts_do_not_give_a_seed_extra_weight():
    rows=observations()+[observation(seed=1,origin_slot=4040,after=7.)]
    result=fit(rows)
    # Seed effects are4,2,2; average them rather than pooling4 contexts.
    assert result[0]['estimate']==pytest.approx(8/3)
    assert result[0]['independent_seeds']==[1,2,3]
    assert result[0]['population_control_eligible']
    assert not result[0]['per_ue_causal_derivatives_identified']


@pytest.mark.parametrize('mutation',[
    {'arm':'FROZEN'}, {'split':'holdout'}, {'combination_evaluation_data_used':True},
    {'learned_policy_controls_used':True}, {'normal_controls_verified':False},
    {'only_competitor_weights_changed':False}, {'members':['ue_6']},
])
def test_deployment_results_or_different_controls_cannot_be_fitting_inputs(mutation):
    rows=observations();rows[0].update(mutation)
    with pytest.raises(ValueError):fit(rows)


def fixture(tmp_path):
    matrix=tmp_path/'base.json';matrix.write_text('{}')
    recipe_path=tmp_path/'recipe.json';recipe_path.write_text('{}')
    data=tmp_path/'measurements.json';data.write_text(json.dumps(dict(observations=observations(),combination_evaluation_data_used=False)))
    certificate=dict(schema=SCHEMA,environment_id='fixture',combination_evaluation_data_used=False,
        base_matrix_sha256=sha(matrix),runtime_recipe_sha256=sha(recipe_path),
        measurements=dict(path=data.name,sha256=sha(data)),estimates=fit(observations()))
    cert=tmp_path/'certificate.json';cert.write_text(json.dumps(certificate))
    entry=dict(influences=str(matrix),runtime_recipe=str(recipe_path),
        action_gain_supplement=dict(path=str(cert),sha256=sha(cert)))
    graph=dict(nodes=[dict(id='rcp:ue_scheduler_weights',kind='rcp'),
        dict(id='kpi:mec_offloaded_latency_ms',kind='kpi',unit='ms'),
        dict(id='kpi:ue_power_w',kind='kpi',unit='W')],
        edges=[dict(id='weights:'+key,kind='influences',source='rcp:ue_scheduler_weights',target='kpi:'+key,
                    sensitivities=[])for key in ('mec_offloaded_latency_ms','ue_power_w')])
    return graph,entry,dict(id='fixture'),certificate


def test_admission_preserves_topology_and_attaches_canonical_units(tmp_path):
    graph,entry,recipe,_=fixture(tmp_path);before=deepcopy(graph)
    updated,info=apply(graph,entry,recipe)
    assert graph==before
    assert [(e['id'],e['source'],e['target'])for e in updated['edges']]==[(e['id'],e['source'],e['target'])for e in graph['edges']]
    assert all(e['sensitivities'][0]['projection_units']['valid']for e in updated['edges'])
    latency = updated['edges'][0]['sensitivities'][0]
    assert latency['importance']['signed_reference_importance'] == pytest.approx(2 * .5 / 10)
    assert latency['importance']['normalization_basis'] == 'absolute_paired_reference_kpi'
    assert latency['contrast_label'] == latency['label']
    assert latency['estimate'] == 2.  # display normalization cannot rewrite the gain
    assert not info['topology_changed'] and not info['combination_evaluation_data_used']


def test_supplement_cannot_create_a_new_candidate_relation(tmp_path):
    graph,entry,recipe,_=fixture(tmp_path);graph['edges'].pop()
    with pytest.raises(ValueError,match='cannot create'):apply(graph,entry,recipe)


def test_forged_coefficient_is_rejected_even_with_an_updated_file_hash(tmp_path):
    graph,entry,recipe,certificate=fixture(tmp_path)
    certificate['estimates'][0]['estimate']=100.
    cert=tmp_path/'certificate.json';cert.write_text(json.dumps(certificate))
    entry['action_gain_supplement']['sha256']=sha(cert)
    with pytest.raises(ValueError,match='coefficients differ'):apply(graph,entry,recipe)
