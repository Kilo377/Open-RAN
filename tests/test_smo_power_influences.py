"""Unit conversion must preserve experiment context and statistical evidence."""
from copy import deepcopy
import hashlib
import io
import json
import tarfile

import pytest

from smo.power_influences import derive_power_estimate, derive_power_influences


def fixture_row():
    return dict(parameter='ue_offload_fraction',kpi='ue_energy_j',kind='range_linear_fit',
        unit='J / fraction',scoring_start_ms=200,scoring_end_ms=700,scoring_window_ms=500,
        horizon_ms=600,data_sha256='dataset-fixture',reference_parameter=.5,
        estimate=-.3,standard_error=.05,confidence_interval=[-.5,-.1],replicate_range=[-.4,-.2],
        reference_kpi_mean=3.,intercept_at_reference=2.9,sampled_levels=[0,.5,1],
        sample_count=3,high_minus_low_percent=-10.,sign_agreement=3,
        seed_fits=[dict(seed=17,beta=-.3,intercept=3.2,mean=3.)],
        grid_points=[dict(x=.5,mean=3.,valid_seed_count=3,by_seed=[dict(seed=17,value=3.)])],
        fit_diagnostics=dict(within_seed_r2=.9,rmse=.1,leave_level_out_rmse=.2,max_abs_residual=.3,
                             usable_for_linear_control=True,heuristic_thresholds=dict(min_r2=.7)),
        entity_estimates={'ue_0':-.1},entity_uncertainty={'ue_0':dict(estimate=-.1,standard_error=.01,confidence_interval=[-.12,-.08],sign_agreement=3)})


def proof():
    return dict(measurement_definition='scoring_window_energy_counter_delta',duration_s=.5,
        scoring_start_ms=200,scoring_end_ms=700,scoring_window_ms=500,
        measurement_dataset_sha256='dataset-fixture',archived_analysis_source_sha256='analysis-fixture')


def test_fixed_scoring_window_scales_all_energy_statistics_but_not_experiment_coordinates():
    row=fixture_row();before=deepcopy(row)
    converted=derive_power_estimate(row,proof(),source_estimates_sha256='source-fixture')
    assert row==before
    assert converted['kpi']=='ue_power_w' and converted['unit']=='W / fraction'
    for key in ('estimate','standard_error','reference_kpi_mean','intercept_at_reference'):
        assert converted[key]==row[key]*2
    assert converted['confidence_interval']==[-1.,-.2]
    assert converted['replicate_range']==[-.8,-.4]
    assert converted['seed_fits'][0]==dict(seed=17,beta=-.6,intercept=6.4,mean=6.)
    assert converted['grid_points'][0]['x']==.5 and converted['grid_points'][0]['mean']==6.
    assert converted['grid_points'][0]['by_seed'][0]==dict(seed=17,value=6.)
    assert converted['entity_estimates']=={'ue_0':-.2}
    assert converted['entity_uncertainty']['ue_0']['confidence_interval']==[-.24,-.16]
    assert converted['entity_uncertainty']['ue_0']['sign_agreement']==3
    assert converted['fit_diagnostics']['rmse']==.2
    assert converted['fit_diagnostics']['within_seed_r2']==.9
    assert converted['sampled_levels']==row['sampled_levels'] and converted['reference_parameter']==.5
    assert converted['high_minus_low_percent']==-10. and converted['sample_count']==3
    assert converted['horizon_ms']==600
    assert converted['derived_from']['duration_s']==.5  # Not horizon=.6s or full run=.8s.
    assert converted['derived_from']['source_estimates_sha256']=='source-fixture'


def test_binary_contrast_pair_observations_use_same_power_normalization():
    row=fixture_row();row.update(kind='finite_contrast',unit='J / contrast',
        seed_pairs=[dict(seed=17,low=3.,high=2.,reference=3.)],finite_step_high_minus_low=1.,
        parameter_levels={'low':'active','high':'micro_sleep'})
    converted=derive_power_estimate(row,proof(),source_estimates_sha256='source')
    assert converted['unit']=='W / contrast'
    assert converted['seed_pairs']==[dict(seed=17,low=6.,high=4.,reference=6.)]
    assert converted['parameter_levels']==row['parameter_levels']


@pytest.mark.parametrize('change',['definition','missing_duration','zero_duration','wrong_window','dataset','unit'])
def test_unproven_cumulative_or_wrong_timebase_is_rejected(change):
    row=fixture_row();p=proof()
    if change=='definition':p['measurement_definition']='terminal_cumulative_energy'
    elif change=='missing_duration':p.pop('duration_s')
    elif change=='zero_duration':p['duration_s']=0
    elif change=='wrong_window':p['duration_s']=.8
    elif change=='dataset':p['measurement_dataset_sha256']='different'
    else:row['unit']='W / fraction'
    with pytest.raises(ValueError):derive_power_estimate(row,p,source_estimates_sha256='source')


def make_files(folder, *, cumulative=False):
    dataset=dict(schema='oran.rcp-grid-measurements.v1',metric_scope='ue_population.v1',
        cases=[dict(metrics=dict(ue_energy_j=3.),proof=dict(verified_held_parameter_slots=1000))])
    data_bytes=json.dumps(dataset).encode();(folder/'measurement-dataset.json').write_bytes(data_bytes)
    source="""
start=round(plan['scoring_start_ms']*2)
end=round(plan['scoring_end_ms']*2)
if rows==start:
    first=deepcopy(row)
if rows==end:
    last=deepcopy(row)
values['ue_energy_j']=sum(last['ue_services'][uid]['ue_energy_total_j']-first['ue_services'][uid]['ue_energy_total_j'] for uid in energy_ids)
"""
    if cumulative:source=source.replace("-first['ue_services'][uid]['ue_energy_total_j']",'')
    source=source.encode()
    with tarfile.open(folder/'analysis-source.tgz','w:gz') as archive:
        info=tarfile.TarInfo('smo/identify.py');info.size=len(source);archive.addfile(info,io.BytesIO(source))
    (folder/'analysis-source-files.json').write_text(json.dumps({'smo/identify.py':hashlib.sha256(source).hexdigest()}))
    row=fixture_row();row['data_sha256']=hashlib.sha256(data_bytes).hexdigest()
    original=dict(schema='oran.intervention-edges.v1',metric_scope='ue_population.v1',estimates=[row])
    path=folder/'influences.json';path.write_text(json.dumps(original))
    return path


def test_high_level_derivation_preserves_original_source_and_adds_provenance(tmp_path):
    path=make_files(tmp_path);original=path.read_bytes();out=tmp_path/'derived'/'influences.json'
    result=derive_power_influences(path,out)
    assert path.read_bytes()==original
    assert len(result['estimates'])==2
    assert result['estimates'][0]==json.loads(original)['estimates'][0]
    assert result['estimates'][1]['kpi']=='ue_power_w'
    assert result['power_derivation']['source_sha256']==hashlib.sha256(original).hexdigest()
    assert json.loads(out.read_text())==result
    with pytest.raises(ValueError,match='overwrite'):derive_power_influences(path,path)


def test_archived_terminal_energy_cannot_be_relabelled_as_power(tmp_path):
    path=make_files(tmp_path,cumulative=True)
    with pytest.raises(ValueError,match='counter differences'):derive_power_influences(path)


def test_published_and_full_derived_rows_match_graph_service_exact_equality(tmp_path):
    path=make_files(tmp_path)
    published=json.loads(path.read_text())
    full=deepcopy(published)
    extra=deepcopy(full['estimates'][0]);extra['contrast_id']='another_control_context'
    full['estimates'].append(extra)
    full_path=tmp_path/'all-effects.json';full_path.write_text(json.dumps(full))
    a=derive_power_influences(path,tmp_path/'derived'/'influences.json')
    b=derive_power_influences(full_path,tmp_path/'derived'/'all-effects.json')
    power=next(row for row in a['estimates'] if row['kpi']=='ue_power_w')
    matrix={ (row['parameter'],row['kpi'],row.get('contrast_id')):row for row in b['estimates'] }
    # GraphService validates complete dict equality, including derivation proof.
    assert matrix[(power['parameter'],power['kpi'],power.get('contrast_id'))]==power
    canonical=hashlib.sha256(full_path.read_bytes()).hexdigest()
    assert power['derived_from']['source_estimates_sha256']==canonical
    assert a['power_derivation']['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert b['power_derivation']['source_sha256']==canonical
    assert a['power_derivation']['source_sha256']!=b['power_derivation']['source_sha256']


def test_published_energy_mismatch_cannot_borrow_canonical_provenance(tmp_path):
    path=make_files(tmp_path);full=json.loads(path.read_text())
    (tmp_path/'all-effects.json').write_text(json.dumps(full))
    changed=json.loads(path.read_text());changed['estimates'][0]['estimate']+=1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='canonical'):derive_power_influences(path)


@pytest.mark.parametrize('change',['fingerprint','case_window','scoring_window','missing_archive'])
def test_invalid_evidence_fails_closed(tmp_path,change):
    path=make_files(tmp_path)
    if change=='fingerprint':(tmp_path/'measurement-dataset.json').write_text('{}')
    elif change=='case_window':
        dataset=json.loads((tmp_path/'measurement-dataset.json').read_text());dataset['cases'][0]['proof']['verified_held_parameter_slots']=1200
        raw=json.dumps(dataset).encode();(tmp_path/'measurement-dataset.json').write_bytes(raw)
        estimates=json.loads(path.read_text());estimates['estimates'][0]['data_sha256']=hashlib.sha256(raw).hexdigest();path.write_text(json.dumps(estimates))
    elif change=='scoring_window':
        estimates=json.loads(path.read_text());estimates['estimates'][0]['scoring_end_ms']=800;path.write_text(json.dumps(estimates))
    else:(tmp_path/'analysis-source.tgz').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):derive_power_influences(path)
