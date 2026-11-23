"""Synthetic prefix/provenance fixtures; no invented xApp performance."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from smo import evaluation_reference as reference
from smo.evaluation_origin_guard import OriginGuard,validate_proof
from smo.deployment_worker import plan,fixed_origin_comparison


def prepared_fixture(tmp_path,monkeypatch):
    output=tmp_path/'jobs';run=output/'j/data/run';(run/'xapp_model_checkpoints').mkdir(parents=True)
    model=run/'xapp_model_checkpoints/m.json';model.write_text('fixture frozen model, not executed')
    (run/'xapp_hot_start.json').write_text(json.dumps(dict(models={'a':dict(
        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),checkpoint='xapp_model_checkpoints/m.json')})))
    config=dict(source_sha256='original-wrapper',manifest=dict(rng_seed=7,slot_duration_ms=.5,total_slots=1420,
        ues=[dict(ue_id='u')],cells=[dict(cell_id='c')]),configs=dict(phy=dict(rt_sys=dict(ue_services={}))))
    (run/'config_snapshot.json').write_text(json.dumps(config))
    archive=tmp_path/'source.tgz'
    with tarfile.open(archive,'w:gz')as target:
        raw=b'unchanged fixture network dynamics';item=tarfile.TarInfo('core/state.py');item.size=len(raw)
        target.addfile(item,io.BytesIO(raw))
    batch=dict(id='b',seed=7,worker_source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        members=[dict(job_id='j',phase='complete',comparison=dict(baseline_origin='fresh_run_all_xapps_off',baseline={'slot':520}))])
    rows=[dict(slot_id=slot,ues=[dict(ue_id='u',cell_id='c',packet_delivered_bytes=slot*100,
        packet_completed_packets=slot,packet_completion_latency_ms_sum=slot,packet_pending_bytes=0)],
        ue_services={'u':dict(ue_energy_total_j=slot*.001)},ap_services=dict(cells={'c':dict(
            ap_energy_total_j=slot*.005,ap_available_slots_total=slot,ap_policy='active')}),
        xapp_bindings={},traffic_arrivals={'u':{'exogenous_bytes':100,'mec_result_bytes':0}})for slot in range(1,522)]
    preview=dict(slot=520,status='ready',apps=[dict(app_id='a',mode='cold_activation')])
    def recorded(path,name):
        return iter(rows if name=='rt_sys_trace'else[preview]if name=='xapp_decision_previews'else[])
    monkeypatch.setattr(reference,'records',recorded)
    result=reference.prepare_origin(batch,output,source_archive=archive)
    assert result['status']=='ready'
    monkeypatch.setattr('smo.evaluation_origin_guard.dynamics_fingerprint',lambda _:result['proof']['dynamics_sha256'])
    return result,config,rows,batch,output,archive


def test_preparation_uses_exact_before_prefix_and_preserves_source(tmp_path,monkeypatch):
    prepared,config,rows,batch,output,archive=prepared_fixture(tmp_path,monkeypatch)
    original=deepcopy((config,rows,batch));proof=prepared['proof']
    assert proof['baseline_slot']==520 and proof['observe_slots']==500
    assert prepared['frame']['states']['kpi:ue_power_w']['value']==pytest.approx(2.)
    assert not proof['after_kpis_used']and proof['new_simulations']==0
    rows[-1]['ue_services']['u']['ue_energy_total_j']=999999 # AFTER cannot enter origin
    again=reference.prepare_origin(batch,output,source_archive=archive)
    assert again['proof']==proof
    assert batch==original[2]and config==original[0]
    assert reference.prepare_origin(batch,output,observe_slots=1000,source_archive=archive)['status']=='unavailable'
    assert reference.prepare_origin(batch,output)['reason']=='evaluation_origin_source_archive_required'


def test_guard_verifies_before_any_selection_and_rejects_changed_state(tmp_path,monkeypatch):
    prepared,config,rows,_,_,_=prepared_fixture(tmp_path,monkeypatch)
    run=Path(prepared['run_path']);guard=OriginGuard(prepared['proof'],config,run)
    for row in rows[:520]:guard.observe(row)
    verified=json.loads((run/'evaluation_origin_verified.json').read_text())
    assert guard.verified and verified['verified_before_activation']and verified['network_dynamics_code_equal']
    assert verified['origin_id']==prepared['proof']['origin_id']
    guard=OriginGuard(prepared['proof'],config,run)
    for row in rows[:519]:guard.observe(row)
    changed=deepcopy(rows[519]);changed['ue_services']['u']['ue_energy_total_j']+=.1
    with pytest.raises(ValueError,match='BEFORE state differs'):guard.observe(changed)
    guard=OriginGuard(prepared['proof'],config,run)
    with pytest.raises(ValueError,match='activated before'):guard.observe({**rows[0],'xapp_bindings':{'c':{'a':['a']}}})


def test_guard_rejects_code_models_configuration_and_tampered_proof(tmp_path,monkeypatch):
    prepared,config,_,_,_,_=prepared_fixture(tmp_path,monkeypatch);proof=prepared['proof'];run=Path(prepared['run_path'])
    bad=deepcopy(proof);bad['before_frame']['states']['kpi:ue_power_w']['value']=999
    with pytest.raises(ValueError,match='digest mismatch'):validate_proof(bad)
    cfg=deepcopy(config);cfg['manifest']['rng_seed']=8
    with pytest.raises(ValueError,match='configuration differs'):OriginGuard(proof,cfg,run)
    with monkeypatch.context()as context:
        context.setattr('smo.evaluation_origin_guard.dynamics_fingerprint',lambda _:'changed-code')
        with pytest.raises(ValueError,match='dynamics code differs'):OriginGuard(proof,config,run)
    (run/'xapp_model_checkpoints/m.json').write_text('changed model')
    with pytest.raises(ValueError,match='model_pin_mismatch'):OriginGuard(proof,config,run)


def test_worker_origin_window_seed_and_cli_are_frozen(tmp_path,monkeypatch):
    prepared,_,_,_,_,_=prepared_fixture(tmp_path,monkeypatch)
    request=dict(id='a'*32,seed=7,app_ids=['xapp:ue_energy_rf'],observe_slots=500,expected_origin=prepared['proof'])
    _,_,baseline,_,cmd=plan(request,tmp_path)
    assert baseline==520 and '--evaluation-origin-file'in cmd
    for changes in ({'seed':8},{'observe_slots':1000}):
        with pytest.raises(ValueError,match='origin differs'):plan({**request,**changes},tmp_path)


def test_live_tuple_configuration_matches_its_archived_json_representation():
    config=dict(manifest=dict(positions=((1.,2.,3.),(4.,5.,6.))),configs=dict(phy=dict(rx_shape=(2,1))))
    archived=json.loads(json.dumps(config))
    live=reference.origin_configuration(config);saved=reference.origin_configuration(archived)
    assert live==saved and reference.comparable(live,saved)
    assert live['manifest']['positions']==[[1.,2.,3.],[4.,5.,6.]]
    assert config['configs']['phy']['rx_shape']==(2,1) # caller state is unchanged


def test_common_after_window_does_not_move_with_first_receipt():
    from test_smo_intent_comparison import setup,review,scalar_index
    store,ident,graph,source,records=setup();prior=review(store,ident,graph,source,source['frames'][-1])
    index=scalar_index(source);before=index.window([r['kpi']for r in prior['rows']],100,600,through=1400)
    request=dict(records=records,observe_slots=500)
    first=fixed_origin_comparison(prior,request,graph,index,before,600)
    moved=deepcopy(prior);moved['application_slot']=800
    second=fixed_origin_comparison(moved,request,graph,index,before,600)
    assert first['after_window']==second['after_window']==dict(window_start_slot=600,slot=1100,window_slots=500)
    assert first['rows']==second['rows']
    assert first['application_slot']==640 and second['application_slot']==800
    assert first['deployment_latency_included']and first['fixed_combination_observation_window']
