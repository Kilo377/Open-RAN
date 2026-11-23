"""Operator runtime admission and window tests; no simulations or SSH calls."""
import json
from copy import deepcopy
from pathlib import Path
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from smo.__main__ import handler
from test_smo_deployment import service_fixture
from smo.runtime_recipe import (SCHEMA,PROFILE_NAMES,sha256,digest,source_pins,
    load_recipe,worker_recipe,validate_worker_recipe)
from smo.model_release import MODEL_IDS


@pytest.fixture
def runtime_fixture(tmp_path):
    """Pinned inert files exercise admission; none are simulator inputs."""
    root=tmp_path/'source';(root/'core').mkdir(parents=True)
    for n in range(317):(root/'core'/f'fixture{n}.py').write_text(f'# inert fixture {n}\n')
    models={}
    for n,model_id in enumerate(MODEL_IDS):
        name=f'artifacts/xapps/chongyu/20261117/models/{model_id}.json'
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'fixture':n}));models[name]=sha256(p)
    profiles={};environment=tmp_path/'environment';environment.mkdir()
    for key,name in PROFILE_NAMES.items():
        p=environment/name;p.write_text(json.dumps({'fixture':key}));profiles[key]=dict(path=str(p),sha256=sha256(p))
    pins=source_pins(root);run=tmp_path/'recorded';run.mkdir()
    config={'manifest':{'rng_seed':7,'slot_duration_ms':.5},'configs':{}}
    (run/'config_snapshot.json').write_text(json.dumps(config))
    proof=dict(schema='oran.evaluation-origin.v1',seed=7,baseline_slot=2040,observe_slots=2000,
        configuration=config,before_frame={'slot':2040,'states':{}},dynamics_sha256=pins['sha256'],
        model_sha256={MODEL_IDS[Path(name).stem]:pin for name,pin in models.items()},
        source_sha256='a'*64,source_archive_sha256='b'*64,after_kpis_used=False,intent_bounds_rebased=False)
    proof['origin_id']=digest(proof);origin=tmp_path/'origin.json';origin.write_text(json.dumps(proof))
    recipe=dict(schema=SCHEMA,id='fixture_common',seed=7,baseline_slot=2040,origin_observe_slots=2000,
        tail_slots=400,traffic_load=1.5,dynamics_sha256=pins['sha256'],dynamics_members=317,
        validated_source_sha256='a'*64,validated_source_archive_sha256='b'*64,
        frozen_models=models,profiles=profiles,validated_origin=dict(path=str(origin),sha256=sha256(origin),
            run=str(run),config_sha256=sha256(run/'config_snapshot.json')))
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe))
    (root/'.smo-owned-worker.json').write_text(json.dumps({'sha256':'c'*64}))
    return root,path,recipe,proof


def test_local_recipe_checks_each_profile_source_models_and_origin(runtime_fixture):
    root,path,recipe,proof=runtime_fixture
    assert load_recipe(path,source_root=root,seed=7)==(recipe,proof)
    for key in PROFILE_NAMES:
        p=Path(recipe['profiles'][key]['path']);raw=p.read_bytes();p.write_text('tampered')
        with pytest.raises(ValueError,match='profile SHA-256'):load_recipe(path,source_root=root)
        p.write_bytes(raw)
    model=root/next(iter(recipe['frozen_models']));raw=model.read_bytes();model.write_text('tampered')
    with pytest.raises(ValueError,match='frozen model differs'):load_recipe(path,source_root=root)
    model.write_bytes(raw)
    (root/'core/fixture0.py').write_text('# changed dynamics\n')
    with pytest.raises(ValueError,match='dynamics differ'):load_recipe(path,source_root=root)


def test_worker_plan_keeps_certified_before_independent_of_requested_after(runtime_fixture):
    from smo.deployment_worker import plan
    root,_,recipe,proof=runtime_fixture
    frozen=worker_recipe(recipe,root,proof)
    original=deepcopy(frozen)
    for after,total in ((500,2940),(20000,22440)):
        request=dict(id='d'*32,seed=7,app_ids=['xapp:ue_energy_rf'],observe_slots=after,
            expected_origin=proof,runtime_recipe=frozen,expected_source_archive_sha256='c'*64)
        _,_,baseline,slots,command=plan(request,root)
        assert baseline==2040 and slots==total
        assert command[command.index('--traffic-load')+1]=='1.5'
        assert command[command.index('--xapp-preview-before-slot')+1]=='2040'
        for flag,key in (('--traffic-profile','traffic_profile'),('--ue-services-config','ue_services_config'),
                         ('--ap-services-profile','ap_services_profile'),('--config','base_config')):
            assert command[command.index(flag)+1]==frozen['profiles'][key]['path']
        assert '--evaluation-origin-file'in command
        with pytest.raises(ValueError,match='seeds differ'):plan({**request,'seed':8},root)
    assert frozen==original and proof['observe_slots']==2000


def test_worker_rejects_recipe_path_escape_proof_or_snapshot_change(runtime_fixture):
    root,_,recipe,proof=runtime_fixture;remote=worker_recipe(recipe,root,proof)
    assert validate_worker_recipe(remote,root,proof,'c'*64)==remote
    bad=deepcopy(remote);bad['profiles']['base_config']['path']=str(root.parent/'outside.json')
    bad.pop('recipe_sha256');bad['recipe_sha256']=digest(bad)
    with pytest.raises(ValueError,match='outside the owned'):validate_worker_recipe(bad,root,proof,'c'*64)
    badproof=deepcopy(proof);badproof['seed']=8;badproof.pop('origin_id');badproof['origin_id']=digest(badproof)
    with pytest.raises(ValueError,match='origin differ'):validate_worker_recipe(remote,root,badproof,'c'*64)
    with pytest.raises(ValueError,match='snapshot differs'):validate_worker_recipe(remote,root,proof,'e'*64)
    p=Path(remote['profiles']['base_config']['path']);raw=p.read_bytes();p.unlink()
    other=root.parent/'same-bytes.json';other.write_bytes(raw);p.symlink_to(other)
    with pytest.raises(ValueError,match='outside the owned'):validate_worker_recipe(remote,root,proof,'c'*64)


def test_batch_freezes_recipe_and_preserves_legacy_requests(runtime_fixture,monkeypatch):
    from smo.deployment import DeploymentManager
    from smo.deployment_batch import BatchEvaluator
    root,path,recipe,proof=runtime_fixture
    # Admission checks the real tree in production; this fixture supplies the
    # already-tested inert source tree for the manager constructor only.
    import smo.runtime_recipe as runtime
    original=runtime.load_recipe
    monkeypatch.setattr(runtime,'load_recipe',lambda p,seed:original(p,source_root=root,seed=seed))
    monkeypatch.setattr(DeploymentManager,'watch',lambda *a,**k:None)
    monkeypatch.setattr(BatchEvaluator,'launch',lambda *a:None)
    (path.parent/'source.tgz').write_bytes(b'inert pinned harness snapshot')
    config=dict(host='user@100.0.0.1',socket='/tmp/fixture.sock',project_root=str(root),python='/fixture/python',
        local_output=str(path.parent/'jobs'),seed=7,label='Fixture',runtime_recipe=str(path))
    pool=path.parent/'pool.json';pool.write_text(json.dumps(config));manager=DeploymentManager(pool)
    group=dict(id='fixture-group',app_ids=['xapp:ue_energy_rf'],selected_names=['UE power'])
    batch=manager.batch.start(records=[],groups=[group],observe_slots=20000,context={},snapshot={},ranges=[],show_dashed=False)
    assert batch['context']['evaluation_origin']==proof
    manager.runtime_recipe['traffic_load']=9
    row=manager.start(records=[],group=group,observe_slots=20000,context=batch['context'],batch_id=batch['id'],batch_index=0)
    assert row['runtime_recipe']['traffic_load']==1.5 and row['expected_origin']['observe_slots']==2000
    assert row['observe_slots']==20000 and row['runtime_recipe']['validated_source_archive_sha256']=='b'*64
    assert row['expected_source_archive_sha256']==sha256(path.parent/'source.tgz')


def test_evaluation_uses_certified_before_and_requested_after(runtime_fixture,monkeypatch):
    from smo.deployment_worker import evaluate
    from test_smo_intent_comparison import setup
    root,_,recipe,proof=runtime_fixture;run=Path(recipe['validated_origin']['run'])
    _,_,_,_,records=setup();queries=[]
    states={'kpi:ue_power_w':{'value':6.},'kpi:mec_expected_gflops':{'value':2.}}
    proof=deepcopy(proof);proof['before_frame']['states']=deepcopy(states)
    (run/'evaluation_origin_verified.json').write_text(json.dumps(dict(origin_id=proof['origin_id'],verified_before_activation=True)))
    class Recorded:
        slots=[22440]
        receipts={('ue_energy_rf',2080):dict(app_id='xapp:ue_energy_rf',slot=2080,status='applied')}
        def __init__(self,*args):pass
        def sync(self):pass
        def window(self,keys,start,end,*,through):
            queries.append((start,end));return dict(slot=end,time_ms=end*.5,window_start_slot=start,
                window_slots=end-start,states={k:deepcopy(v)for k,v in states.items()if k[4:]in keys})
    monkeypatch.setattr('smo.window_metrics.RecordedWindows',Recorded)
    request=dict(id='d'*32,seed=7,records=records,app_ids=['xapp:ue_energy_rf'],observe_slots=20000,
        expected_origin=proof,runtime_recipe=worker_recipe(recipe,root,proof),expected_source_archive_sha256='c'*64)
    result=evaluate(request,run,2040)
    assert (40,2040)in queries and (2040,22040)in queries and all(a>=0 for a,b in queries)
    assert result['baseline_window']==dict(window_start_slot=40,slot=2040,window_slots=2000)
    assert result['after_window']==dict(window_start_slot=2040,slot=22040,window_slots=20000)
    assert result['observe_slots']==20000
    assert result['comparison_basis'].startswith('declared_independent_')
    assert result['rows'][0]['upper']==records[0]['ranges'][0]['upper']


def test_origin_metadata_compatibility_keeps_every_physical_check(tmp_path,monkeypatch):
    from test_smo_evaluation_origin import prepared_fixture
    from smo.evaluation_reference import origin_configuration,prefix_configuration_compatibility
    from smo.evaluation_origin_guard import OriginGuard,validate_proof
    prepared,config,rows,_,_,_=prepared_fixture(tmp_path,monkeypatch)
    config['manifest']['traffic_profile']=dict(path='/old/traffic.json',sha256='a'*64,overrides={'u':{'rate':1.}})
    config['ue_services_profile']=dict(path='/old/service.json',sha256='b'*64)
    proof=deepcopy(prepared['proof']);proof['configuration']=origin_configuration(config)
    proof.pop('origin_id');proof['origin_id']=digest(proof);original=deepcopy(proof)
    for after in (500,20000):
        actual=deepcopy(config);actual['manifest']['total_slots']=proof['baseline_slot']+after+400
        actual['manifest']['traffic_profile']['path']='/owned/environment/traffic.json'
        actual['ue_services_profile']['path']='/owned/environment/ue-services.json'
        guard=OriginGuard(proof,actual,Path(prepared['run_path']))
        for row in rows[:520]:guard.observe(row)
        assert guard.verified
        assert guard.configuration_compatibility['configuration_equal']
        assert set(guard.configuration_compatibility['ignored_metadata_fields'])>= {'manifest.traffic_profile.path','ue_services_profile.path'}
        validate_proof(proof);assert proof==original
        for pointer in ('traffic','traffic_hash','service_hash','seed'):
            bad=deepcopy(actual)
            if pointer=='traffic':bad['manifest']['traffic_profile']['overrides']['u']['rate']=2.
            if pointer=='traffic_hash':bad['manifest']['traffic_profile']['sha256']='c'*64
            if pointer=='service_hash':bad['ue_services_profile']['sha256']='d'*64
            if pointer=='seed':bad['manifest']['rng_seed']=8
            assert not prefix_configuration_compatibility(bad,proof['configuration'])['configuration_equal']
            with pytest.raises(ValueError,match='configuration differs'):OriginGuard(proof,bad,Path(prepared['run_path']))


def test_shorter_run_trajectory_requires_each_slot_position_velocity_and_ue_identity():
    from smo.evaluation_reference import prefix_configuration_compatibility,trajectory_horizon_compatibility
    old=dict(manifest=dict(total_slots=22440,slot_duration_ms=.5,ues=[dict(ue_id='ue_0',trajectory=dict(
        times_s=[0.,5.,11.22],positions_m=[[0.,0.,1.5],[7.5,0.,1.5],[-1.83,0.,1.5]]))]),
        configs={'sim':{'total_slots':22440}})
    short=deepcopy(old);short['manifest']['total_slots']=2940;short['configs']['sim']['total_slots']=2940
    short['manifest']['ues'][0]['trajectory']=dict(times_s=[0.,1.47],positions_m=[[0.,0.,1.5],[2.205,0.,1.5]])
    original=deepcopy((old,short))
    proof=prefix_configuration_compatibility(short,old,baseline_slot=2040)
    assert proof['configuration_equal']
    geometry=proof['trajectory_horizon_compatibility']
    assert geometry['checked_before_slots']==2040 and geometry['checked_through_slot']==2940
    assert geometry['position_velocity_checks']==2941 and geometry['maximum_speed_equal']
    assert 'configs.sim.total_slots'in proof['ignored_metadata_fields']
    assert (old,short)==original
    for kind in ('position','speed','ue_identity','interior_turn','dt'):
        bad=deepcopy(short);trajectory=bad['manifest']['ues'][0]['trajectory']
        if kind=='position':trajectory['positions_m'][0][0]+=.001
        if kind=='speed':trajectory['positions_m'][-1][0]+=.000001
        if kind=='ue_identity':bad['manifest']['ues'][0]['ue_id']='another_ue'
        if kind=='interior_turn':
            trajectory['times_s'].insert(1,.5);trajectory['positions_m'].insert(1,[.7,0.,1.5])
        if kind=='dt':bad['manifest']['slot_duration_ms']=.6
        assert not prefix_configuration_compatibility(bad,old,baseline_slot=2040)['configuration_equal']
        assert not trajectory_horizon_compatibility(bad['manifest'],old['manifest'],2040)['geometry_equal']


def test_trajectory_equivalence_uses_the_same_native_interpolation():
    pytest.importorskip('numpy')
    from phy.rt_sys.channel import position_at
    from smo.evaluation_reference import trajectory_horizon_compatibility
    old={'total_slots':22440,'slot_duration_ms':.5,'ues':[{'ue_id':'ue_0','trajectory':{
        'times_s':[0.,5.6,11.22],'positions_m':[[-11.4,-30.,1.5],[-3.,-30.,1.5],[-11.43,-30.,1.5]]}}]}
    short=deepcopy(old);short['total_slots']=2940
    short['ues'][0]['trajectory']={'times_s':[0.,1.47],'positions_m':[[-11.4,-30.,1.5],[-9.195,-30.,1.5]]}
    proof=trajectory_horizon_compatibility(short,old,2040);assert proof['geometry_equal']
    for slot in range(2941):
        a,av=position_at(short['ues'][0]['trajectory'],slot*.0005)
        b,bv=position_at(old['ues'][0]['trajectory'],slot*.0005)
        assert max(abs(a-b))<=proof['position_absolute_tolerance_m']
        assert max(abs(av-bv))<=proof['velocity_absolute_tolerance_mps']


def test_browser_cannot_supply_operator_recipe_profiles_or_origin():
    service, group = service_fixture()
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f'http://127.0.0.1:{server.server_port}'
    ordinary = dict(active_intent_ids=['test-intent'], slot=600,
                    observe_slots=500, show_dashed=False)
    injected = {
        'runtime_recipe': {'traffic_load': 9},
        'traffic_profile': '/tmp/browser-chosen.json',
        'ue_services_config': '/tmp/browser-chosen.json',
        'base_config': '/tmp/browser-chosen.json',
        'expected_origin': {'origin_id': 'browser-chosen'},
        'seed': 123,
    }
    try:
        for endpoint in ('/api/deployment/start', '/api/deployment/batch/start'):
            body = {**ordinary, **({'group_id': group['id']}
                                   if endpoint == '/api/deployment/start' else {})}
            for key, value in injected.items():
                request = Request(address + endpoint,
                                  data=json.dumps({**body, key: value}).encode(),
                                  headers={'Content-Type': 'application/json'})
                with pytest.raises(HTTPError) as error:
                    urlopen(request, timeout=5)
                assert error.value.code == 400
        assert service.deployments.calls == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
