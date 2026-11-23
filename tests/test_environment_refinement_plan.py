"""Versioned design/proof tests only; no actual experiment plan or remote run."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import environment_refinement_plan as planner
from experiments.environment_validation_plan import CANDIDATES, profile


@pytest.fixture
def design(tmp_path):
    parent_profile = profile(CANDIDATES[2])
    profile_path = tmp_path/'parent-c3.json'
    profile_path.write_bytes(planner.encoded(parent_profile))
    users = []
    for uid, rate in zip(planner.EMBB, (21_000_000, 18_000_000, 15_000_000)):
        users.append(dict(ue_id=uid, slice_id='embb', arrival_model='on_off', arrival_rate_bps=rate,
            burst_size_bytes=3300, on_duration_slots=40, off_duration_slots=16, start_offset_slots=0))
    users.append(dict(ue_id='ue_1', slice_id='voice', arrival_model='cbr', arrival_rate_bps=420000))
    users += [dict(ue_id=uid,slice_id='urllc',**parent_profile['ues'][uid])for uid in planner.URLLC]
    service=json.loads((planner.ROOT/planner.SERVICE).read_text())
    snapshot=dict(source_sha256=planner.worker.SOURCE_SHA256,
        manifest=dict(scene='munich', ues=[dict(ue_id=u['ue_id'])for u in users], cells=[{}, {}, {}],
            slot_duration_ms=.5,carrier_frequency_hz=3_500_000_000,bandwidth_hz=10_000_000,
            traffic_profile=dict(sha256=planner.worker.sha(profile_path))),
        configs=dict(ue=dict(ues=users),phy=dict(rt_sys=dict(ue_services=service))),
        ue_services_profile=dict(sha256=planner.worker.sha(planner.ROOT/planner.SERVICE)))
    snapshot_path=tmp_path/'parent-snapshot.json'
    snapshot_path.write_bytes(planner.encoded(snapshot))
    return dict(schema=planner.SCHEMA,study_id='test-refinement-r2',stage='pilot',
        recipes=[dict(id='bulk105-result400',embb_scale=1.05,mec_result_bytes=400)],
        assignments=[dict(worker_id='agx1',seed=20261207,recipe_ids=['bulk105-result400']),
                     dict(worker_id='agx2',seed=20261208,recipe_ids=['bulk105-result400'])],
        arms=['OFF','C','D'],allow_scout_on=True,include_native_reference=False,
        deadline_utc='2099-01-01T00:00:00+00:00',runner_sha256=planner.worker.sha(planner.worker.__file__),
        parent=dict(traffic_profile=str(profile_path),traffic_profile_sha256=planner.worker.sha(profile_path),
                    config_snapshot=str(snapshot_path),config_snapshot_sha256=planner.worker.sha(snapshot_path)))


def validation(design):
    result=deepcopy(design)
    result.update(stage='validation',arms=['OFF',*planner.APPS],include_native_reference=True,
                  discovery_seeds=[20261207,20261208])
    for assignment,seed in zip(result['assignments'],(20261211,20261212)):
        assignment['seed']=seed
    return result


def test_v4_adds_one_pinned_base_and_truthful_geometry_metadata(design):
    design.update(runner_version=4, runner_sha256=planner.worker.sha(planner.worker4.__file__))
    bundle = planner.build_bundle(design)
    base_name = 'profiles/'+design['study_id']+'_ue0_road0_phase065_base.json'
    frozen = json.loads((planner.ROOT/planner.worker4.BASE_CONFIG).read_text())
    assert bundle[base_name] == {**frozen,'ue_route_overrides':planner.worker4.ROUTE_OVERRIDE}
    pin = planner.hashlib.sha256(planner.encoded(bundle[base_name])).hexdigest()
    for wid in ('agx1','agx2'):
        plan = bundle['pilot/'+wid+'.json']
        assert plan['schema'] == planner.worker4.SCHEMA
        assert plan['require_on_admission'] is True
        assert all(c['base_config_sha256']==pin for c in plan['cases'])
        planner.worker4.validate_plan(plan, plan['plan_sha256'])
    assert bundle['protocol.json']['unchanged_topology']['geometry_route_radio_overrides_changed']
    assert bundle['protocol.json']['ON_admission_marker']=='ON_ADMISSION_ALLOWED'
    assert bundle['recipe-differences.json']['changes'][design['recipes'][0]['id']]['physical_topology_unchanged'] is False


@pytest.mark.parametrize('version',[2,3])
def test_pre_v4_builder_json_outputs_are_byte_identical_except_self_manifest(design,version):
    import importlib.util
    path=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122/protocol/environment_refinement_plan-pre-v4-geometry-snapshot.py')
    spec=importlib.util.spec_from_file_location('pre_v4_builder',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    runner=planner.worker if version==2 else planner.worker3
    design.update(runner_version=version,runner_sha256=planner.worker.sha(runner.__file__))
    before=old.build_bundle(design,root=planner.ROOT)
    after=planner.build_bundle(design)
    assert set(before)==set(after)
    assert all(planner.encoded(before[k])==planner.encoded(after[k])for k in before if k!='manifest.json')
    assert before['manifest.json']['files_sha256']==after['manifest.json']['files_sha256']


@pytest.mark.parametrize('version',[2,3])
def test_legacy_runner_cannot_silently_consume_geometry_override(design,version):
    runner=planner.worker if version==2 else planner.worker3
    design.update(runner_version=version,runner_sha256=planner.worker.sha(runner.__file__),
                  ue0_route_override=dict(route_index=0,start_fraction=.65))
    with pytest.raises(ValueError,match='v4 geometry'):
        planner.validate_design(design)


def test_scout_on_needs_explicit_environment_discovery_flag(design):
    planner.validate_design(design)
    design['allow_scout_on']=False
    with pytest.raises(ValueError,match='allow_scout_on'):
        planner.validate_design(design)
    design['arms']=['OFF']
    planner.validate_design(design)


@pytest.mark.parametrize('mapping',[
    {'ue_0':1,'ue_4':1.5},
    {'ue_0':1,'ue_4':1,'ue_8':1,'ue_1':1},
    {'ue_0':1,'ue_4':1,'ue_8':0.49},
    {'ue_0':1,'ue_4':1,'ue_8':2.000001},
    {'ue_0':1,'ue_4':1,'ue_8':float('inf')},
    {'ue_0':1,'ue_4':float('nan'),'ue_8':1},
    {'ue_0':True,'ue_4':1,'ue_8':1},
    {'ue_0':1,'ue_4':'1','ue_8':1},None,
])
def test_per_ue_scale_map_is_complete_bounded_finite_and_not_boolean(design,mapping):
    design['recipes'][0].update(embb_scale=1,embb_scale_by_ue=mapping)
    with pytest.raises(ValueError):
        planner.validate_design(design)


def test_per_ue_and_nontrivial_global_scaling_cannot_be_combined(design):
    design['recipes'][0]['embb_scale_by_ue']={'ue_0':1.5,'ue_4':1,'ue_8':.5}
    with pytest.raises(ValueError,match='cannot combine'):
        planner.validate_design(design)
    design['recipes'][0].pop('embb_scale')
    planner.validate_design(design)
    assert planner.embb_scales(design['recipes'][0])=={'ue_0':1.5,'ue_4':1,'ue_8':.5}


@pytest.mark.parametrize('fraction',[None,True,False,0,.5,.75,.81,1.1,float('nan'),float('inf'),'0.8'])
def test_initial_task_routing_accepts_only_explicit_supported_fractions(design,fraction):
    design['recipes'][0]['initial_ue4_offload_fraction']=fraction
    with pytest.raises(ValueError,match='Initial UE4 offload fraction'):
        planner.validate_design(design)


@pytest.mark.parametrize('fraction',[.8,1,1.])
def test_supported_initial_routing_does_not_change_task_shape_or_UE0(design,fraction):
    design['recipes'][0].update(initial_ue4_offload_fraction=fraction,mec_result_bytes=200)
    planner.validate_design(design)
    parent,traffic,_=planner.load_parent(design)
    service=json.loads((planner.ROOT/planner.SERVICE).read_text())
    before=deepcopy(service)
    spec=design['recipes'][0]
    profile,changed,effective=planner.make_recipe(design['study_id'],spec,parent['traffic_profile'],traffic,service)
    default={k:v for k,v in spec.items()if k!='initial_ue4_offload_fraction'}
    original_profile,original_service,original_effective=planner.make_recipe(design['study_id'],default,
        parent['traffic_profile'],traffic,service)
    assert effective==original_effective and profile['ues']==original_profile['ues']
    assert changed['mec']['tasks_by_ue']['ue_0']==before['mec']['tasks_by_ue']['ue_0']
    assert service==before
    difference=planner.raw_diff(service,changed)
    if fraction==1:
        assert difference==[]
        assert planner.encoded(profile)==planner.encoded(original_profile)
        assert planner.encoded(changed)==planner.encoded(original_service)
    else:
        assert difference==[dict(path='/mec/tasks_by_ue/ue_4/offload_fraction',before=1.,after=.8)]
        assert 'INITIAL TASK ROUTING' in profile['provenance']
        assert 'arrival-only' not in profile['provenance']
    tasks=changed['mec']['tasks_by_ue']
    assert all((p['interval_ms'],p['compute_gflop'],p['deadline_ms'],p['input_bytes'],p['result_bytes'])
               ==(50,.05,100,1000,200)for p in tasks.values())
    assert planner.initial_mec_demand(changed)=={'ue_0':.5,'ue_4':fraction}


def test_initial_routing_preserves_production_arrivals_and_labels_expected_work(design):
    design['recipes'][0].update(embb_scale=1,embb_scale_by_ue={'ue_0':.5,'ue_4':1.85,'ue_8':.85},
        mec_result_bytes=200,urllc_pattern='rare6')
    parent,traffic,_=planner.load_parent(design)
    service=json.loads((planner.ROOT/planner.SERVICE).read_text())
    spec=design['recipes'][0]
    _,_,ordinary=planner.make_recipe(design['study_id'],spec,parent['traffic_profile'],traffic,service)
    spec['initial_ue4_offload_fraction']=.8
    _,_,routing=planner.make_recipe(design['study_id'],spec,parent['traffic_profile'],traffic,service)
    assert planner.sampled_load(ordinary,observe_slots=4000)==planner.sampled_load(routing,observe_slots=4000)
    files=planner.build_bundle(design);rid=spec['id']
    check=files['actual-load-checks.json']['recipes'][rid]
    assert check['initial_expected_mec_demand_gflops']==1.3
    assert check['initial_task_routing']['change_kind']=='INITIAL TASK ROUTING'
    assert check['initial_task_routing']['ue0_initial_offload_fraction']==.5
    assert check['initial_task_routing']['ue4_initial_expected_demand_exceeds_F_budget'] is True
    assert check['initial_task_routing']['trained_actions_changed'] is False
    assert files['recipe-differences.json']['changes'][rid]['services']==[
        dict(path='/mec/tasks_by_ue/ue_4/offload_fraction',before=1.,after=.8)]
    assert 'INITIAL TASK ROUTING' in files['protocol.json']['unchanged_topology']['interpretation']


@pytest.mark.parametrize('version,parallel',[(2,2),(2,3),(3,4),(3,True),(4,1)])
def test_runner_version_and_parallel_limits_are_explicit(design,version,parallel):
    design.update(runner_version=version,max_parallel=parallel)
    with pytest.raises(ValueError):
        planner.validate_design(design)


def test_v3_requires_its_actual_wrapper_pin(design):
    design['runner_version']=3
    with pytest.raises(ValueError,match='selected worker'):
        planner.validate_design(design)
    design['runner_sha256']=planner.worker.sha(planner.worker3.__file__)
    planner.validate_design(design)
    assert planner.selected_runner(design)[2]==2


def test_v2_default_and_v3_plans_have_identical_physical_cases(design):
    serial=planner.build_bundle(design)
    parallel=deepcopy(design)
    parallel.update(runner_version=3,max_parallel=2,runner_sha256=planner.worker.sha(planner.worker3.__file__))
    concurrent=planner.build_bundle(parallel)
    for host in ('agx1','agx2'):
        a,b=serial['pilot/'+host+'.json'],concurrent['pilot/'+host+'.json']
        assert a['schema']==planner.worker.SCHEMA and a['max_parallel']==1
        assert b['schema']==planner.worker3.SCHEMA and b['max_parallel']==2
        assert b['runner_path'].endswith('/runner/worker-v3.py')
        assert a['cases']==b['cases'] and a['source_sha256']==b['source_sha256']
        assert a['frozen_models']==b['frozen_models']
        planner.worker3.validate_plan(b,b['plan_sha256'],plan_path=b['active_plan_path'])
    assert serial['recipe-differences.json']==concurrent['recipe-differences.json']
    assert serial['actual-load-checks.json']==concurrent['actual-load-checks.json']


@pytest.mark.parametrize('change',[
    lambda d:d['recipes'][0].update(embb_scale=1.2),
    lambda d:d['recipes'][0].update(mec_result_bytes=1600),
    lambda d:d['recipes'][0].update(server_gflops=1),
    lambda d:d['recipes'][0].update(interval_ms=25),
    lambda d:d['recipes'][0].update(urllc_pattern='invented'),
    lambda d:d['recipes'][0].update(id='native_reference'),
    lambda d:d.update(study_id='../prior'),
    lambda d:d.update(runner_sha256='f'*64),
    lambda d:d.update(arms=['C','OFF']),
    lambda d:d.update(assignments=d['assignments'][:1]),
])
def test_forbidden_physics_task_policy_and_assignment_changes_rejected(design,change):
    change(design)
    with pytest.raises(ValueError):
        planner.validate_design(design)


def test_parent_actual_c3_and_service_are_pinned(design):
    parent,traffic,_=planner.load_parent(design)
    assert traffic['ue_0']['arrival_rate_bps']==21_000_000
    assert parent['traffic_profile']['ues']['ue_6']['arrival_rate_bps']==6_592_000
    path=Path(design['parent']['config_snapshot'])
    changed=json.loads(path.read_text())
    changed['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['ue_0']['compute_gflop']=.1
    path.write_bytes(planner.encoded(changed))
    with pytest.raises(ValueError,match='checksum'):
        planner.load_parent(design)
    design['parent']['config_snapshot_sha256']=planner.worker.sha(path)
    with pytest.raises(ValueError,match='Recorded parent services'):
        planner.load_parent(design)


def test_effective_recipe_only_changes_three_embb_streams_and_two_result_sizes(design):
    parent,traffic,_=planner.load_parent(design)
    original_service=json.loads((planner.ROOT/planner.SERVICE).read_text())
    arrival,services,effective=planner.make_recipe(design['study_id'],design['recipes'][0],
        parent['traffic_profile'],traffic,original_service)
    for uid in planner.URLLC:
        assert arrival['ues'][uid]==parent['traffic_profile']['ues'][uid]
    assert effective['ue_1']==traffic['ue_1']
    assert effective['ue_0']['arrival_rate_bps']==22_050_000
    assert effective['ue_0']['burst_size_bytes']==3465
    differences=planner.raw_diff(original_service,services)
    assert {d['path']for d in differences}=={'/mec/tasks_by_ue/ue_0/result_bytes','/mec/tasks_by_ue/ue_4/result_bytes'}
    assert all(d['before']==200 and d['after']==400 for d in differences)


@pytest.mark.parametrize('scale',[1.,1.05,1.1])
@pytest.mark.parametrize('result_bytes',[200,400,800])
def test_every_permitted_recipe_retains_task_shape_and_URLLC(design,scale,result_bytes):
    parent,traffic,_=planner.load_parent(design)
    service=json.loads((planner.ROOT/planner.SERVICE).read_text())
    spec=dict(id='explicit-test-recipe',embb_scale=scale,mec_result_bytes=result_bytes)
    arrivals,changed,effective=planner.make_recipe(design['study_id'],spec,parent['traffic_profile'],traffic,service)
    for uid in planner.URLLC:
        assert arrivals['ues'][uid]==parent['traffic_profile']['ues'][uid]
    for uid in ('ue_0','ue_4'):
        task=changed['mec']['tasks_by_ue'][uid]
        assert (task['interval_ms'],task['compute_gflop'],task['deadline_ms'],task['input_bytes'])==(50,.05,100,1000)
        assert task['result_bytes']==result_bytes
    assert changed['drx']==service['drx'] and changed['power']==service['power']
    assert changed['mec']['server_gflops']==service['mec']['server_gflops']


def test_bundle_proves_actual_load_and_v2_pairing_without_creating_files(design,tmp_path):
    files=planner.build_bundle(design)
    assert not (tmp_path/'output').exists()
    profile_names=[n for n in files if n.startswith('profiles/')]
    assert len(profile_names)==2 and all(design['study_id']in n for n in profile_names)
    for wid in ('agx1','agx2'):
        p=files['pilot/'+wid+'.json']
        assert p['schema']==planner.worker.SCHEMA
        assert [c['arm']for c in p['cases']]==['OFF','C','D']
        assert p['output_root'].endswith('/test-refinement-r2/pilot')
        assert p['active_plan_path'].endswith('/plans/test-refinement-r2/pilot.json')
        assert p['runner_path'].endswith('/runner/worker-v2.py')
        assert all(c['traffic_load']==1.5 for c in p['cases'])
        planner.worker.validate_plan(p,p['plan_sha256'],plan_path=p['active_plan_path'])
    actual=files['actual-load-checks.json']['recipes']['bulk105-result400']
    assert actual['exact_parent_URLLC_and_voice'] is True
    assert actual['mec_result_packets_per_task']==2
    for uid in planner.EMBB:
        observed=actual['actual'][uid]['whole']['actual_byte_change_percent']
        assert 4.8<observed<5.2
    # Hash every JSON payload before any disk write, including both profile kinds.
    import hashlib
    for name,pin in files['manifest.json']['files_sha256'].items():
        assert hashlib.sha256(planner.encoded(files[name])).hexdigest()==pin
    planner.write_bundle(files,tmp_path/'output')
    planner.write_bundle(files,tmp_path/'output')  # exact idempotent replay only
    path=tmp_path/'output'/profile_names[0]
    retained=path.read_bytes()
    bad=deepcopy(files);bad[profile_names[0]]['profile_id']='changed'
    with pytest.raises(ValueError,match='Never overwrite'):
        planner.write_bundle(bad,tmp_path/'output')
    assert path.read_bytes()==retained


@pytest.mark.parametrize('pattern',['mixed4','mixed5','rare6'])
def test_mixed_recipe_metadata_claims_conservation_without_claiming_same_trace(design,pattern):
    design['recipes'][0].update(urllc_pattern=pattern,mec_result_bytes=200)
    files=planner.build_bundle(design)
    rid=design['recipes'][0]['id']
    diff=files['recipe-differences.json']['changes'][rid]
    assert diff['URLLC_unchanged'] is False
    assert diff['URLLC_registered_window_bytes_packets_sizes_preserved'] is True
    assert diff['mec_result_payload_changed'] is False
    check=files['actual-load-checks.json']['recipes'][rid]
    assert check['exact_parent_URLLC_and_voice'] is False
    assert check['exact_parent_URLLC_trace'] is False and check['exact_parent_voice'] is True
    assert check['registered_URLLC_window_bytes_packets_sizes_preserved'] is True
    assert check['URLLC_timing']['first_high_global_slot']==1
    for uid,mean in planner.urllc_patterns.MEAN_PACKET_BYTES.items():
        for name,count in [('before',2000),('pilot_after',2000),('validation_after',4000)]:
            row=check['actual'][uid][name]
            assert row['bytes']==count*mean and row['packets']==count and row['packet_sizes_bytes']==[mean]
    plans=[files['pilot/'+w+'.json']for w in ('agx1','agx2')]
    assert all(p['source_sha256']==planner.worker.SOURCE_SHA256 for p in plans)
    assert plans[0]['frozen_models']==plans[1]['frozen_models']
    for value in files.values():
        if isinstance(value,dict) and value.get('profile_id'):
            assert 'Different temporal shape' in value['provenance']


def test_asymmetric_rare6_proof_samples_each_ue_without_changing_topology_or_services(design):
    mapping={'ue_0':1.25,'ue_4':1.5,'ue_8':.5}
    design['recipes'][0].update(embb_scale=1,embb_scale_by_ue=mapping,
                               urllc_pattern='rare6',mec_result_bytes=200)
    design['observe_slots']=4000
    files=planner.build_bundle(design)
    rid=design['recipes'][0]['id']
    check=files['actual-load-checks.json']['recipes'][rid]
    assert check['nominal_embb_multiplier'] is None
    assert check['nominal_embb_multiplier_by_ue']==mapping
    assert check['physical_topology_unchanged'] is True
    assert files['protocol.json']['unchanged_topology']['ue_count']==6
    assert files['protocol.json']['unchanged_topology']['cell_count']==3
    assert files['recipe-differences.json']['changes'][rid]['services']==[]
    for uid in planner.EMBB:
        for row in check['actual'][uid].values():
            assert row['nominal_multiplier']==mapping[uid]
            assert row['bytes']==row['expected_bytes'] and row['packets']==row['expected_packets']
        delta=check['actual'][uid]['after']['actual_byte_change_percent']
        assert (delta>0) if mapping[uid]>1 else (delta<0)
    for uid,mean in planner.urllc_patterns.MEAN_PACKET_BYTES.items():
        after=check['actual'][uid]['after']
        assert after['bytes']==4000*mean and after['packets']==4000
        assert after['packet_sizes_bytes']==[mean]
    assert check['exact_parent_voice'] is True


@pytest.mark.parametrize('hotspot_scale',[1.51,1.75,1.85,2.0])
def test_extended_hotspot_bound_changes_only_ordinary_offered_work(design,hotspot_scale):
    mapping={'ue_0':.5,'ue_4':hotspot_scale,'ue_8':.85}
    design['recipes'][0].update(embb_scale=1,embb_scale_by_ue=mapping,
                               urllc_pattern='rare6',mec_result_bytes=200)
    design['observe_slots']=4000
    files=planner.build_bundle(design)
    rid=design['recipes'][0]['id']
    check=files['actual-load-checks.json']['recipes'][rid]
    assert planner.embb_scales(design['recipes'][0])==mapping
    assert files['recipe-differences.json']['changes'][rid]['services']==[]
    assert check['physical_topology_unchanged'] is True
    assert check['exact_parent_voice'] is True
    for uid in planner.EMBB:
        for row in check['actual'][uid].values():
            assert row['nominal_multiplier']==mapping[uid]
            assert row['bytes']==row['expected_bytes']
            assert row['packets']==row['expected_packets']
        delta=check['actual'][uid]['after']['actual_byte_change_percent']
        assert (delta>0) if uid=='ue_4' else (delta<0)
    for uid,mean in planner.urllc_patterns.MEAN_PACKET_BYTES.items():
        row=check['actual'][uid]['after']
        assert row['bytes']==4000*mean and row['packets']==4000
        assert row['packet_sizes_bytes']==[mean]


def test_validation_requires_all_six_on_same_recipe_and_new_seed_history(design):
    value=validation(design)
    planner.validate_design(value)
    for arms in (['OFF','C','D'],['OFF','A','B','C','D','E']):
        bad={**value,'arms':arms}
        with pytest.raises(ValueError,match='all six'):
            planner.validate_design(bad)
    bad=deepcopy(value);bad['assignments'][0]['seed']=20261203
    with pytest.raises(ValueError,match='already used'):
        planner.validate_design(bad)
    bad=deepcopy(value);bad['assignments'][0]['seed']=20261207
    with pytest.raises(ValueError,match='already used'):
        planner.validate_design(bad)
    bad=deepcopy(value);bad.pop('discovery_seeds')
    with pytest.raises(ValueError,match='discovery history'):
        planner.validate_design(bad)


@pytest.mark.parametrize('stage,observe',[
    ('pilot',1999),('pilot',3000),('pilot',6000),('pilot',20000),('pilot',True),('pilot',4000.),
    ('validation',2000),('validation',3000),('validation',19999),('validation',20001),
    ('validation',True),('validation','4000'),('validation',20000.),
])
def test_only_explicit_supported_observation_windows_are_accepted(design,stage,observe):
    value=validation(design) if stage=='validation'else design
    value['observe_slots']=observe
    with pytest.raises(ValueError,match='Pilot AFTER'):
        planner.validate_design(value)


def test_pilot_can_use_explicit_two_second_after_with_unchanged_before_origin(design):
    design['observe_slots']=4000
    files=planner.build_bundle(design)
    protocol=files['protocol.json']
    assert protocol['stage']=='pilot' and protocol['observe_slots']==4000
    assert (protocol['baseline_slot'],protocol['origin_observe_slots'],protocol['tail_slots'],protocol['total_slots'])==(2040,2000,400,6440)
    for host in ('agx1','agx2'):
        for case in files['pilot/'+host+'.json']['cases']:
            assert (case['slots'],case['baseline_slot'],case['origin_observe_slots'],case['observe_slots'])==(6440,2040,2000,4000)
    proof=files['actual-load-checks.json']['recipes'][design['recipes'][0]['id']]['actual']
    assert all(rows['before']['sampled_slots']==2000 and rows['after']['sampled_slots']==4000 for rows in proof.values())


def test_validation_ten_second_window_preserves_origin_physics_and_rare6_means(design):
    value=validation(design)
    value.update(observe_slots=20000,runner_version=3,max_parallel=2,
        runner_sha256=planner.worker.sha(planner.worker3.__file__))
    value['recipes'][0].update(embb_scale=1,embb_scale_by_ue={'ue_0':.5,'ue_4':1.85,'ue_8':.85},
        mec_result_bytes=200,urllc_pattern='rare6',initial_ue4_offload_fraction=.8)
    files=planner.build_bundle(value);rid=value['recipes'][0]['id'];protocol=files['protocol.json']
    assert (protocol['baseline_slot'],protocol['origin_observe_slots'],protocol['observe_slots'],
            protocol['tail_slots'],protocol['total_slots'])==(2040,2000,20000,400,22440)
    for host in ['agx1','agx2']:
        plan=files['validation/'+host+'.json']
        planner.worker3.validate_plan(plan,plan['plan_sha256'],plan_path=plan['active_plan_path'])
        assert all((c['slots'],c['baseline_slot'],c['origin_observe_slots'],c['observe_slots'])
                   ==(22440,2040,2000,20000)for c in plan['cases'])
    check=files['actual-load-checks.json']['recipes'][rid]
    assert check['URLLC_timing']['requested_after_window']==dict(
        first_slot=2041,last_slot=22040,sampled_slots=20000,duration_s=10.)
    for uid,packet_bytes in planner.urllc_patterns.MEAN_PACKET_BYTES.items():
        after=check['actual'][uid]['after'];cycle=check['actual'][uid]['cycle']
        assert (after['sampled_slots'],after['packets'],after['bytes'])==(20000,20000,20000*packet_bytes)
        assert after['packet_sizes_bytes']==[packet_bytes]
        assert after['mean_mbps']==cycle['mean_mbps']
    assert check['initial_expected_mec_demand_gflops']==1.3
    assert check['fixed_mec_task_shape']==dict(interval_ms=50,compute_gflop=.05,deadline_ms=100,input_bytes=1000)


def test_validation_builds_16_runs_and_preserves_native_reference_separately(design):
    value=validation(design)
    value['recipes'][0].update(embb_scale=1.1,mec_result_bytes=800)
    files=planner.build_bundle(value)
    assert files['protocol.json']['physical_run_count']==16
    for wid in ('agx1','agx2'):
        p=files['validation/'+wid+'.json']
        assert len(p['cases'])==8
        native,*treatment=p['cases']
        assert native['scene']=='native_reference' and native['traffic_profile']is None
        assert native['traffic_load']==1 and native['ue_services_config'].endswith('/ue_services.json')
        assert [c['arm']for c in treatment]==['OFF',*planner.APPS]
        assert all(c['observe_slots']==4000 and c['origin_observe_slots']==2000 for c in p['cases'])
        assert len({c['ue_services_config_sha256']for c in treatment})==1
        assert len({c['traffic_profile_sha256']for c in treatment})==1
        assert native['origin_key']!=treatment[0]['origin_key']
    actual=files['actual-load-checks.json']['recipes']['bulk105-result400']
    assert actual['mec_result_packets_per_task']==4
    assert 9.8<actual['actual']['ue_0']['whole']['actual_byte_change_percent']<10.2


def test_file_paths_cannot_escape_or_alias_prior_bundle(tmp_path):
    with pytest.raises(ValueError,match='escapes'):
        planner.write_bundle({'../escaped.json':{}},tmp_path/'output')
    prior=tmp_path/'prior';prior.mkdir();(prior/'keep.json').write_text('retained')
    output=tmp_path/'output';output.mkdir();(output/'alias').symlink_to(prior,target_is_directory=True)
    with pytest.raises(ValueError,match='escapes'):
        planner.write_bundle({'alias/new.json':{}},output)
    assert (prior/'keep.json').read_text()=='retained'
