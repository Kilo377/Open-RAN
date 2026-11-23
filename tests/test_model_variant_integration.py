"""Temporary synthetic metadata only; no selected-model install or simulator."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ric.near_rt.pools import build_pool_xapp
import ric.near_rt.pools.chongyu.ue_energy_rf as legacy
import ric.near_rt.pools.chongyu.ue_energy_rf_action_sets as extended
from smo import model_release as release

ROOT=Path(__file__).resolve().parents[1]


def operator_audit(name):
    directory=Path(os.environ.get('ORAN_INTEGRATION_AUDIT_DIR',ROOT.parent/'audit'))
    path=directory/name
    if not path.is_file():
        pytest.skip('Archived operator audit unavailable; set ORAN_INTEGRATION_AUDIT_DIR to run this preservation check')
    return json.loads(path.read_text())


def item(variant=None,parameters=None):
    result=dict(pool='chongyu',implementation_id='ue_energy_rf',instance_id='ue_energy_rf',
        cell_ids=['cell_0'],parameters=parameters or {})
    if variant is not None:result['implementation_variant']=variant
    return result


def user():
    return NS(ue_id='ue_0',serving_cell_id='cell_0',queue_bytes=1000.,hol_delay_ms=2.,
        throughput_mbps=1.,offered_rate_bps=1e6,reference_sinr_db=10.,
        ue_task_metrics={'pending':0},ue_task_cohorts=[],ue_service_policy=deepcopy(legacy.ACTIONS[1]),
        kpi_metadata={'ue_energy_total_j':{'produced_at_slot':2041}})


@pytest.mark.parametrize('version,last,on',[('extended24',23,1.),('balanced42',41,10.)])
def test_factory_default_preserves_legacy_and_extended_executes_final_arm(version,last,on):
    old=build_pool_xapp(item(),'chongyu')
    assert type(old) is legacy.UeEnergyRF and old.runtime_factory.endswith('ue_energy_rf:UeEnergyRF')
    new=build_pool_xapp(item(version,{'action_set_version':version,'deploy_drx':True}),'chongyu')
    assert type(new) is extended.UeEnergyRF and len(new.actions)==last+1
    assert new.implementation_id==old.implementation_id=='ue_energy_rf'
    new.mode='frozen';new.select(['cell_0'],{'cell_0':1})
    new.learner.predict=lambda rows:[[100.-i,1.,0.,20.]for i in range(len(rows))]
    obs=NS(slot_id=2041,ue_observations=[user()],control_ownership={'cell_0':{}})
    action=new.run(obs,NS()).config_actions[0]
    assert action.ue_drx_configs['ue_0']==new.actions[last]['drx']
    assert action.ue_drx_configs['ue_0']['on_duration_ms']==on
    assert action.ue_drx_configs['ue_0']['inactivity_ms']==0.
    assert action.ue_offload_fraction['ue_0']==1.
    assert new.last_decisions[0]['action_set_version']==version


@pytest.mark.parametrize('variant,version',[(None,'extended24'),('legacy9','extended24'),
    ('extended24','legacy9'),('extended24',None),('arbitrary','extended24'),
    (None,'balanced42'),('legacy9','balanced42'),('extended24','balanced42'),
    ('balanced42','extended24'),('balanced42','legacy9'),('balanced42',None)])
def test_variant_mismatch_cannot_fall_back_to_legacy(variant,version):
    parameters={'action_set_version':version}if version is not None else{}
    with pytest.raises(ValueError):build_pool_xapp(item(variant,parameters),'chongyu')


def test_catalogue_and_new_policy_are_exact_reviewed_v4_bytes():
    assert release.energy_catalogue('extended24')==extended.actions_for_version('extended24')
    assert release.energy_catalogue('legacy9')==legacy.ACTIONS
    assert release.catalogue_sha256('extended24')=='9279436f9f4b32536ffa780ffe1a7350c08e355501b9e485bbe07734e578b380'
    assert release.energy_catalogue('balanced42')==extended.actions_for_version('balanced42')
    assert release.catalogue_sha256('balanced42')=='2dcb2568cd66fc9b0640c871a47a8da916b5562c33320c96c88c2d4769c9bd20'
    assert release.sha(ROOT/release.VARIANT_MODULE)=='2406c5912b10f312d8f1c403d39c7fa78d1b00814e53b036cb0ff78a3b10263b'


@pytest.fixture(params=['extended24','balanced42'])
def synthetic_release(tmp_path,request):
    # These are original released weights copied into a pytest-only fixture;
    # no new candidate is selected, published or installed in the staging tree.
    origin=ROOT/'artifacts/xapps/chongyu/20261117'
    data=json.loads((origin/'release.json').read_text());data=deepcopy(data)
    data.update(release='20991231',synthetic_fixture=True)
    data['files_sha256']={}
    version=request.param
    entry=data['models']['x01'];entry['runtime_parameters']['action_set_version']=version
    entry.update(implementation_variant=version,action_catalogue=release.energy_catalogue(version),
        action_catalogue_sha256=release.catalogue_sha256(version))
    for base in ('artifacts/xapps/chongyu','outputs/trained_xapps'):
        bundle=tmp_path/base/data['release'];(bundle/'models').mkdir(parents=True)
        for name,entry in data['models'].items():
            relative=entry['path'];raw=(origin/relative).read_bytes();(bundle/relative).write_bytes(raw)
            data['files_sha256'][relative]=hashlib.sha256(raw).hexdigest()
        (bundle/'release.json').write_text(json.dumps(data))
    installer=tmp_path/'scripts/orin/install-trained-xapps.py';installer.parent.mkdir(parents=True)
    installer.write_bytes((ROOT/'scripts/orin/install-trained-xapps.py').read_bytes())
    variant=tmp_path/release.VARIANT_MODULE;variant.parent.mkdir(parents=True)
    variant.write_bytes((ROOT/release.VARIANT_MODULE).read_bytes())
    descriptor=dict(id=data['release'],manifest_sha256=release.sha(tmp_path/'artifacts/xapps/chongyu'/data['release']/'release.json'),
        action_set_version=version,action_catalogue_sha256=release.catalogue_sha256(version))
    recipe=dict(model_release=descriptor,frozen_models={
        'artifacts/xapps/chongyu/'+data['release']+'/'+e['path']:e['sha256']for e in data['models'].values()})
    return tmp_path,data,recipe


def test_release_pins_model_manifest_action_set_and_install_together(synthetic_release):
    root,data,recipe=synthetic_release
    assert release.verify_release(recipe,root)['release']=='20991231'
    assert release.verify_release(recipe,root,installed=True)['release']=='20991231'
    bad=deepcopy(recipe);bad['model_release']['action_set_version']='legacy9'
    bad['model_release']['action_catalogue_sha256']=release.catalogue_sha256('legacy9')
    with pytest.raises(ValueError):release.verify_release(bad,root)
    if recipe['model_release']['action_set_version']=='balanced42':
        bad=deepcopy(recipe);bad['model_release']['action_set_version']='extended24'
        bad['model_release']['action_catalogue_sha256']=release.catalogue_sha256('extended24')
        with pytest.raises(ValueError):release.verify_release(bad,root)
    bad=deepcopy(recipe);bad['model_release']['manifest_sha256']='f'*64
    with pytest.raises(ValueError):release.verify_release(bad,root)
    bad=deepcopy(recipe);key=next(iter(bad['frozen_models']));bad['frozen_models'][key]='f'*64
    with pytest.raises(ValueError):release.verify_release(bad,root)


def test_hot_configuration_routes_extended_factory_and_rejects_default_legacy(synthetic_release,monkeypatch):
    from experiments.hot_xapps import configure
    root,data,recipe=synthetic_release
    version=recipe['model_release']['action_set_version']
    args=NS(xapp_pool='chongyu',seed=341710,xapp_model_dir=root/'outputs/trained_xapps/20991231',
        energy_xapp_action_set_version=version,xapps_start='',energy_xapp_deploy_drx=True)
    manifest=dict(ues=[{'ue_id':'ue_'+str(i)}for i in(0,1,2,4,6,8)],
        cells=[{'cell_id':'cell_0'}],slot_duration_ms=.5,total_slots=2940)
    configs={'ric':{},'phy':{'rt_sys':{}}}
    configure(configs,manifest,args,root/'outputs/synthetic-run',root)
    config=next(a for a in configs['ric']['xapps']if a['implementation_id']=='ue_energy_rf')
    assert config['implementation_variant']==version
    assert config['parameters']['action_set_version']==version
    monkeypatch.setattr(extended,'ROOT',root)
    app=build_pool_xapp(config,'chongyu')
    assert type(app)is extended.UeEnergyRF and app.mode=='frozen' and len(app.actions)==len(release.energy_catalogue(version))
    args.energy_xapp_action_set_version='legacy9'
    with pytest.raises(ValueError):configure({'ric':{},'phy':{'rt_sys':{}}},manifest,args,root/'outputs/another',root)
    if version=='balanced42':
        args.energy_xapp_action_set_version='extended24'
        with pytest.raises(ValueError):configure({'ric':{},'phy':{'rt_sys':{}}},manifest,args,root/'outputs/wrong24',root)


def test_prefix_and_preview_reject_model_or_variant_mismatch(synthetic_release):
    _,_,recipe=synthetic_release
    version=recipe['model_release']['action_set_version']
    proof={'configuration':{'configs':{'ric':{'xapps':[{
        'implementation_id':'ue_energy_rf','implementation_variant':version,
        'parameters':{'action_set_version':version}}]}}}}
    release.validate_origin_variant(recipe,proof)
    proof['configuration']['configs']['ric']['xapps'][0]['implementation_variant']='legacy9'
    with pytest.raises(ValueError):release.validate_origin_variant(recipe,proof)
    preview={'apps':[dict(app_id='xapp:'+name,model_sha256=pin,mode='cold_activation')
        for name,pin in release.expected_model_pins(recipe).items()]}
    energy=next(a for a in preview['apps']if a['app_id']=='xapp:ue_energy_rf')
    energy.update(action_set_version=version,implementation_variant=version,
        action_catalogue_sha256=release.catalogue_sha256(version))
    release.validate_preview(recipe,preview)
    energy['implementation_variant']='legacy9'
    with pytest.raises(ValueError):release.validate_preview(recipe,preview)
    energy['implementation_variant']=version;energy['model_sha256']='f'*64
    with pytest.raises(ValueError):release.validate_preview(recipe,preview)


@pytest.mark.parametrize('after_slots',[500,20000])
def test_deployment_passes_real_release_dir_and_action_flag_with_fixed_before(synthetic_release,monkeypatch,after_slots):
    from smo import deployment_worker,runtime_recipe
    root,_,recipe=synthetic_release
    recipe.update(seed=20261214,baseline_slot=2040,tail_slots=400,origin_observe_slots=2000,traffic_load=1.5,
        profiles={k:{'path':str(root/(k+'.json'))}for k in runtime_recipe.PROFILE_NAMES})
    monkeypatch.setattr(runtime_recipe,'validate_worker_recipe',lambda *a:None)
    request=dict(id='a'*32,seed=20261214,app_ids=['xapp:ue_energy_rf'],observe_slots=after_slots,
        runtime_recipe=recipe,expected_source_archive_sha256='b'*64)
    _,_,baseline,total,argv=deployment_worker.plan(request,root)
    assert baseline==2040 and total==2040+400+after_slots
    assert argv[argv.index('--xapp-model-dir')+1]==str(root/'outputs/trained_xapps/20991231')
    assert argv[argv.index('--energy-xapp-action-set-version')+1]==recipe['model_release']['action_set_version']
    assert argv[argv.index('--xapps-start')+1]==''


def test_recipe_schema_accepts_old317_and_exact_new_member_count(synthetic_release):
    from smo import runtime_recipe
    root,_,new=synthetic_release
    old_models=json.loads((ROOT/'artifacts/xapps/chongyu/20261117/release.json').read_text())['models']
    recipe=dict(schema=runtime_recipe.SCHEMA,id='validated_common_v1',seed=20261214,
        baseline_slot=2040,origin_observe_slots=2000,tail_slots=400,traffic_load=1.5,dynamics_members=317,
        dynamics_sha256='a'*64,validated_source_sha256='b'*64,validated_source_archive_sha256='c'*64,
        profiles={name:dict(path=str(root/(name+'.json')),sha256='d'*64)for name in runtime_recipe.PROFILE_NAMES},
        frozen_models={'artifacts/xapps/chongyu/20261117/'+e['path']:e['sha256']for e in old_models.values()},
        validated_origin={})
    runtime_recipe._shape(recipe)
    assert len(release.expected_model_pins(recipe))==6 and release.runtime_arguments(recipe,root)==[]
    recipe.update(new);recipe['dynamics_members']=318
    runtime_recipe._shape(recipe)
    recipe['model_release']['action_catalogue_sha256']='0'*64
    with pytest.raises(ValueError):runtime_recipe._shape(recipe)


def test_historical_contract_context_and_old_six_artifacts_stay_byte_identical():
    base=operator_audit('integration-staging-base-manifest.json')
    from smo.contracts import build_graph
    graph=build_graph(ROOT)
    for name,pin in graph['contract_sha256'].items():
        assert pin==base['files'][name]['sha256']==release.sha(ROOT/name)
    for name,pin in base['files'].items():
        if name.startswith('artifacts/xapps/chongyu/20261117/'):
            assert release.sha(ROOT/name)==pin['sha256']


def test_original_90_intents_presets_context_and_R_source_remain_unchanged():
    from smo.contracts import build_graph
    from smo.population import graph_view
    from smo.intent_ledger import IntentLedger
    audit=operator_audit('integration-staging-review.json')
    base=operator_audit('integration-staging-base-manifest.json')
    ui=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122/common-validation-indexed/ui')
    manifest=json.loads((ui/'scenario-manifest-v1.json').read_text())
    entry=next(e for e in manifest['scenarios']if e['id']=='validated_common_v1')
    path=Path(entry['intent_store']);ledger=json.loads(path.read_text())
    original=json.loads(Path(entry['intent_migration']['source_store']).read_text())
    graph=graph_view(build_graph(ROOT))
    context=dict(metric_scope=graph['metric_scope'],contracts=graph['contract_sha256'],run=str(Path(entry['replay']).resolve()))
    loaded=IntentLedger(context,path)
    assert len(loaded.records)==90 and list(loaded.records.values())==ledger['records']==original['records']
    assert context==ledger['context'] and manifest['default']=='validated_common_v1'
    assert release.sha(path)==audit['ledger_check']['ledger_sha256']
    assert release.sha(manifest['preset_store'])==audit['ledger_check']['preset_sha256']
    for relative,record in base['files'].items():
        assert release.sha(Path(base['source_root'])/relative)==record['sha256']
