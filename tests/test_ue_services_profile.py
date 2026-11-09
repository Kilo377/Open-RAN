"""A pressure profile must change only the declared default UE4 routing policy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.run_rt_sys import configure_ue_services, parse_args

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'configs/rt_sys/ue_services.json'
PRESSURE=ROOT/'configs/rt_sys/ue_services_mec_offload_first.json'
MANIFEST={'ues':[{'ue_id':f'ue_{i}'} for i in (0,1,2,4,6,7)]}


def configuration():
    return {'phy':{'rt_sys':{'ue_services':{'native_marker':True}}}}


def test_profile_changes_only_ue4_default_and_preserves_original_file():
    before=NATIVE.read_bytes()
    native=json.loads(before);pressure=json.loads(PRESSURE.read_bytes())
    assert pressure['mec']['tasks_by_ue']['ue_4']['offload_fraction']==1
    pressure['mec']['tasks_by_ue']['ue_4']['offload_fraction']=.5
    assert pressure==native
    assert NATIVE.read_bytes()==before


def test_unspecified_profile_keeps_existing_native_configuration():
    cfg=configuration();before=deepcopy(cfg)
    assert configure_ue_services(cfg,MANIFEST,NS()) is None
    assert cfg==before
    assert configure_ue_services(cfg,MANIFEST,NS(ue_services='drx-mec')) is None
    assert cfg['phy']['rt_sys']['ue_services']==json.loads(NATIVE.read_bytes())


@pytest.mark.parametrize('mode',['off','rules','frozen'])
def test_each_arm_receives_the_same_pressure_profile_and_valid_task_contract(tmp_path,mode):
    from experiments.online_rf_xapps import configure
    from ric.near_rt.pools.chongyu.mec_offload_budget_qgbdt import MecOffloadBudgetQGBDT
    cfg=configuration();cfg['ric']={'xapp_pool':'chongyu','rapps':[],'xapps':[]}
    manifest=deepcopy(MANIFEST)
    manifest.update(cells=[{'cell_id':'cell_0'}],slot_duration_ms=.5)
    args=NS(ue_services='drx-mec',ue_services_config=PRESSURE,
        budget_xapp_mode=mode,xapp_pool='chongyu',seed=211,control_mode='baseline')
    metadata=configure_ue_services(cfg,manifest,args)
    assert metadata=={'path':str(PRESSURE.resolve()),'sha256':hashlib.sha256(PRESSURE.read_bytes()).hexdigest()}
    assert cfg['phy']['rt_sys']['ue_services']==json.loads(PRESSURE.read_bytes())
    configure(cfg,manifest,args,tmp_path/'outputs'/'arm',tmp_path)
    if mode!='off':
        profile=cfg['ric']['xapps'][0]['parameters']['task_profile']
        assert profile=={'interval_ms':50,'compute_gflop':.05,'deadline_ms':100}
        controller=MecOffloadBudgetQGBDT(['cell_0'],parameters={'task_profile':profile,'mode':'rules'})
        assert controller.feasible()==(0.,.5)


@pytest.mark.parametrize('preset',[None,'off'])
def test_disabled_or_unspecified_preset_rejects_custom_profile_atomically(preset):
    cfg=configuration();before=deepcopy(cfg)
    with pytest.raises(ValueError,match='enabled --ue-services preset'):
        configure_ue_services(cfg,MANIFEST,NS(ue_services=preset,ue_services_config=PRESSURE))
    assert cfg==before


def test_malformed_profile_cannot_partially_replace_service_settings(tmp_path):
    bad=tmp_path/'bad.json';bad.write_text(json.dumps({'drx':{},'mec':{'tasks_by_ue':[]}}))
    cfg=configuration();before=deepcopy(cfg)
    with pytest.raises(ValueError,match='per-UE mappings'):
        configure_ue_services(cfg,MANIFEST,NS(ue_services='drx-mec',ue_services_config=bad))
    assert cfg==before


def test_pressure_profile_retains_existing_selected_ue_filter():
    cfg=configuration()
    configure_ue_services(cfg,{'ues':[{'ue_id':'ue_4'}]},NS(ue_services='drx-mec',ue_services_config=PRESSURE))
    services=cfg['phy']['rt_sys']['ue_services']
    assert set(services['mec']['tasks_by_ue'])=={'ue_4'}
    assert services['drx_by_ue']=={}


def test_cli_and_live_comparison_preserve_the_explicit_profile(monkeypatch):
    from experiments.live_xapp_test import ARGUMENTS
    monkeypatch.setattr('sys.argv',['run_rt_sys.py','--ue-services','drx-mec','--ue-services-config',str(PRESSURE)])
    assert parse_args().ue_services_config==PRESSURE
    assert 'ue_services_config' in ARGUMENTS
