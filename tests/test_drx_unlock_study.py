"""Check chosen vs executed DRX evidence; decision logs alone are insufficient."""
import gzip,json
from pathlib import Path
from experiments.run_drx_unlock_study import action_audit


def test_action_audit_distinguishes_a_proposal_from_executed_profile(tmp_path):
    (tmp_path/'energy_rf_events.jsonl').write_text(json.dumps(dict(kind='decision',action=8))+'\n')
    native=dict(enabled=True,cycle_ms=40,on_duration_ms=4,inactivity_ms=8)
    other=dict(enabled=True,cycle_ms=80,on_duration_ms=4,inactivity_ms=1)
    with gzip.open(tmp_path/'rt_sys_trace.jsonl.gz','wt')as stream:
        for profile in(native,other):
            stream.write(json.dumps(dict(ue_services={'ue_0':{'ue_service_policy':{'drx':profile}}}))+'\n')
    (tmp_path/'final_summary.json').write_text(json.dumps(dict(ue_services={})))
    audit=action_audit(tmp_path)
    assert audit['chosen_non_native_drx']==1
    assert audit['executed_non_native_drx_samples']==1
    assert audit['executed_policy_samples']==2


def test_unlocked_release_preserves_forest_bytes_and_exposes_exact_nine_actions(tmp_path):
    import importlib.util
    from smo.contracts import build_graph
    from smo.intent_match import profiles
    root=Path(__file__).resolve().parents[1]
    old=root/'artifacts/xapps/chongyu/20261109';new=old.with_name('20261117')
    assert (old/'models/x01.json').read_bytes()==(new/'models/x01.json').read_bytes()
    graph=build_graph();app=next(n for n in graph['nodes']if n.get('key')=='x01')
    assert app['runtime_parameters']['deploy_drx']is True
    spec=importlib.util.spec_from_file_location('installer',root/'scripts/orin/install-trained-xapps.py');installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    assert installer.load_release(new)['release']=='20261117'
    target=tmp_path/'outputs/current';installer.install(new,target,tmp_path)
    assert '--energy-xapp-deploy-drx' in json.loads((target/'runtime-arguments.json').read_text())['x01']
    from ric.near_rt.pools.chongyu.ue_energy_rf import ACTIONS
    assert [dict(drx=v['parts'][1]['value'],offload_fraction=v['parts'][0]['value'])for v in profiles(app)]==ACTIONS
