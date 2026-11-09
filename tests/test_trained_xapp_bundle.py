"""The portable release must preserve verified checkpoints and runtime loading."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/xapps/chongyu/20261109'
spec = importlib.util.spec_from_file_location('install_trained_xapps',
    ROOT / 'scripts/orin/install-trained-xapps.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def complete_bundle(tmp_path):
    manifest = json.loads((BUNDLE/'release.json').read_text())
    if not manifest['complete']:
        pytest.skip('The new X12 training artifact has not been finalized yet')
    path = tmp_path/'release'
    shutil.copytree(BUNDLE, path)
    return path


def test_incomplete_release_cannot_be_installed(tmp_path):
    path=tmp_path/'incomplete';path.mkdir()
    (path/'release.json').write_text(json.dumps({'schema':'oran.chongyu-trained-model-release.v1',
        'complete':False,'pool':'chongyu','models':{}}))
    with pytest.raises(ValueError,match='complete six-model'):
        installer.install(path,tmp_path/'outputs/release',tmp_path)
    assert not (tmp_path/'outputs/release').exists()


def test_install_is_idempotent_and_preserves_existing_changed_checkpoint(complete_bundle,tmp_path):
    target=tmp_path/'outputs/release'
    assert installer.install(complete_bundle,target,tmp_path)['status']=='installed'
    assert installer.install(complete_bundle,target,tmp_path)['status']=='already_installed'
    changed=target/'models/x08.json';changed.write_text('{"existing":"changed"}')
    with pytest.raises(ValueError,match='Existing install differs'):
        installer.install(complete_bundle,target,tmp_path)
    assert changed.read_text()=='{"existing":"changed"}'


def test_tampered_release_does_not_create_a_partial_install(complete_bundle,tmp_path):
    model=complete_bundle/'models/x13.json';model.write_bytes(model.read_bytes()+b' ')
    with pytest.raises(ValueError,match='checksum mismatch'):
        installer.install(complete_bundle,tmp_path/'outputs/release',tmp_path)
    assert not (tmp_path/'outputs/release').exists()


def test_install_rejects_non_outputs_destination(complete_bundle,tmp_path):
    with pytest.raises(ValueError,match='under project outputs'):
        installer.install(complete_bundle,tmp_path/'models/release',tmp_path)


def test_installed_four_model_manifest_is_accepted_by_evaluation_driver(complete_bundle,tmp_path):
    from experiments.run_xapp_long_validation import load_models
    target=tmp_path/'outputs/release';installer.install(complete_bundle,target,tmp_path)
    models=load_models(target,root=tmp_path)
    assert set(models)=={'x01','x02','x08','x12_mec'}
    release=installer.load_release(complete_bundle)
    assert all(models[a]['sha256']==release['models'][a]['sha256'] for a in models)
    arguments=json.loads((target/'runtime-arguments.json').read_text())
    assert arguments['x12_urllc'][-2:]==['--urllc-xapp-policy-version','qos_v2']


def test_all_six_real_checkpoints_initialize_frozen_apps_and_predict_without_mutation(complete_bundle,tmp_path,monkeypatch):
    from ric.near_rt.pools.chongyu import ue_energy_rf as ue
    from ric.near_rt.pools.chongyu import ap_energy_rf as ap
    from ric.near_rt.pools.chongyu import throughput_scheduler_mlp as throughput
    from ric.near_rt.pools.chongyu import latency_scheduler_rf as mec
    from ric.near_rt.pools.chongyu import latency_urllc_rf as urllc
    from ric.near_rt.pools.chongyu import latency_urllc_qos_rf as qos
    from ric.near_rt.pools.chongyu import mec_offload_budget_qgbdt as budget
    for module in (ue,ap,throughput,mec,urllc,qos,budget):
        monkeypatch.setattr(module,'ROOT',tmp_path)
    target=tmp_path/'outputs/release';installer.install(complete_bundle,target,tmp_path)
    release=installer.load_release(complete_bundle)
    classes={'x01':ue.UeEnergyRF,'x02':ap.ApEnergyRF,'x08':throughput.ThroughputSchedulerMLP,
        'x12_mec':mec.LatencySchedulerRF,'x12_urllc':urllc.LatencyUrllcRF,
        'x13':budget.MecOffloadBudgetQGBDT}
    for name,klass in classes.items():
        entry=release['models'][name];path=target/entry['path'];before=path.read_bytes()
        params=dict(entry['runtime_parameters'],mode='frozen',model_path=str(path))
        app=klass(['cell_0','cell_1','cell_2'],parameters=params)
        assert app.learner.trained
        assert app.learner.sample_count==entry['measured_rows']
        assert app.learner.trained_samples==entry['fitted_rows']
        predicted=app.learner.predict(app.learner._x[:2])
        assert len(predicted)==2
        assert path.read_bytes()==before
