"""Frozen model backups and observational progress never mutate a live queue."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments import run_xapp_long_validation as study


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/orin/checkpoint_xapp_evaluation.py'
SPEC = importlib.util.spec_from_file_location('checkpoint_support_for_tests', SCRIPT)
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / 'project'; root.mkdir()
    groups = ('core', 'phy', 'ran', 'ric', 'state', 'models', 'traffic', 'experiments', 'configs')
    for group in groups: (root / group).mkdir()
    (root / 'core/simulator.py').write_text('# frozen live network source\n')
    dump(root / 'configs/rt_sys/agx_munich_6ue_depth2_fast.json', dict(radio=dict(slot_duration_ms=.5)))
    dump(root / 'configs/rt_sys/ap_services_reference.json', dict(ap_services=dict(enabled=True)))
    models = root / 'outputs/frozen_models'; models.mkdir(parents=True)
    entries = {}
    for app in study.APP_IDS:
        filename = app + '.json'
        dump(models / filename, dict(format=study.FORMATS[app], x=[[0.]] * 24,
            y=[[1.]] * 24, trained_samples=24, fit_count=1,
            feature_names=['f'], fit_parameters=dict(n_jobs=1), seed=55,
            forest=[dict(children_left=[-1], children_right=[-1], feature=[-2],
                threshold=[-2.], value=[[1.]])], coefs=[[[1.]]], intercepts=[[0.]]))
        entries[app] = dict(path=filename, sha256=study.sha256(models / filename))
    dump(models / 'models.json', dict(models=entries))
    queue = root / 'outputs/validation'; queue.mkdir()
    args = study.parse_args(['--model-dir', str(models), '--output', str(queue),
                             '--reuse-matched-baselines'])
    plan = study.build_plan(args, root=root)
    state = study.prepare_queue(plan)
    # Archives are content-validated, not trusted by their filename.
    archive = root / 'outputs/declared-source.tgz'
    with tarfile.open(archive, 'w:gz') as stream:
        for group in groups: stream.add(root / group, arcname=group)
    return SimpleNamespace(root=root, queue=queue, cp=queue / 'checkpoints',
        models=models, archive=archive, plan=plan, state=state)


def initialize(scene):
    return checkpoint.initialize_checkpoint(scene.queue, scene.cp, scene.archive)


def complete_first_case(scene):
    case = scene.plan['cases'][0]
    folder = Path(case['output']); folder.mkdir(parents=True)
    (folder / 'raw.jsonl').write_text('{"slot_id":1}\n')
    audit_path = scene.queue / 'audits' / (case['name'] + '.json')
    dump(audit_path, dict(case=case['name'], source_sha256=scene.plan['source_sha256'],
        recorded_file_sha256=hashes(folder)))
    entry = dict(name=case['name'], output=str(folder), audit=str(audit_path),
        audit_sha256=study.sha256(audit_path), reused_from=None, physical_run=True)
    state = deepcopy(scene.state); state.update(status='running',
        completed_cases=[entry], active_case=dict(name='x01_frozen', child_pid=12345,
        progress=dict(last_completed_slot=4000, simulated_seconds=2.)))
    dump(scene.queue / 'queue-status.json', state)
    return case, audit_path, entry


def test_snapshot_is_verified_atomic_and_idempotent_without_changing_live_inputs(scene):
    before_models = hashes(scene.models)
    before_queue = hashes(scene.queue)
    source_before = study.source_fingerprint(scene.root)
    result = initialize(scene)
    assert result['checkpoint_kind'] == 'frozen_model_and_group_progress'
    assert result['supports_mid_simulation_resume'] is False
    modelset = Path(result['modelset_path'])
    first_hashes = hashes(modelset)
    first_times = {str(p.relative_to(modelset)): p.stat().st_mtime_ns
                   for p in modelset.rglob('*') if p.is_file()}
    again = initialize(scene)
    assert again['modelset_id'] == result['modelset_id']
    assert hashes(modelset) == first_hashes
    assert {str(p.relative_to(modelset)): p.stat().st_mtime_ns
            for p in modelset.rglob('*') if p.is_file()} == first_times
    verified = checkpoint.verify_checkpoint(scene.queue, scene.cp)
    assert verified['valid'] is True and verified['models_verified'] == 4
    assert verified['raw_case_data_verified'] is False
    assert hashes(scene.models) == before_models
    assert study.source_fingerprint(scene.root) == source_before
    after_originals = {key:value for key,value in hashes(scene.queue).items()
                       if not key.startswith('checkpoints/')}
    assert after_originals == before_queue
    assert checkpoint.archive_fingerprint(scene.archive) == source_before


@pytest.mark.parametrize('target', ['model', 'manifest', 'profile', 'source'])
def test_changed_live_protected_inputs_refuse_initial_checkpoint(scene, target):
    selected = dict(model=scene.models / 'x01.json', manifest=scene.models / 'models.json',
        profile=Path(scene.plan['profile']), source=scene.root / 'core/simulator.py')[target]
    selected.write_text(selected.read_text() + '\n')
    with pytest.raises((ValueError, RuntimeError)):
        initialize(scene)
    assert not list((scene.cp / 'modelsets').glob('*/manifest.json'))


def test_source_archive_must_reconstruct_declared_source_fingerprint(scene):
    wrong = scene.root / 'core/simulator.py'
    wrong.write_text('# different archived source\n')
    with tarfile.open(scene.archive, 'w:gz') as stream:
        for group in ('core', 'phy', 'ran', 'ric', 'state', 'models', 'traffic', 'experiments', 'configs'):
            stream.add(scene.root / group, arcname=group)
    wrong.write_text('# frozen live network source\n')
    with pytest.raises((ValueError, RuntimeError)):
        initialize(scene)
    assert not list((scene.cp / 'modelsets').glob('*/manifest.json'))


def test_interrupted_copy_does_not_publish_a_valid_modelset_and_retry_works(scene):
    original = checkpoint.copy_verified
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2: raise OSError('simulated interrupted copy')
        return original(*args, **kwargs)
    with patch.object(checkpoint, 'copy_verified', side_effect=interrupted):
        with pytest.raises(OSError, match='interrupted'):
            initialize(scene)
    assert not list((scene.cp / 'modelsets').glob('*/manifest.json'))
    assert not (scene.cp / 'latest-modelset.json').exists()
    initialize(scene)
    assert checkpoint.verify_checkpoint(scene.queue, scene.cp)['valid'] is True


def test_archived_model_tampering_is_rejected_without_rewriting_checkpoint(scene):
    result = initialize(scene)
    path = Path(result['modelset_path']) / 'models/x01.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint.verify_checkpoint(scene.queue, scene.cp)
    with pytest.raises((ValueError, RuntimeError)):
        initialize(scene)
    assert path.read_text().endswith('\n\n')


def test_progress_and_completed_receipt_preserve_original_case_files(scene):
    result = initialize(scene)
    case, audit, entry = complete_first_case(scene)
    case_before = hashes(Path(case['output']))
    checkpoint.snapshot_progress(scene.queue, scene.cp, result['modelset_id'])
    latest = json.loads((scene.cp / 'latest-progress.json').read_text())
    assert latest['modelset_id'] == result['modelset_id']
    receipt = json.loads((scene.cp / 'completed' / (case['name'] + '.json')).read_text())
    assert receipt['case_audit_sha256'] == entry['audit_sha256']
    assert receipt['original_completed_entry'] == entry
    assert receipt['modelset_id'] == result['modelset_id']
    assert receipt['queue_plan_sha256'] == study.sha256(scene.queue / 'queue-plan.json')
    assert hashes(Path(case['output'])) == case_before
    assert checkpoint.verify_checkpoint(scene.queue, scene.cp)['completed_receipts_verified'] == 1
    state = json.loads((scene.queue / 'queue-status.json').read_text())
    state['active_case']['progress']['last_completed_slot'] = 8000
    state['active_case']['progress']['simulated_seconds'] = 4.
    dump(scene.queue / 'queue-status.json', state)
    checkpoint.snapshot_progress(scene.queue, scene.cp, result['modelset_id'])
    assert json.loads((scene.cp / 'completed' / (case['name'] + '.json')).read_text()) == receipt
    latest = json.loads((scene.cp / 'latest-progress.json').read_text())
    assert '8000' in json.dumps(latest)
    audit.write_text(audit.read_text() + '\n')
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint.snapshot_progress(scene.queue, scene.cp, result['modelset_id'])


def test_checkpoint_lock_is_independent_of_queue_lock_and_rejects_another_sidecar(scene):
    with study.queue_lock(scene.queue):
        with checkpoint.checkpoint_lock(scene.cp):
            with pytest.raises((ValueError, RuntimeError, BlockingIOError)):
                with checkpoint.checkpoint_lock(scene.cp):
                    pytest.fail('a second checkpoint writer acquired the sidecar lock')


def test_restore_to_new_directory_keeps_original_model_manifest_and_bytes(scene):
    initialize(scene)
    before = hashes(scene.models)
    destination = scene.root / 'outputs/restored_models'
    checkpoint.restore_models(scene.queue, scene.cp, destination)
    assert hashes(destination) == before
    assert hashes(scene.models) == before
    assert checkpoint.verify_checkpoint(scene.queue, scene.cp)['valid'] is True
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        checkpoint.restore_models(scene.queue, scene.cp, destination)


def test_restore_refuses_live_input_or_queue_paths_even_when_empty(scene):
    initialize(scene)
    for file in scene.models.iterdir(): file.unlink()
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        checkpoint.restore_models(scene.queue, scene.cp, scene.models)
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        checkpoint.restore_models(scene.queue, scene.cp, scene.queue / 'new_model_folder')
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        checkpoint.restore_models(scene.queue, scene.cp, scene.root / 'restored_outside_outputs')
