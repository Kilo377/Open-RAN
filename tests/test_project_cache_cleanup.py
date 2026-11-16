import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SPEC=importlib.util.spec_from_file_location('project_cache_cleanup',Path(__file__).resolve().parents[1]/'scripts/orin/clean-project-caches.py')
gc=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gc)


def test_cleanup_preserves_recordings_models_and_nested_run_channel_caches(tmp_path,monkeypatch):
    root=tmp_path/'project';(root/'experiments').mkdir(parents=True)
    (root/'experiments/run_rt_sys.py').write_text('# test project')
    keep=['outputs/run/rt_sys_trace.jsonl.gz','outputs/run/channel_cache/paths.npz',
          'outputs/checkpoints/model.json','artifacts/xapps/model.json','data/train.json']
    drop=['.cache/drjit/kernel.bin','.cache/cupy/kernel.bin','.pytest_cache/test.json',
          'channel_cache/paths.npz','experiments/__pycache__/module.pyc']
    for name in keep+drop:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
    monkeypatch.setattr(gc,'active_processes',lambda _:[])
    report=gc.clean(root,apply=False)
    assert not report['applied']and all((root/f).exists()for f in keep+drop)
    report=gc.clean(root,apply=True)
    assert report['protected_data_unchanged']and report['protected_files']==len(keep)
    assert all((root/f).read_bytes()==f.encode()for f in keep)
    assert not any((root/f).exists()for f in drop)


def test_cache_symlinks_are_not_followed(tmp_path,monkeypatch):
    root=tmp_path/'project';(root/'experiments').mkdir(parents=True)
    (root/'experiments/run_rt_sys.py').touch()
    outside=tmp_path/'training';outside.mkdir();(outside/'model.json').write_text('keep')
    (root/'channel_cache').symlink_to(outside,target_is_directory=True)
    monkeypatch.setattr(gc,'active_processes',lambda _:[])
    gc.clean(root,True)
    assert (outside/'model.json').read_text()=='keep'and(root/'channel_cache').is_symlink()


def test_active_project_process_blocks_mutation(tmp_path,monkeypatch):
    (tmp_path/'experiments').mkdir();(tmp_path/'experiments/run_rt_sys.py').touch()
    cache=tmp_path/'channel_cache';cache.mkdir();(cache/'x').write_text('keep')
    monkeypatch.setattr(gc,'active_processes',lambda _:[dict(pid=42,command='simulation')])
    with pytest.raises(RuntimeError,match='active'):gc.clean(tmp_path,True)
    assert(cache/'x').read_text()=='keep'


def test_relative_python_command_is_detected_by_its_project_working_directory(tmp_path):
    (tmp_path/'experiments').mkdir();(tmp_path/'experiments/run_rt_sys.py').touch()
    cache=tmp_path/'channel_cache';cache.mkdir();(cache/'x').write_text('keep')
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],cwd=tmp_path)
    try:
        with pytest.raises(RuntimeError,match='active'):gc.clean(tmp_path,True)
        assert(cache/'x').read_text()=='keep'
    finally:child.terminate();child.wait(timeout=5)
