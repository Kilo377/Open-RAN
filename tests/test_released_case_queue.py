"""The case runner must warm Base, preserve evidence, and reap its children."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import os
import sys
import time

import pytest

from experiments import released_case_queue as queue


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root=tmp_path/'project';(root/'core').mkdir(parents=True)
    (root/'core/runtime.py').write_text('# immutable test-only runtime\n')
    model=root/'outputs/models/model.json';model.parent.mkdir(parents=True)
    model.write_text('{"test_only":true}\n')
    output=root/'outputs/study';events=root/'events.jsonl'
    code='''from pathlib import Path
import json,os,sys,time
folder,events,name,fail=sys.argv[1:];p=Path(folder);p.mkdir(parents=True)
def record(kind):
 with Path(events).open('a') as stream:stream.write(json.dumps([kind,name,time.monotonic(),os.getpid()])+'\\n')
record('start')
if name==fail:sys.exit(7)
time.sleep(.12)
(p/'finished.txt').write_text(name)
record('end')
'''
    cases=[dict(name=n,output=str(output/'cases'/n),command=[sys.executable,'-c',code,
        str(output/'cases'/n),str(events),n,'none']) for n in ('base','a','b','both')]
    plan=dict(project_root=str(root),output=str(output),source_sha256=queue.source_fingerprint(root),
        models={'toy':dict(path=str(model),sha256=queue.sha256(model))},cases=cases)
    args=NS(resume=False,max_parallel=2,min_free_gb=.25,estimated_case_gb=.01,
        heartbeat_s=.01,termination_grace_s=.2)
    monkeypatch.setattr(queue.shutil,'disk_usage',lambda _:NS(free=10*1024**3))

    def audit(case, plan):
        folder=Path(case['output'])
        assert (folder/'finished.txt').read_text()==case['name']
        return dict(case=case['name'],files_sha256={p.name:queue.sha256(p) for p in folder.iterdir()})

    def compare(arms, plan, completed=False):
        assert 'base' in arms
        return dict(completed=completed,arms=sorted(arms))
    return args,plan,events,audit,compare


def test_base_warms_first_then_bounded_parallel_and_completed_resume_does_not_rerun(fixture, monkeypatch):
    args,plan,events,audit,compare=fixture
    assert queue.run_queue(args,plan,audit,compare)==0
    rows=[json.loads(s) for s in events.read_text().splitlines()]
    base_end=next(r[2] for r in rows if r[:2]==['end','base'])
    assert all(r[2]>=base_end for r in rows if r[0]=='start' and r[1]!='base')
    running=peak=0
    for kind,name,stamp,pid in sorted(rows,key=lambda r:r[2]):
        running+=1 if kind=='start' else -1
        peak=max(peak,running)
    assert peak==args.max_parallel==2
    state=queue.load(Path(plan['output'])/'queue-status.json')
    assert state['completed'] and len(state['completed_cases'])==4 and not state['active_cases']
    args.resume=True;before=events.read_bytes()
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed child reran'))
    assert queue.run_queue(args,plan,audit,compare)==0
    assert events.read_bytes()==before


def test_tampered_completed_recording_rejected_without_rerun(fixture, monkeypatch):
    args,plan,events,audit,compare=fixture
    queue.run_queue(args,plan,audit,compare)
    path=Path(plan['cases'][1]['output'])/'finished.txt';path.write_text('changed evidence')
    args.resume=True
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**k:pytest.fail('No rerun may conceal tampering'))
    with pytest.raises(ValueError,match='recording changed'):
        queue.run_queue(args,plan,audit,compare)
    assert path.read_text()=='changed evidence'


def test_frozen_input_change_rejected_before_launch(fixture, monkeypatch):
    args,plan,events,audit,compare=fixture
    Path(plan['models']['toy']['path']).write_text('{"changed":true}')
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**k:pytest.fail('Changed model launched'))
    with pytest.raises(ValueError,match='model changed'):
        queue.run_queue(args,plan,audit,compare)
    assert not events.exists()


def test_disk_guard_stops_without_child_or_deleting_data(fixture, monkeypatch):
    args,plan,events,audit,compare=fixture
    keep=Path(plan['project_root'])/'keep.txt';keep.write_text('retain')
    monkeypatch.setattr(queue.shutil,'disk_usage',lambda _:NS(free=1))
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**k:pytest.fail('Disk-blocked child launched'))
    assert queue.run_queue(args,plan,audit,compare)==75
    assert queue.load(Path(plan['output'])/'queue-status.json')['status']=='blocked_disk_space'
    assert keep.read_text()=='retain' and not events.exists()


def test_failed_child_reaps_other_parallel_children_and_preserves_recordings(fixture):
    args,plan,events,audit,compare=fixture
    plan['cases'][2]['command'][-1]='b'
    with pytest.raises(RuntimeError,match='child failed: b'):
        queue.run_queue(args,plan,audit,compare)
    state=queue.load(Path(plan['output'])/'queue-status.json')
    assert state['status']=='failed' and not state['completed']
    assert (Path(plan['cases'][0]['output'])/'finished.txt').is_file()
    for row in [json.loads(s) for s in events.read_text().splitlines()]:
        if row[0]=='start':
            with pytest.raises(ProcessLookupError):os.kill(row[3],0)


def test_duplicate_arm_cannot_silently_replace_a_declared_case(fixture):
    args,plan,events,audit,compare=fixture
    plan['cases'].append(dict(plan['cases'][0]))
    with pytest.raises(ValueError,match='four-arm case'):
        queue.run_queue(args,plan,audit,compare)
    assert not events.exists()


def test_integer_action_counter_keys_resume_without_rerunning_completed_data(fixture,monkeypatch):
    args,plan,events,audit,compare=fixture
    plan['models']['toy']['readiness']={'action_counts':{0:12,1:17}}
    assert queue.run_queue(args,plan,audit,compare)==0
    saved=queue.load(Path(plan['output'])/'queue-plan.json')
    assert saved['models']['toy']['readiness']['action_counts']=={'0':12,'1':17}
    args.resume=True;before=events.read_bytes()
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed child reran'))
    assert queue.run_queue(args,plan,audit,compare)==0
    assert events.read_bytes()==before


def stagger_children(plan):
    """A finishes while B is still doing independent, successful work."""
    for case in plan['cases']:
        code=case['command'][2].replace('time.sleep(.12)',
            "time.sleep(.8 if name=='b' else .02)")
        case['command'][2]=code.replace("(p/'finished.txt').write_text(name)",
            "(p/'finished.txt').write_text(name)\n(p/'run_status.json').write_text('{\"completed\":true}')")


def test_audit_failure_drains_healthy_peer_stops_new_work_and_resumes_saved_data(fixture,monkeypatch):
    args,plan,events,audit,compare=fixture
    stagger_children(plan)
    failed={'enabled':True};failure_times=[];comparisons=[];logs={}
    original_popen=queue.subprocess.Popen

    def observed_popen(command,**kwargs):
        logs[command[-2]]=kwargs['stdout']
        return original_popen(command,**kwargs)

    monkeypatch.setattr(queue.subprocess,'Popen',observed_popen)

    def independent_audit(case,plan):
        if case['name']=='a' and failed['enabled']:
            rows=[json.loads(line) for line in events.read_text().splitlines()]
            peer=next(row for row in rows if row[:2]==['start','b'])
            os.kill(peer[3],0)  # The expensive peer is actually alive now.
            assert not any(row[:2]==['end','b'] for row in rows)
            failure_times.append(time.monotonic())
            raise ValueError('A recording audit failed')
        if case['name']=='b' and failed['enabled']:
            draining=queue.load(Path(plan['output'])/'queue-status.json')
            assert draining['status']=='draining_after_audit_failure'
            assert draining['failures'][0]['case']=='a'
            assert draining['unlaunched_cases']==['both']
        return audit(case,plan)

    def checked_compare(arms,plan,completed=False):
        comparisons.append((set(arms),completed))
        return compare(arms,plan,completed)

    with pytest.raises(ValueError,match='A recording audit failed'):
        queue.run_queue(args,plan,independent_audit,checked_compare)
    rows=[json.loads(line) for line in events.read_text().splitlines()]
    assert next(row[2] for row in rows if row[:2]==['end','b'])>failure_times[0]
    assert not any(row[:2]==['start','both'] for row in rows)
    assert all(log.closed for log in logs.values())
    state=queue.load(Path(plan['output'])/'queue-status.json')
    assert state['status']=='failed' and not state['completed'] and not state['active_cases']
    assert {item['name'] for item in state['completed_cases']}=={'base','b'}
    assert state['unlaunched_cases']==['both']
    assert len(state['failures'])==1
    assert state['failures'][0]['case']=='a' and state['failures'][0]['phase']=='audit'
    assert state['failures'][0]['message']=='A recording audit failed'
    assert comparisons[-1]==({'base','b'},False)
    assert not any(completed for _,completed in comparisons)
    partial=queue.load(Path(plan['output'])/'comparison-summary.json')
    assert not partial['completed'] and set(partial['arms'])=={'base','b'}
    assert not (Path(plan['output'])/'audits/a.json').exists()
    assert (Path(plan['cases'][1]['output'])/'finished.txt').read_text()=='a'
    for row in rows:
        if row[0]=='start':
            with pytest.raises(ProcessLookupError):os.kill(row[3],0)

    # A corrected auditor can re-check A's complete retained recording. Base/B
    # are verified by saved hashes; only the previously unlaunched arm runs.
    failed['enabled']=False;args.resume=True;logs.clear()
    assert queue.run_queue(args,plan,independent_audit,checked_compare)==0
    assert set(logs)=={'both'}
    after=[json.loads(line) for line in events.read_text().splitlines()]
    assert all(sum(row[:2]==['start',name] for row in after)==1 for name in ('base','a','b','both'))
    assert queue.load(Path(plan['output'])/'queue-status.json')['completed']


def test_base_audit_failure_cannot_launch_any_on_arm(fixture):
    args,plan,events,audit,compare=fixture

    def failed_base(case,plan):
        raise ValueError('Base audit failed')

    with pytest.raises(ValueError,match='Base audit failed'):
        queue.run_queue(args,plan,failed_base,compare)
    rows=[json.loads(line) for line in events.read_text().splitlines()]
    assert {row[1] for row in rows}=={'base'}
    state=queue.load(Path(plan['output'])/'queue-status.json')
    assert state['status']=='failed' and not state['completed']
    assert state['completed_cases']==[] and state['unlaunched_cases']==['a','b','both']


def test_matched_comparison_failure_still_terminates_and_reaps_peer(fixture):
    args,plan,events,audit,compare=fixture
    stagger_children(plan)

    def reject_unmatched(arms,plan,completed=False):
        if 'a' in arms:raise ValueError('Unmatched inputs')
        return compare(arms,plan,completed)

    with pytest.raises(ValueError,match='Unmatched inputs'):
        queue.run_queue(args,plan,audit,reject_unmatched)
    rows=[json.loads(line) for line in events.read_text().splitlines()]
    assert any(row[:2]==['start','b'] for row in rows)
    assert not any(row[:2]==['end','b'] for row in rows)
    state=queue.load(Path(plan['output'])/'queue-status.json')
    assert state['status']=='failed' and not state['completed']
    assert all(item.get('phase')!='audit' for item in state['failures'])
    for row in rows:
        if row[0]=='start':
            with pytest.raises(ProcessLookupError):os.kill(row[3],0)
