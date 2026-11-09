"""Serial long-horizon collection preserves paired inputs and resumable evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import run_xapp_long_validation as study


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def option(command, flag):
    """Argparse's final occurrence is authoritative if a launcher repeats a flag."""
    index = max(i for i, value in enumerate(command) if value == flag)
    return command[index + 1]


class TestLongValidation(unittest.TestCase):
    def fixture(self, reuse=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        for name in ('core', 'phy', 'ran', 'ric', 'state', 'models', 'traffic', 'experiments', 'configs'):
            (root / name).mkdir()
        (root / 'core/simulator.py').write_text('# immutable tested runtime\n')
        dump(root / 'configs/rt_sys/agx_munich_6ue_depth2_fast.json',
             dict(radio=dict(slot_duration_ms=.5)))
        dump(root / 'configs/rt_sys/ap_services_reference.json',
             dict(profile_id='test', ap_services=dict(enabled=True)))
        launcher = root / 'scripts/orin/run-six-ue-cuda-nr.sh'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('#!/bin/sh\nexit 0\n')
        models = root / 'outputs/frozen_models'
        mapping = {}
        for key in ('x01', 'x02', 'x08', 'x12_mec'):
            payload = dict(format='chongyu.online-throughput-mlp' if key == 'x08' else
                'chongyu.online-ap-rf' if key == 'x02' else 'chongyu.online-energy-rf',
                version=2 if key in ('x08', 'x12_mec') else 1,
                x=[[0.]] * 24, y=[[0.]] * 24, trained_samples=24, fit_count=1,
                prediction_batches=0, feature_names=['test_feature'], target_names=['test_target'],
                forest=[dict(children_left=[-1], children_right=[-1], feature=[-2],
                    threshold=[-2.], value=[[0.]])], coefs=[[[0.]]], intercepts=[[0.]],
                fit_parameters={}, fit_s=0., predict_s=0.)
            path = models / (key + '.json'); dump(path, payload)
            mapping[key] = dict(path=path.name, sha256=study.sha256(path))
        dump(models / 'models.json', dict(models=mapping))
        arguments = ['--model-dir', str(models), '--output', str(root / 'outputs/long')]
        if reuse: arguments.append('--reuse-matched-baselines')
        args = study.parse_args(arguments)
        plan = study.build_plan(args, root=root)
        return root, args, plan

    def test_eight_serial_cases_keep_pair_duration_seed_load_and_frozen_controls(self):
        root, args, plan = self.fixture()
        self.assertEqual(len(plan['cases']), 8)
        self.assertEqual(plan['physical_run_count'], 8)
        expected = ['x01_off', 'x01_frozen', 'x02_off', 'x02_frozen',
                    'x08_off', 'x08_frozen', 'x12_mec_off', 'x12_mec_frozen']
        self.assertEqual([c['name'] for c in plan['cases']], expected)
        active = {'x01': '--energy-xapp-mode', 'x02': '--ap-xapp-mode',
                  'x08': '--throughput-xapp-mode', 'x12_mec': '--latency-xapp-mode'}
        all_flags = list(active.values()) + ['--urllc-xapp-mode', '--budget-xapp-mode']
        for case in plan['cases']:
            self.assertEqual(case['duration_s'], 30)
            self.assertEqual(case['slots'], 60000)
            cmd = case['command']
            self.assertEqual(float(option(cmd, '--duration-s')), 30)
            self.assertEqual(int(option(cmd, '--seed')), 55)
            self.assertEqual(option(cmd, '--model'), 'fast')
            self.assertEqual(option(cmd, '--record-compression'), 'gzip')
            self.assertEqual(option(cmd, '--xapp-pool'), 'chongyu')
            self.assertEqual(float(option(cmd, '--traffic-load')), 2 if case['app'] in ('x02', 'x08') else 1)
            self.assertEqual('--ap-services-profile' in cmd, case['app'] in ('x02', 'x08'))
            for flag in all_flags:
                wanted = 'frozen' if case['mode'] == 'frozen' and flag == active[case['app']] else 'off'
                self.assertEqual(option(cmd, flag), wanted)
        for bare, app in zip(plan['cases'][::2], plan['cases'][1::2]):
            self.assertEqual(bare['app'], app['app'])
            self.assertEqual(bare['traffic_load'], app['traffic_load'])
            self.assertEqual(option(bare['command'], '--seed'), option(app['command'], '--seed'))

    def test_changed_runtime_or_protected_model_is_rejected(self):
        root, args, plan = self.fixture()
        study.verify_inputs(plan, root=root)
        runtime = root / 'core/simulator.py'; original = runtime.read_text()
        runtime.write_text(original + '# changed after collection plan\n')
        with self.assertRaises((ValueError, RuntimeError)): study.verify_inputs(plan, root=root)
        runtime.write_text(original)
        model = Path(plan['models']['x01']['path']); original = model.read_text()
        model.write_text(original + '\n')
        with self.assertRaises((ValueError, RuntimeError)): study.verify_inputs(plan, root=root)

    def test_changed_manifest_or_profile_is_rejected(self):
        for relative in ('outputs/frozen_models/models.json', 'configs/rt_sys/ap_services_reference.json'):
            with self.subTest(relative=relative):
                root, args, plan = self.fixture()
                path = root / relative
                path.write_text(path.read_text() + '\n')
                with self.assertRaises((ValueError, RuntimeError)): study.verify_inputs(plan, root=root)

    def test_duplicate_queue_lock_is_rejected(self):
        root, args, plan = self.fixture()
        output = Path(args.output); output.mkdir(parents=True)
        with study.queue_lock(output):
            with self.assertRaises((ValueError, RuntimeError, BlockingIOError)):
                with study.queue_lock(output):
                    self.fail('second collector acquired the same queue')

    def process_fixture(self, fail_once=False, sleep_seconds=.02, reuse=False):
        root, args, plan = self.fixture(reuse=reuse)
        # These real children write only tiny lifecycle records. Control disk
        # availability independently of the host; dedicated disk-guard tests
        # below override this mock with explicit low-space transitions.
        available=SimpleNamespace(total=10*1024**3,used=2*1024**3,free=8*1024**3)
        disk=patch.object(study.shutil,'disk_usage',return_value=available)
        disk.start();self.addCleanup(disk.stop)
        args.heartbeat_s = .01
        args.termination_grace_s = .2
        flag, events = root / 'fail-once', root / 'events.jsonl'
        if fail_once: flag.write_text('fail the second case once')
        # A real, tiny child makes ordering/failure/signal tests exercise Popen,
        # new process sessions and wait/reaping without importing the simulator.
        script = '''import json,os,sys,time
from pathlib import Path
args=sys.argv[1:];folder=Path(args[args.index('--output')+1])
events,flag,bad,delay=args[:4];name=folder.name
folder.mkdir(parents=True,exist_ok=True)
def event(kind):
 with Path(events).open('a') as f:f.write(json.dumps([kind,name,os.getpid()])+'\\n')
event('start');(folder/'partial.txt').write_text(str(os.getpid()))
if name==bad and Path(flag).exists():
 Path(flag).unlink();event('fail');sys.exit(7)
time.sleep(float(delay));(folder/'finished.txt').write_text(name);event('end')
'''
        for case in plan['cases']:
            case['command'] = [sys.executable, '-c', script, str(events),
                str(flag), 'x01_frozen', str(sleep_seconds), '--output', case['output']]
        return root, args, plan, events

    def completed_audit(self, case, plan):
        folder = Path(case['output'])
        self.assertEqual((folder / 'finished.txt').read_text(), case['name'])
        return dict(case=case['name'], recorded_file_sha256={
            str(p.relative_to(folder)): study.sha256(p)
            for p in folder.rglob('*') if p.is_file()})

    def test_real_children_are_strictly_serial_and_verified_resume_starts_none(self):
        root, args, plan, events = self.process_fixture()
        with patch.object(study, 'audit_case', side_effect=self.completed_audit), \
                patch.object(study, 'pair_summaries', return_value={}):
            self.assertEqual(study.run_queue(args, plan), 0)
            records = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([(r[0], r[1]) for r in records], [
                (phase, case['name']) for case in plan['cases'] for phase in ('start', 'end')])
            status_path = Path(plan['output']) / 'queue-status.json'
            status = study.load(status_path)
            self.assertTrue(status['completed'])
            self.assertEqual(len(status['completed_cases']), 8)
            args.resume = True
            with patch.object(study.subprocess, 'Popen', side_effect=AssertionError('completed child reran')):
                self.assertEqual(study.run_queue(args, plan), 0)
            self.assertEqual(len(study.load(status_path)['completed_cases']), 8)
            self.assertEqual(len(events.read_text().splitlines()), 16)

    def test_explicit_baseline_reuse_keeps_eight_pairs_entries_but_only_six_children(self):
        root, args, plan, events = self.process_fixture(reuse=True)
        self.assertEqual(len(plan['cases']), 8)
        self.assertEqual(plan['physical_run_count'], 6)
        cases = {c['name']: c for c in plan['cases']}
        aliases = {'x08_off': 'x02_off', 'x12_mec_off': 'x01_off'}
        for alias, source in aliases.items():
            self.assertEqual(cases[alias]['reused_from'], source)
            self.assertEqual(cases[alias]['output'], cases[source]['output'])
            self.assertEqual(cases[alias]['command'], cases[source]['command'])
        with patch.object(study, 'audit_case', side_effect=self.completed_audit), \
                patch.object(study, 'pair_summaries', return_value={}):
            self.assertEqual(study.run_queue(args, plan), 0)
            records = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([(r[0], r[1]) for r in records], [
                (phase, c['name']) for c in plan['cases'] if 'reused_from' not in c
                for phase in ('start', 'end')])
            output = Path(plan['output']); state = study.load(output / 'queue-status.json')
            self.assertEqual(len(state['completed_cases']), 8)
            for alias, source in aliases.items():
                audit = study.load(output / 'audits' / (alias + '.json'))
                source_path = output / 'audits' / (source + '.json')
                self.assertEqual(audit['reused_from'], source)
                self.assertEqual(audit['source_baseline_audit_sha256'], study.sha256(source_path))
                self.assertEqual(audit['recorded_file_sha256'], study.load(source_path)['recorded_file_sha256'])
                entry = next(e for e in state['completed_cases'] if e['name'] == alias)
                self.assertEqual(entry['reused_from'], source)
                self.assertIs(entry['physical_run'], False)
                folder = Path(cases[source]['output'])
                self.assertEqual({str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()},
                                 set(audit['recorded_file_sha256']))
            args.resume = True
            with patch.object(study.subprocess, 'Popen', side_effect=AssertionError('reused baseline reran')):
                self.assertEqual(study.run_queue(args, plan), 0)
            self.assertEqual(len(events.read_text().splitlines()), 12)
            # The dependency is a frozen, audited baseline, not a mutable cache.
            source_path = output / 'audits/x01_off.json'
            source_path.write_text(source_path.read_text() + '\n')
            with patch.object(study.subprocess, 'Popen', side_effect=AssertionError('changed source audit reran')):
                with self.assertRaisesRegex(ValueError, 'audit changed'):
                    study.run_queue(args, plan)
            self.assertEqual(len(events.read_text().splitlines()), 12)

    def test_baseline_reuse_rejects_a_changed_declared_input_command(self):
        root, args, plan, events = self.process_fixture(reuse=True)
        alias = next(c for c in plan['cases'] if c['name'] == 'x08_off')
        alias['command'] = alias['command'] + ['--traffic-load', '3']
        with patch.object(study, 'audit_case', side_effect=self.completed_audit), \
                patch.object(study, 'pair_summaries', return_value={}):
            with self.assertRaises((ValueError, RuntimeError)):
                study.run_queue(args, plan)
        starts = [json.loads(line)[1] for line in events.read_text().splitlines()
                  if json.loads(line)[0] == 'start']
        self.assertEqual(starts, ['x01_off', 'x01_frozen', 'x02_off', 'x02_frozen'])
        status = study.load(Path(plan['output']) / 'queue-status.json')
        self.assertEqual(status['status'], 'failed')
        self.assertFalse(status['completed'])

    def test_failed_child_stops_queue_and_resume_archives_only_its_partial_attempt(self):
        root, args, plan, events = self.process_fixture(fail_once=True)
        with patch.object(study, 'audit_case', side_effect=self.completed_audit), \
                patch.object(study, 'pair_summaries', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'child failed'):
                study.run_queue(args, plan)
            status_path = Path(plan['output']) / 'queue-status.json'
            first = study.load(status_path)
            self.assertEqual(first['status'], 'failed')
            self.assertEqual([c['name'] for c in first['completed_cases']], ['x01_off'])
            self.assertEqual(first['failures'][0]['case'], 'x01_frozen')
            self.assertFalse(Path(plan['cases'][2]['output']).exists())
            failed_marker = (Path(plan['cases'][1]['output']) / 'partial.txt').read_text()
            args.resume = True
            self.assertEqual(study.run_queue(args, plan), 0)
            final = study.load(status_path)
            self.assertTrue(final['completed'])
            self.assertEqual(len(final['attempts']), 1)
            archived = Path(final['attempts'][0]['archived'])
            self.assertEqual((archived / 'case/partial.txt').read_text(), failed_marker)
            self.assertTrue((archived / 'child.log').is_file())
            names = [json.loads(line)[1] for line in events.read_text().splitlines()
                     if json.loads(line)[0] == 'start']
            self.assertEqual(names, ['x01_off', 'x01_frozen'] +
                             [c['name'] for c in plan['cases'][1:]])

    def test_resume_rejects_tampered_completed_recording_without_launching(self):
        root, args, plan, events = self.process_fixture(fail_once=True)
        with patch.object(study, 'audit_case', side_effect=self.completed_audit), \
                patch.object(study, 'pair_summaries', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'child failed'):
                study.run_queue(args, plan)
            marker = Path(plan['cases'][0]['output']) / 'finished.txt'
            marker.write_text('modified completed evidence')
            before = events.read_bytes(); args.resume = True
            with patch.object(study.subprocess, 'Popen', side_effect=AssertionError('tampered history reran')):
                with self.assertRaisesRegex(ValueError, 'recording changed'):
                    study.run_queue(args, plan)
            self.assertEqual(events.read_bytes(), before)
            self.assertEqual(marker.read_text(), 'modified completed evidence')

    def test_resume_rejects_added_recording_stream_without_launching(self):
        root, args, plan, events = self.process_fixture(fail_once=True)
        with patch.object(study, 'audit_case', side_effect=self.completed_audit), \
                patch.object(study, 'pair_summaries', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'child failed'):
                study.run_queue(args, plan)
            extra = Path(plan['cases'][0]['output']) / 'rt_sys_trace.jsonl'
            extra.write_text('an unverified additional raw stream\n')
            before = events.read_bytes(); args.resume = True
            with patch.object(study.subprocess, 'Popen', side_effect=AssertionError('changed inventory reran')):
                with self.assertRaisesRegex(ValueError, 'inventory|recording'):
                    study.run_queue(args, plan)
            self.assertEqual(events.read_bytes(), before)
            self.assertTrue(extra.is_file())

    def test_resume_requires_the_exact_original_plan(self):
        root, args, plan = self.fixture()
        with study.queue_lock(plan['output']):
            study.prepare_queue(plan)
        changed = deepcopy(plan); changed['seed'] = 56
        with self.assertRaisesRegex(ValueError, 'Resume parameters'):
            study.prepare_queue(changed, resume=True)
        with self.assertRaises(FileExistsError): study.prepare_queue(plan, resume=False)

    def test_insufficient_disk_blocks_before_child_or_existing_data_changes(self):
        root, args, plan = self.fixture()
        sentinel = root / 'keep-me.txt'; sentinel.write_text('unrelated data')
        args.min_free_gb, args.estimated_case_gb = 1., .8
        disk = SimpleNamespace(total=10*1024**3, used=8*1024**3, free=int(1.79*1024**3))
        with patch.object(study.shutil, 'disk_usage', return_value=disk), \
                patch.object(study.subprocess, 'Popen') as child:
            self.assertEqual(study.run_queue(args, plan), 75)
        child.assert_not_called()
        status = study.load(Path(plan['output']) / 'queue-status.json')
        self.assertEqual(status['status'], 'blocked_disk_space')
        self.assertEqual(status['disk']['next_case'], 'x01_off')
        self.assertFalse(status['completed'])
        self.assertEqual(status['completed_cases'], [])
        self.assertEqual(sentinel.read_text(), 'unrelated data')

    def test_runtime_disk_floor_terminates_and_reaps_child_without_deleting_data(self):
        root, args, plan, events = self.process_fixture(sleep_seconds=30)
        sentinel = root / 'keep-me.txt'; sentinel.write_text('unrelated data')
        actual_popen = study.subprocess.Popen; children = []; checks = []
        def launch(*a, **k):
            self.assertIs(k.get('start_new_session'), True)
            child = actual_popen(*a, **k); children.append(child); return child
        def disk_usage(path):
            checks.append(path)
            return SimpleNamespace(total=10*1024**3, used=8*1024**3,
                free=3*1024**3 if len(checks) == 1 else int(.9*1024**3))
        with patch.object(study.shutil, 'disk_usage', side_effect=disk_usage), \
                patch.object(study.subprocess, 'Popen', side_effect=launch):
            self.assertEqual(study.run_queue(args, plan), 75)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        with self.assertRaises(ProcessLookupError): os.kill(children[0].pid, 0)
        status = study.load(Path(plan['output']) / 'queue-status.json')
        self.assertEqual(status['status'], 'blocked_disk_space')
        self.assertIn('Runtime disk guard', status['disk']['scope'])
        self.assertEqual(status['disk']['next_case'], 'x01_off')
        self.assertFalse(status['completed'])
        self.assertEqual(status['completed_cases'], [])
        self.assertEqual(sentinel.read_text(), 'unrelated data')

    def test_sigterm_reaps_active_child_and_preserves_interrupted_queue(self):
        root, args, plan, events = self.process_fixture(sleep_seconds=30)
        status_path = Path(plan['output']) / 'queue-status.json'
        observed = []
        def interrupt():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if status_path.is_file():
                    state = study.load(status_path)
                    active = state.get('active_case')
                    if isinstance(active, dict) and active.get('child_pid'):
                        observed.append(active['child_pid'])
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                time.sleep(.005)
        thread = threading.Thread(target=interrupt, daemon=True)
        thread.start()
        self.assertEqual(study.run_queue(args, plan), 128 + signal.SIGTERM)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(observed), 1)
        with self.assertRaises(ProcessLookupError): os.kill(observed[0], 0)
        status = study.load(status_path)
        self.assertEqual(status['status'], 'interrupted')
        self.assertFalse(status['completed'])
        self.assertIsNone(status['active_case'])
        self.assertEqual(status['completed_cases'], [])


if __name__ == '__main__':
    unittest.main()
