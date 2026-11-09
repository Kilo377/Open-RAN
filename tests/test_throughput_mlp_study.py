"""Actual packets, fitted action coverage and fixed ablations define X08 verdicts."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_throughput_mlp_study import (APP, ARMS, POPULATION,
    actual_packet_metrics, audit_measured_collection, collection_eval_compatibility,
    command_for, evaluate_study, main, measured_data_sha256, parse_args, readiness,
    prepare_measured_recovery, refit_measured_collection, training_evidence)


def optimizer_proof():
    return dict(objective='half_mean_standardized_squared_error_plus_L2',
        gradient_convention='project_corrected_mean_output', sklearn_version='1.7.2',
        source_sha256='b'*64, relative_tolerance=1e-5, consistency_max_relative_error=0.,
        checks=[dict(outputs=n,alpha=a,native_relative_error=8. if n==9 else 0.,corrected_relative_error=0.)
                for n in (1,9) for a in (0.,1.)])


class TestThroughputMLPStudy(unittest.TestCase):
    def fixture(self, native_complete=1, mlp_complete=2, late_urllc=False, overflow_urllc=False, policy='guarded'):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        root=Path(directory.name);folders={a:root/a for a in ARMS}
        stamp=dict(samples=24,trained_samples=24,fit_count=1,prediction_batches=10,fit_parameters={'activation':'tanh'},
                   sha256='frozen-model-bytes',optimizer_verified=True,optimizer_metadata=optimizer_proof())
        for arm,folder in folders.items():
            folder.mkdir();mode='off' if arm=='off' else 'frozen' if arm=='mlp' else 'fixed'
            profiles=[dict(ue_id=u,slice_id='urllc' if u in ('ue_2','ue_6') else 'embb',
                           latency_budget_ms=10 if u in ('ue_2','ue_6') else 100) for u in sorted(POPULATION)]
            app=dict(implementation_id=APP,parameters=dict(mode=mode,fixed_action=1 if arm=='neutral' else 3,policy=policy))
            cfg=dict(manifest=dict(ues=[dict(ue_id=u) for u in sorted(POPULATION)],cells=[{}, {}, {}],
                total_slots=200,slot_duration_ms=.5,rng_seed=55),configs=dict(ue=dict(ues=profiles),
                phy=dict(rt_sys={}),sim={},cell={},beamforming={},ric=dict(xapps=[] if arm=='off' else [app])))
            (folder/'config_snapshot.json').write_text(json.dumps(cfg))
            events=[];packets={}
            for uid in sorted(POPULATION):
                n=native_complete if arm in ('off','neutral') else mlp_complete if arm=='mlp' else 2
                if isinstance(n,dict):n=n[uid]
                rejected=overflow_urllc and arm=='mlp' and uid=='ue_2'
                islate=late_urllc and arm=='mlp' and uid=='ue_2'
                pids=[uid+':1',uid+':2']
                events.extend([dict(kind='arrival',ue_id=uid,packet_id=pids[0],slot_id=1,bytes=100,admitted=not rejected),
                               dict(kind='arrival',ue_id=uid,packet_id=pids[1],slot_id=195,bytes=100,admitted=True)])
                complete=[]
                if n>=1 and not rejected:
                    latency=20. if islate else 1.;complete.append(latency)
                    events.append(dict(kind='delivery',ue_id=uid,packet_id=pids[0],slot_id=41 if islate else 3,
                                       bytes=100,completed=True,latency_ms=latency))
                if n==2:
                    complete.append(2.)
                    events.append(dict(kind='delivery',ue_id=uid,packet_id=pids[1],slot_id=199,
                                       bytes=100,completed=True,latency_ms=2.))
                overdue=(n==0 or islate) and not rejected
                if overdue:
                    events.append(dict(kind='deadline_miss',ue_id=uid,packet_id=pids[0],slot_id=25))
                admitted=1 if rejected else 2
                count=len(complete)
                packets[uid]=dict(initial_bytes=0,offered_bytes=200,overflow_bytes=100 if rejected else 0,
                    overflow_packets=int(rejected),admitted_packets=admitted,delivered_bytes=count*100,
                    completed_packets=count,pending_bytes=(admitted-count)*100,pending_packets=admitted-count,
                    completion_latency_ms_sum=sum(complete),deadline_missed_packets=int(overdue))
            summary=dict(packet_metrics=packets,xapp_library=dict(instances=[]))
            receipts=[]
            if arm!='off':
                weights={u:1. for u in POPULATION}
                if arm!='neutral':weights['ue_8']=2.
                summary['xapp_library']['instances']=[dict(implementation_id=APP,
                    writable_actions=['scheduler_mode','ue_scheduler_weights'],active_cell_ids=['cell_0'])]
                summary['throughput_xapp']=dict(stamp,prediction_batches=12,model_sha256=stamp['sha256']) if arm=='mlp' else dict(mode='fixed')
                summary['throughput_xapp'].update(policy=policy,constraints_active=arm=='mlp' and policy=='guarded')
                receipt=dict(status='applied',provenance=[dict(implementation_id=APP)],
                    fields=[dict(field='scheduler_mode',value='latency_aware'),dict(field='ue_scheduler_weights',value=weights)],
                    after={'cell_0':dict(scheduler_mode='latency_aware',ue_scheduler_weights=weights)})
                receipts=[receipt]
                (folder/'throughput_mlp_events.jsonl').write_text(json.dumps(dict(kind='executed_policy_confirmed'))+'\n')
            (folder/'final_summary.json').write_text(json.dumps(summary))
            (folder/'control_receipts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
            with gzip.open(folder/'packet_events.jsonl.gz','wt') as f:
                for row in events:f.write(json.dumps(row)+'\n')
            with gzip.open(folder/'rt_sys_trace.jsonl.gz','wt') as f:
                for slot in range(1,201):
                    f.write(json.dumps(dict(slot_id=slot,traffic_arrivals={u:dict(exogenous_bytes=100 if slot in (1,195) else 0,mec_result_bytes=0) for u in POPULATION}))+'\n')
        return folders,stamp

    def test_fixed_and_neutral_are_separate_from_learned_gain(self):
        result=evaluate_study(*self.fixture())
        self.assertTrue(result['controller_effective'])
        self.assertAlmostEqual(result['network_goodput_gain_vs_rr']['mlp'],1.)
        self.assertEqual(result['mlp_gain_vs_fixed'],0.)
        self.assertFalse(result['positive_ml_advantage_vs_fixed'])
        self.assertEqual(result['arms']['mlp']['users']['ue_2']['completed_packet_p95_latency_ms'],1.95)

    def test_large_network_gain_cannot_hide_a_starved_ue(self):
        delivered={u:2 for u in POPULATION};delivered['ue_0']=0
        result=evaluate_study(*self.fixture(mlp_complete=delivered))
        self.assertGreater(result['network_goodput_gain_vs_rr']['mlp'],.05)
        self.assertFalse(result['guards']['mlp']['per_ue_delivery']['ue_0'])
        self.assertEqual(result['classification'],'qos_guard_failed')
        self.assertTrue(result['objective_target_met'])
        self.assertFalse(result['offline_qos_guard_passed'])
        self.assertFalse(result['controller_effective'])

    def test_throughput_policy_reports_objective_gain_and_qos_harm_separately(self):
        delivered={u:2 for u in POPULATION};delivered['ue_0']=0
        result=evaluate_study(*self.fixture(mlp_complete=delivered,policy='throughput'),policy='throughput')
        self.assertEqual(result['throughput_xapp_policy'],'throughput')
        self.assertEqual(result['classification'],'throughput_target_met_with_qos_impact')
        self.assertEqual(result['offline_qos_guard_role'],'reported_research_outcome')
        self.assertTrue(result['objective_target_met'])
        self.assertFalse(result['offline_qos_guard_passed'])
        self.assertFalse(result['controller_effective'])
        self.assertFalse(result['guards']['mlp']['per_ue_delivery']['ue_0'])
        self.assertEqual(result['arms']['mlp']['users']['ue_2']['completed_packet_p95_latency_ms'],1.95)

    def test_throughput_policy_preserves_unique_original_deadline_failure_audit(self):
        result=evaluate_study(*self.fixture(late_urllc=True,policy='throughput'))
        self.assertEqual(result['classification'],'throughput_target_met_with_qos_impact')
        self.assertTrue(result['objective_target_met'])
        self.assertFalse(result['offline_qos_guard_passed'])
        ue=result['arms']['mlp']['users']['ue_2']
        self.assertEqual(ue['unique_deadline_failed_packets'],1)
        self.assertEqual(ue['mature_failure_fraction'],1.)
        self.assertEqual(result['guards']['mlp']['urllc_deadline_failure']['ue_2']['original_budget_ms'],10.)

    def test_throughput_policy_does_not_label_subtarget_gain_as_success(self):
        result=evaluate_study(*self.fixture(native_complete=2,policy='throughput'))
        self.assertFalse(result['objective_target_met'])
        self.assertTrue(result['offline_qos_guard_passed'])
        self.assertFalse(result['controller_effective'])
        self.assertEqual(result['classification'],'throughput_below_target_headroom_limited')
        delivered={u:2 for u in POPULATION};delivered['ue_0']=0
        result=evaluate_study(*self.fixture(native_complete=2,mlp_complete=delivered,policy='throughput'))
        self.assertFalse(result['objective_target_met'])
        self.assertFalse(result['offline_qos_guard_passed'])
        self.assertEqual(result['classification'],'throughput_below_target_with_qos_impact')

    def test_policy_config_summary_and_requested_policy_must_match(self):
        for arm in ('neutral','fixed','mlp'):
            with self.subTest(arm=arm):
                folders,stamp=self.fixture(policy='throughput');path=folders[arm]/'final_summary.json'
                summary=json.loads(path.read_text());summary['throughput_xapp']['policy']='guarded'
                path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError,'policy mismatch'):
                    evaluate_study(folders,stamp,policy='throughput')
        folders,stamp=self.fixture(policy='throughput')
        with self.assertRaisesRegex(ValueError,'policy mismatch'):
            evaluate_study(folders,stamp,policy='guarded')

    def test_fixed_ablation_constraints_are_inactive_for_either_policy(self):
        for policy in ('guarded','throughput'):
            with self.subTest(policy=policy):
                folders,stamp=self.fixture(policy=policy)
                result=evaluate_study(folders,stamp,policy=policy)
                for arm in ('neutral','fixed'):
                    self.assertFalse(result['policy_evidence'][arm]['constraints_active'])
                self.assertEqual(result['policy_evidence']['mlp']['constraints_active'],policy=='guarded')
                if policy=='throughput':
                    self.assertEqual(result['classification'],'throughput_target_met_qos_preserved')
                path=folders['fixed']/'final_summary.json';summary=json.loads(path.read_text())
                summary['throughput_xapp']['constraints_active']=True;path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError,'constraints mismatch'):
                    evaluate_study(folders,stamp,policy=policy)

    def test_late_completed_and_logged_overdue_packet_are_counted_once(self):
        result=evaluate_study(*self.fixture(late_urllc=True))
        ue=result['arms']['mlp']['users']['ue_2']
        self.assertEqual(ue['late_completed_packets'],1)
        self.assertEqual(ue['logged_deadline_misses'],1)
        self.assertEqual(ue['unique_deadline_failed_packets'],1)
        self.assertEqual(ue['mature_failure_fraction'],1.)
        self.assertEqual(result['classification'],'qos_guard_failed')

    def test_dropping_url_packet_cannot_improve_the_guard(self):
        result=evaluate_study(*self.fixture(overflow_urllc=True))
        ue=result['arms']['mlp']['users']['ue_2']
        self.assertEqual(ue['overflow_packets'],1)
        self.assertEqual(ue['mature_failure_fraction'],1.)
        self.assertTrue(result['guards']['mlp']['per_ue_delivery']['ue_2'])
        self.assertFalse(result['guards']['mlp']['urllc_deadline_failure']['ue_2']['passed'])

    def test_pending_inside_its_deadline_is_reported_as_censored(self):
        result=evaluate_study(*self.fixture())
        ue=result['arms']['off']['users']['ue_2']
        self.assertEqual(ue['pending_within_deadline_packets'],1)
        self.assertEqual(ue['censored_offered_packets'],1)
        self.assertEqual(ue['mature_offered_packets'],1)
        self.assertEqual(ue['mature_failure_fraction'],0.)

    def test_no_native_offered_headroom_is_an_honest_weak_result(self):
        result=evaluate_study(*self.fixture(native_complete=2))
        self.assertTrue(result['native_recorded_offered_headroom_below_target'])
        self.assertFalse(result['controller_effective'])
        self.assertEqual(result['classification'],'below_target_headroom_limited')

    def test_model_must_make_new_frozen_predictions(self):
        folders,stamp=self.fixture();stamp['prediction_batches']=12
        result=evaluate_study(folders,stamp)
        self.assertEqual(result['classification'],'insufficient_control_evidence')

    def test_throughput_objective_requires_genuine_valid_control(self):
        folders,stamp=self.fixture(policy='throughput');stamp['prediction_batches']=12
        result=evaluate_study(folders,stamp)
        self.assertGreater(result['network_goodput_gain_vs_rr']['mlp'],.05)
        self.assertFalse(result['objective_target_met'])
        self.assertFalse(result['controller_effective'])
        self.assertTrue(result['offline_qos_guard_passed'])
        self.assertEqual(result['classification'],'insufficient_control_evidence')

    def test_actual_loaded_model_hash_must_match_frozen_artifact(self):
        folders,stamp=self.fixture();path=folders['mlp']/'final_summary.json'
        summary=json.loads(path.read_text());summary['throughput_xapp']['model_sha256']='different-loaded-forest'
        path.write_text(json.dumps(summary))
        self.assertEqual(evaluate_study(folders,stamp)['classification'],'insufficient_control_evidence')

    def test_mu_cell_and_beamforming_inputs_must_match(self):
        folders,stamp=self.fixture();path=folders['mlp']/'config_snapshot.json'
        original=json.loads(path.read_text())
        for section in ('sim','cell','beamforming'):
            with self.subTest(section=section):
                changed=deepcopy(original);changed['configs'][section]['changed']='not-matched'
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'mismatch'):evaluate_study(folders,stamp)

    def test_actual_weight_mismatch_is_not_a_real_control_effect(self):
        folders,stamp=self.fixture();path=folders['mlp']/'control_receipts.jsonl'
        row=json.loads(path.read_text());row['after']['cell_0']['ue_scheduler_weights']=dict(row['after']['cell_0']['ue_scheduler_weights'],ue_8=1.)
        path.write_text(json.dumps(row)+'\n')
        self.assertEqual(evaluate_study(folders,stamp)['classification'],'insufficient_control_evidence')

    def test_wrong_neutral_ablation_and_changed_radio_are_rejected(self):
        folders,stamp=self.fixture();path=folders['neutral']/'config_snapshot.json'
        cfg=json.loads(path.read_text());cfg['configs']['ric']['xapps'][0]['parameters']['fixed_action']=3
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'ablation'):evaluate_study(folders,stamp)
        cfg['configs']['ric']['xapps'][0]['parameters']['fixed_action']=1;cfg['configs']['phy']['changed']=True
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'mismatch'):evaluate_study(folders,stamp)

    def test_readiness_uses_fitted_prefix_not_extra_saved_rows(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);path=Path(directory.name)/'model.json'
        onehot=lambda i:[float(i==j) for j in range(4)]
        rows=[onehot(i) for i in range(4) for _ in range(6)]
        model=dict(format='chongyu.online-throughput-mlp',version=2,feature_names=['action_'+str(i) for i in range(4)],
            x=rows,y=[[0.]*9]*24,trained_samples=24,fit_count=1,prediction_batches=0,fit_parameters={},
            fit_diagnostics=dict(converged=True),coefs=[[]],x_scaler={'mean':[]},y_scaler={'mean':[]},optimizer_metadata=optimizer_proof())
        path.write_text(json.dumps(model));self.assertTrue(readiness(path)['ready'])
        legacy=deepcopy(model);legacy['version']=1;legacy.pop('optimizer_metadata')
        path.write_text(json.dumps(legacy));self.assertFalse(readiness(path)['ready'])
        model['x']=[onehot(0)]*7+[onehot(1)]*6+[onehot(2)]*6+[onehot(3)]*5+[onehot(3)]
        path.write_text(json.dumps(model));self.assertFalse(readiness(path)['ready'])

    def training_fixture(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);folder=Path(directory.name)
        snapshot={'cell_0':dict(scheduler_mode='round_robin',ue_scheduler_weights={u:1. for u in POPULATION})}
        targets=[.06]+[.01]*6+[30.,30.]
        events=[dict(kind='executed_policy_confirmed',action=0,produced_slot=20,executed_snapshot=snapshot,
                     owners={'cell_0':dict(scheduler_mode=APP,ue_scheduler_weights=APP)}),
            dict(kind='measured_sample',action=0,features=[1.,0.,0.,0.],targets=targets,snapshot=snapshot,
                produced_start_slot=60,produced_end_slot=260,duration_s=.1,membership={u:'cell_0' for u in POPULATION},
                delivered_bits_by_ue={u:1000 for u in POPULATION},offered_bits_by_ue={u:1100 for u in POPULATION},
                deadline_misses_by_ue={u:1 for u in ('ue_2','ue_6')},admitted_packets_by_ue={u:10 for u in ('ue_2','ue_6')},
                overflow_packets_by_ue={u:1 for u in ('ue_2','ue_6')},completed_packets_by_ue={u:9 for u in ('ue_2','ue_6')},
                late_completed_packets_by_ue={u:1 for u in ('ue_2','ue_6')})]
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        (folder/'config_snapshot.json').write_text(json.dumps(dict(manifest=dict(slot_duration_ms=.5,total_slots=1000))))
        model=folder/'model.json';model.write_text(json.dumps(dict(x=[events[-1]['features']],y=[targets])))
        return folder,model,events

    def test_training_labels_are_mbps_and_event_rates_not_predictions(self):
        folder,model,_=self.training_fixture()
        self.assertEqual(training_evidence(folder,model)['new_actual_rows'],1)

    def test_observed_windows_after_training_cap_are_audited_but_not_claimed_as_fitted_data(self):
        folder,model,events=self.training_fixture()
        samples=[]
        for index in range(42):
            row=deepcopy(events[-1]);row.update(produced_start_slot=60+index*200,
                                               produced_end_slot=260+index*200)
            samples.append(row)
        (folder/'config_snapshot.json').write_text(json.dumps(dict(manifest=dict(slot_duration_ms=.5,total_slots=10000))))
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [events[0]]+samples))
        model.write_text(json.dumps(dict(x=[r['features'] for r in samples[:40]],y=[r['targets'] for r in samples[:40]])))
        result=training_evidence(folder,model)
        self.assertEqual(result['new_actual_rows'],40)
        self.assertEqual(result['observed_measured_windows'],42)
        self.assertEqual(result['measured_windows_not_added_due_to_row_cap'],2)
        samples[-1]['targets'][-1]=.3
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [events[0]]+samples))
        with self.assertRaisesRegex(ValueError,'actual delivered'):training_evidence(folder,model)

    def test_training_labels_cannot_be_changed_to_packet_ratios(self):
        folder,model,events=self.training_fixture();events[-1]['targets'][-1]=.3
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        with self.assertRaisesRegex(ValueError,'actual delivered'):training_evidence(folder,model)

    def test_training_waits_for_actual_confirmation_and_washin(self):
        folder,model,events=self.training_fixture();events[0]['produced_slot']=50
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        with self.assertRaisesRegex(ValueError,'wash-in'):training_evidence(folder,model)

    def test_idle_cell_requires_real_mode_owner_but_no_nonexistent_weight_owner(self):
        folder,model,events=self.training_fixture()
        for row in events:
            key='executed_snapshot' if row['kind']=='executed_policy_confirmed' else 'snapshot'
            row[key]['cell_1']=dict(scheduler_mode='round_robin',ue_scheduler_weights={})
        events[0]['owners']['cell_1']={'scheduler_mode':APP}
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        self.assertEqual(training_evidence(folder,model)['new_actual_rows'],1)
        events[0]['owners']['cell_1']={}
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        with self.assertRaisesRegex(ValueError,'ownership'):training_evidence(folder,model)

    def test_occupied_cell_still_requires_actual_weight_owner(self):
        folder,model,events=self.training_fixture()
        del events[0]['owners']['cell_0']['ue_scheduler_weights']
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        with self.assertRaisesRegex(ValueError,'ownership'):training_evidence(folder,model)

    def test_dry_run_preserves_matched_load_and_does_not_schedule_211(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);root=Path(directory.name).resolve()
        output=root/'outputs/study';args=parse_args(['--output',str(output),'--traffic-load','2','--dry-run'])
        stream=io.StringIO()
        with patch('experiments.run_throughput_mlp_study.ROOT',root),patch('experiments.run_throughput_mlp_study.parse_args',return_value=args), \
                patch('experiments.run_throughput_mlp_study.subprocess.run') as jobs,redirect_stdout(stream):main()
        plan=json.loads(stream.getvalue());self.assertFalse(plan['independent_seed_211_scheduled'])
        self.assertFalse(plan['native_traffic']);self.assertEqual(len(plan['commands']),6)
        self.assertEqual(plan['throughput_xapp_policy'],'throughput')
        self.assertTrue(all(c[c.index('--traffic-load')+1]=='2.0' for c in plan['commands']))
        self.assertTrue(all(c[c.index('--throughput-xapp-policy')+1]=='throughput' for c in plan['commands']))
        jobs.assert_not_called();self.assertFalse(output.exists())

    def test_policy_cli_default_explicit_guarded_and_forwarding_for_all_modes(self):
        self.assertEqual(parse_args(['--output','outputs/new']).throughput_xapp_policy,'throughput')
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse_args(['--output','outputs/new','--throughput-xapp-policy','unrecognized'])
        for policy in ('guarded','throughput'):
            args=parse_args(['--output','outputs/new','--throughput-xapp-policy',policy])
            for mode in ('off','online','fixed','frozen'):
                with self.subTest(policy=policy,mode=mode):
                    command=command_for(args,Path('outputs/new'),Path('outputs/new/model.json'),mode,55,1.,mode)
                    self.assertEqual(command[command.index('--throughput-xapp-policy')+1],policy)


class TestMeasuredCollectionReuse(unittest.TestCase):
    def collection(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);folder=root/'online_training';folder.mkdir()
        model=root/'old_model.json';features=['action_'+str(i) for i in range(4)];targets=['target_'+str(i) for i in range(9)]
        rows=[];events=[];labels=[.06]+[.01]*6+[30.,30.]
        for i in range(24):
            action=i%4;onehot=[float(action==j) for j in range(4)];rows.append(onehot)
            mode='round_robin' if action==0 else 'latency_aware'
            snapshot={'cell_0':dict(scheduler_mode=mode,ue_scheduler_weights={u:1. for u in POPULATION})}
            confirmed=20+i*300;start=confirmed+40;end=start+200
            events.append(dict(kind='executed_policy_confirmed',action=action,produced_slot=confirmed,executed_snapshot=snapshot,
                               owners={'cell_0':dict(scheduler_mode=APP,ue_scheduler_weights=APP)}))
            events.append(dict(kind='measured_sample',action=action,features=onehot,targets=labels,snapshot=snapshot,
                produced_start_slot=start,produced_end_slot=end,duration_s=.1,membership={u:'cell_0' for u in POPULATION},
                delivered_bits_by_ue={u:1000 for u in POPULATION},offered_bits_by_ue={u:1100 for u in POPULATION},
                deadline_misses_by_ue={u:1 for u in ('ue_2','ue_6')},admitted_packets_by_ue={u:10 for u in ('ue_2','ue_6')},
                overflow_packets_by_ue={u:1 for u in ('ue_2','ue_6')},completed_packets_by_ue={u:9 for u in ('ue_2','ue_6')},
                late_completed_packets_by_ue={u:1 for u in ('ue_2','ue_6')}))
        payload=dict(format='chongyu.online-throughput-mlp',version=1,seed=173,feature_names=features,target_names=targets,
                     x=rows,y=[labels]*24,trained_samples=24,fit_count=1,prediction_batches=0,fit_parameters={})
        model.write_text(json.dumps(payload))
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        (folder/'network-replay.html').write_text('<html>Original audited collector</html>')
        cfg=dict(source_sha256='a'*64,manifest=dict(slot_duration_ms=.5,total_slots=8000,rng_seed=173,
            cells=[{}, {}, {}],ues=[dict(ue_id=u,trajectory=dict(times_s=[0.,4.],positions_m=[[0.,0.,0.],[4.,0.,0.]])) for u in sorted(POPULATION)]),
            configs=dict(ue={'arrival_rate_bps':100},cell={},beamforming={},sim=dict(rng_seed=173,total_slots=8000),
                         phy={'rt_sys':dict(rng_seed=173,channel_cache_root='old-cache')},
                         ric=dict(xapps=[dict(implementation_id=APP,parameters={'mode':'online'})])))
        (folder/'config_snapshot.json').write_text(json.dumps(cfg))
        (folder/'run_status.json').write_text(json.dumps(dict(completed=True,stage='completed',source_sha256='a'*64,
            replay=dict(audits_passed=True,html=str(folder/'network-replay.html')))))
        (folder/'final_summary.json').write_text(json.dumps({'throughput_xapp':dict(mode='online',samples=24)}))
        return root,folder,model,payload,cfg

    def factory(self, payload, callback=None):
        class FakeLearner:
            optimizer_verified=True
            def __init__(self,features,targets,seed):
                self.features,self.targets,self.seed=features,targets,seed
                self.x=[];self.y=[];self.fit_count=self.trained_samples=0
            def add(self,x,y):self.x.append(deepcopy(x));self.y.append(deepcopy(y))
            def fit(self):
                if callback:callback()
                self.fit_count+=1;self.trained_samples=len(self.x);return True
            def save(self,path):
                value=dict(payload,version=2,x=self.x,y=self.y,seed=self.seed,
                    fit_count=self.fit_count,trained_samples=self.trained_samples,optimizer_metadata=optimizer_proof(),
                    fit_diagnostics={'converged':True},coefs=[[]],x_scaler={'mean':[]},y_scaler={'mean':[]})
                Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))
        return FakeLearner

    def test_reuse_preserves_exact_rows_seed_original_bytes_and_fits_once(self):
        root,folder,model,payload,_=self.collection();old=model.read_bytes();output=root/'new'/'model.json'
        with patch('experiments.run_throughput_mlp_study.throughput_schema',return_value=(payload['feature_names'],payload['target_names'])), \
             patch('experiments.run_throughput_mlp_study.current_source_sha256',return_value='c'*64), \
             patch('experiments.run_throughput_mlp_study.new_measured_learner',side_effect=self.factory(payload)) as created:
            receipt=refit_measured_collection(folder,model,output,1.)
        created.assert_called_once_with(payload['feature_names'],payload['target_names'],173)
        self.assertEqual(model.read_bytes(),old)
        self.assertEqual(measured_data_sha256(json.loads(output.read_text())),measured_data_sha256(payload))
        self.assertEqual(receipt['corrected_fit_count'],1)
        self.assertEqual(receipt['new_physics_training_s'],0.)
        self.assertNotEqual(receipt['collection_source_sha256'],receipt['fitting_source_sha256'])
        self.assertTrue(receipt['new_artifact']['optimizer_verified'])

    def test_incomplete_collector_is_rejected_before_fit(self):
        _,folder,model,_,_=self.collection();path=folder/'run_status.json'
        status=json.loads(path.read_text());status['completed']=False;path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError,'completed'):audit_measured_collection(folder,model,1.)

    def test_model_rows_must_match_actual_collector_events(self):
        _,folder,model,payload,_=self.collection();payload['y'][0]=[999.]+payload['y'][0][1:];model.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'Saved MLP rows'):audit_measured_collection(folder,model,1.)

    def test_changed_original_artifact_during_refit_is_rejected(self):
        root,folder,model,payload,_=self.collection()
        with patch('experiments.run_throughput_mlp_study.throughput_schema',return_value=(payload['feature_names'],payload['target_names'])), \
             patch('experiments.run_throughput_mlp_study.current_source_sha256',return_value='c'*64), \
             patch('experiments.run_throughput_mlp_study.new_measured_learner',side_effect=self.factory(payload,lambda:model.write_text(model.read_text()+'\n'))):
            with self.assertRaisesRegex(ValueError,'Original measured'):refit_measured_collection(folder,model,root/'new.json',1.)

    def test_collector_eval_matches_model_inputs_not_seed_or_duration(self):
        _,folder,_,_,cfg=self.collection();evaluation=deepcopy(cfg)
        evaluation['manifest'].update(rng_seed=55,total_slots=2000)
        for user in evaluation['manifest']['ues']:
            user['trajectory']=dict(times_s=[0.,1.],positions_m=[[0.,0.,0.],[-1.,0.,0.]])
        evaluation['configs']['sim'].update(rng_seed=55,total_slots=2000)
        evaluation['configs']['phy']['rt_sys'].update(rng_seed=55,channel_cache_root='new-cache')
        result=collection_eval_compatibility({'collection_folder':str(folder)},evaluation)
        self.assertTrue(result['verified']);self.assertFalse(result['shared_trajectory_realization_equal_diagnostic'])
        evaluation['configs']['ue']['arrival_rate_bps']=200
        with self.assertRaisesRegex(ValueError,'business or radio'):collection_eval_compatibility({'collection_folder':str(folder)},evaluation)

    def test_measured_reuse_is_exclusive_with_skip_training(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse_args(['--output','outputs/new','--skip-training','--measured-training','outputs/old'])

    def test_measured_dry_run_schedules_no_new_training_physics(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name).resolve()
        args=parse_args(['--output',str(root/'outputs/new'),'--measured-training',str(root/'outputs/old/online_training'),
                         '--measured-model',str(root/'outputs/old/model.json'),'--dry-run'])
        stream=io.StringIO()
        with patch('experiments.run_throughput_mlp_study.ROOT',root),patch('experiments.run_throughput_mlp_study.parse_args',return_value=args), \
             patch('experiments.run_throughput_mlp_study.subprocess.run') as jobs,redirect_stdout(stream):main()
        plan=json.loads(stream.getvalue());self.assertEqual(len(plan['commands']),4)
        self.assertEqual(plan['measured_collection_refit']['fit_count'],1)
        jobs.assert_not_called()

    def recovery_history(self):
        root,_,model,payload,cfg=self.collection()
        actions=[0]*26+[1]*5+[2]*5+[3]*4
        features=payload['feature_names'];targets=payload['target_names'];label=payload['y'][0]
        payload.update(x=[[float(i==j) for j in range(4)] for i in actions],y=[deepcopy(label) for _ in actions],
                       trained_samples=0,fit_count=0,version=2,optimizer_metadata=None)
        model.write_text(json.dumps(payload));episodes=[]
        for index,(before,after,seconds) in enumerate(((0,26,4),(26,40,2))):
            folder=root/('episode_'+str(index));folder.mkdir();episodes.append(folder)
            config=deepcopy(cfg);config['manifest'].update(total_slots=int(seconds*2000),rng_seed=173+index)
            for user in config['manifest']['ues']:
                user['trajectory']=dict(times_s=[0.,float(seconds)],positions_m=[[0.,0.,0.],[float(seconds),0.,0.]])
            config['configs']['sim'].update(total_slots=int(seconds*2000),rng_seed=173+index)
            config['configs']['phy']['rt_sys']['rng_seed']=173+index
            events=self.recovery_events(actions[before:after],label)
            (folder/'config_snapshot.json').write_text(json.dumps(config))
            (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
            (folder/'network-replay.html').write_text('Original completed episode')
            (folder/'run_status.json').write_text(json.dumps(dict(completed=True,stage='completed',source_sha256='a'*64,
                replay=dict(audits_passed=True,html=str(folder/'network-replay.html')))))
            (folder/'final_summary.json').write_text(json.dumps(dict(throughput_xapp=dict(
                mode='online',samples=after,trained_samples=0,fit_count=0,maximum_measured_rows=40))))
        return root,episodes,model,payload,features,targets

    def recovery_events(self,actions,label):
        events=[]
        for index,action in enumerate(actions):
            snapshot={'cell_0':dict(scheduler_mode='round_robin' if action==0 else 'latency_aware',
                                    ue_scheduler_weights={u:1. for u in POPULATION})}
            confirmed=20+index*280;start=confirmed+40;end=start+200
            events.append(dict(kind='executed_policy_confirmed',action=action,produced_slot=confirmed,executed_snapshot=snapshot,
                               owners={'cell_0':dict(scheduler_mode=APP,ue_scheduler_weights=APP)}))
            events.append(dict(kind='measured_sample',action=action,features=[float(action==j) for j in range(4)],targets=label,snapshot=snapshot,
                produced_start_slot=start,produced_end_slot=end,duration_s=.1,membership={u:'cell_0' for u in POPULATION},
                delivered_bits_by_ue={u:1000 for u in POPULATION},offered_bits_by_ue={u:1100 for u in POPULATION},
                deadline_misses_by_ue={u:1 for u in ('ue_2','ue_6')},admitted_packets_by_ue={u:10 for u in ('ue_2','ue_6')},
                overflow_packets_by_ue={u:1 for u in ('ue_2','ue_6')},completed_packets_by_ue={u:9 for u in ('ue_2','ue_6')},
                late_completed_packets_by_ue={u:1 for u in ('ue_2','ue_6')}))
        return events

    def test_recovery_audits_both_original_caps_and_copies_all40_rows_unchanged(self):
        root,episodes,model,payload,features,targets=self.recovery_history();old=model.read_bytes();new=root/'recovery/model.json'
        with patch('experiments.run_throughput_mlp_study.throughput_schema',return_value=(features,targets)):
            lineage=prepare_measured_recovery(episodes,model,new,2.)
        self.assertEqual(model.read_bytes(),old);self.assertEqual(new.read_bytes(),old)
        self.assertEqual(lineage['original_action_samples'],{'0':26,'1':5,'2':5,'3':4})
        self.assertEqual([e['actual_recorded_row_cap'] for e in lineage['episodes']],[40,40])
        self.assertEqual(lineage['original_training_simulated_s'],6.)
        self.assertEqual(lineage['recovery_recorded_row_cap'],48)

    def test_recovery_preserves40_bad_rows_and_audits_48_cap_with_postcap_windows(self):
        root,episodes,old,payload,features,targets=self.recovery_history()
        folder=root/'recovery';folder.mkdir();model=folder/'model.json'
        actions=[3,0,3,0,1,0,2,0]+[0]*5
        events=self.recovery_events(actions,payload['y'][0]);samples=[e for e in events if e['kind']=='measured_sample']
        updated=dict(payload,x=payload['x']+[r['features'] for r in samples[:8]],y=payload['y']+[r['targets'] for r in samples[:8]],
            trained_samples=47,fit_count=1,coefs=[[]],x_scaler={'mean':[]},y_scaler={'mean':[]},optimizer_metadata=optimizer_proof())
        model.write_text(json.dumps(updated))
        (folder/'throughput_mlp_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        (folder/'config_snapshot.json').write_text(json.dumps(dict(manifest=dict(slot_duration_ms=.5,total_slots=4000))))
        (folder/'final_summary.json').write_text(json.dumps(dict(throughput_xapp=dict(maximum_measured_rows=48))))
        proof=training_evidence(folder,model,payload)
        self.assertEqual(proof['fixed_training_row_cap'],48)
        self.assertEqual(proof['new_actual_rows'],8)
        self.assertEqual(proof['observed_measured_windows'],13)
        self.assertEqual(proof['measured_windows_not_added_due_to_row_cap'],5)
        self.assertTrue(readiness(model)['ready'])
        updated['x'][0]=[0.,1.,0.,0.];model.write_text(json.dumps(updated))
        with self.assertRaisesRegex(ValueError,'existing actual MLP data'):training_evidence(folder,model,payload)

    def test_wrong_historical_cap_is_not_relabelled_as_new48(self):
        root,episodes,model,_,features,targets=self.recovery_history()
        path=episodes[0]/'final_summary.json';summary=json.loads(path.read_text())
        summary['throughput_xapp']['maximum_measured_rows']=48;path.write_text(json.dumps(summary))
        with patch('experiments.run_throughput_mlp_study.throughput_schema',return_value=(features,targets)):
            with self.assertRaisesRegex(ValueError,'historical row cap 40'):
                prepare_measured_recovery(episodes,model,root/'new.json',2.)

    def test_recovery_dry_run_has_one_2s_source_case_and_only_four_evaluations(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name).resolve()
        args=parse_args(['--output',str(root/'outputs/new'),'--resume-measured-model',str(root/'outputs/old/model.json'),
            '--resume-measured-episode',str(root/'outputs/old/first'),'--resume-measured-episode',str(root/'outputs/old/second'),
            '--resume-seconds','2','--traffic-load','2','--dry-run'])
        stream=io.StringIO()
        with patch('experiments.run_throughput_mlp_study.ROOT',root),patch('experiments.run_throughput_mlp_study.parse_args',return_value=args), \
             patch('experiments.run_throughput_mlp_study.subprocess.run') as jobs,redirect_stdout(stream):main()
        plan=json.loads(stream.getvalue());self.assertEqual(len(plan['commands']),5)
        first=plan['commands'][0]
        self.assertEqual(first[first.index('--seed')+1],'175')
        self.assertEqual(first[first.index('--duration-s')+1],'2.0')
        self.assertTrue(plan['measured_online_recovery']['no_further_extension'])
        jobs.assert_not_called()


if __name__=='__main__':unittest.main()
