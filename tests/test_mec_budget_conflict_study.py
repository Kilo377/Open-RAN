"""Measured demand, learned coverage and executed arbitration define the case."""
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_mec_budget_conflict_study import (ARMS, BUDGET, ENERGY,
    POPULATION, arbitration_evidence, cohort_demand, evaluate_study, main,
    parse_args, training_evidence, training_readiness)


class TestMecBudgetConflictStudy(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        folders = {arm: root / arm for arm in ARMS}
        stamp = dict(samples=12, trained_samples=12, fit_count=1, prediction_batches=2)
        energy_stamp = dict(samples=24, trained_samples=24, fit_count=7)
        for arm, folder in folders.items():
            folder.mkdir()
            cfg = dict(manifest=dict(ues=[dict(ue_id=u) for u in POPULATION],
                cells=[{}, {}, {}], total_slots=2000, slot_duration_ms=.5, rng_seed=55),
                configs=dict(ue={}, phy=dict(rt_sys=dict(ue_services=dict(task_cohort_ms=100,
                    mec=dict(tasks_by_ue={'ue_4':dict(interval_ms=50,compute_gflop=.05,deadline_ms=100)})))),
                    ric=dict(direct_conflict_mitigation='uniform_random' if arm=='both_random' else 'priority')))
            (folder/'config_snapshot.json').write_text(json.dumps(cfg))
            p = 1. if arm in ('x01','both_random') else .5
            cohort = dict(index=1,start_s=.1,end_s=.2,mature_after_s=.3,generated=2,
                completed=2,on_time_completed=2,failed=0,pending=0,completion_latency_ms_sum=40.,
                mean_offload_fraction=p,offload_fraction=p,offloaded_generated=int(2*p),
                offloaded_completed=int(2*p),offloaded_on_time_completed=int(2*p),
                offloaded_failed=0,offloaded_pending=0,offloaded_completion_latency_ms_sum=20.*int(2*p))
            with gzip.open(folder/'rt_sys_trace.jsonl.gz','wt') as out:
                out.write(json.dumps(dict(slot_id=600,ue_services={u:dict(ue_task_cohorts=[cohort]) for u in ('ue_0','ue_4')}))+'\n')
            summary = dict(ue_services=dict(per_ue={u:dict(ue_energy_total_j=(.8 if arm=='x01' else .9 if arm=='both_random' else 1.)) for u in POPULATION}),
                packet_metrics={u:dict(delivered_bytes=10000) for u in POPULATION},
                xapp_library=dict(instances=[]))
            enabled = [] if arm=='off' else ['energy','budget'] if arm=='both_random' else ['energy'] if arm=='x01' else ['budget']
            receipts=[]
            for kind in enabled:
                ident=ENERGY if kind=='energy' else BUDGET
                fields=['ue_offload_fraction','ue_drx_configs'] if kind=='energy' else ['ue_offload_fraction']
                summary['xapp_library']['instances'].append(dict(implementation_id=ident,writable_actions=fields,active_cell_ids=['cell_2']))
                summary['ue_energy_xapp' if kind=='energy' else 'budget_xapp'] = (
                    dict(energy_stamp,rf_selected_actions=2) if kind=='energy' else dict(stamp,prediction_batches=4))
                log='energy_rf_events' if kind=='energy' else 'budget_qgbdt_events'
                (folder/(log+'.jsonl')).write_text(json.dumps(dict(kind='executed_policy_confirmed',ue_id='ue_4'))+'\n')
                if arm!='both_random':receipts.append(self.receipt(ident,1. if kind=='energy' else .5))
            decisions=[]
            if arm=='both_random':
                rng=random.Random('1701:direct-conflict-mitigation:ue_offload_fraction:ue_4')
                for epoch in (1,2):
                    draw=rng.random();winner=BUDGET if draw<.5 else ENERGY;value=.5 if winner==BUDGET else 1.
                    row=dict(reason='direct_conflict_random_selection',field='ue_offload_fraction',ue_id='ue_4',
                        cell_id='cell_2',window_index=epoch,generation=epoch,probability=.5,draw=draw,
                        mitigation_seed=1701,stream_key='ue_offload_fraction:ue_4',draw_index=epoch,
                        candidates=[dict(source=BUDGET,requested_value=.5),dict(source=ENERGY,requested_value=1.)],
                        winner=winner,selected_value=value)
                    decisions.append(row)
                    receipt=self.receipt(winner,value)
                    receipt['mitigation_records']=[dict(row,status='applied')]
                    receipts.append(receipt)
            (folder/'final_summary.json').write_text(json.dumps(summary))
            self.write_jsonl(folder/'control_receipts.jsonl',receipts)
            self.write_jsonl(folder/'rule_decisions.jsonl',decisions)
        return folders,energy_stamp,stamp

    @staticmethod
    def receipt(ident,p):
        return dict(status='applied',provenance=[dict(implementation_id=ident)],
            fields=[dict(field='ue_offload_fraction',value={'ue_4':p})],
            after={'cell_2':dict(ue_service_policies={'ue_4':dict(offload_fraction=p)})})

    @staticmethod
    def write_jsonl(path,rows):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def evaluate(self,fixture):
        return evaluate_study(*fixture,'frozen')

    def test_m1_and_random_mitigation_result_are_separate(self):
        result=self.evaluate(self.fixture())
        self.assertTrue(result['m1_observed'])
        self.assertEqual(result['classification'],'m1_conflict_observed')
        self.assertFalse(result['random_mitigation_outcome']['aggregate_expected_mec_budget_met'])
        self.assertTrue(result['random_mitigation_outcome']['task_qos_met'])
        self.assertEqual(result['actual_rcp_intersection'],['ue_offload_fraction'])

    def test_realized_random_work_is_not_the_expected_budget_test(self):
        fixture=self.fixture();folder=fixture[0]['budget']
        with gzip.open(folder/'rt_sys_trace.jsonl.gz','rt') as stream:row=json.loads(stream.readline())
        c=row['ue_services']['ue_4']['ue_task_cohorts'][0]
        c.update(offloaded_generated=2,offloaded_completed=2,offloaded_on_time_completed=2,offloaded_completion_latency_ms_sum=40.)
        result=cohort_demand([c],dict(interval_ms=50,compute_gflop=.05),.75)
        self.assertEqual(result['expected_mec_gflops'],.5)
        self.assertAlmostEqual(result['realized_mec_gflops'],1.)
        self.assertEqual(result['expected_budget_violating_cohorts'],0)

    def test_wrong_executed_value_invalidates_arbitration(self):
        fixture=self.fixture();folder=fixture[0]['both_random']
        rows=[json.loads(l) for l in (folder/'control_receipts.jsonl').read_text().splitlines()]
        old=rows[0]['after']['cell_2']['ue_service_policies']['ue_4']['offload_fraction']
        rows[0]['after']['cell_2']['ue_service_policies']['ue_4']['offload_fraction']=1. if old==.5 else .5
        self.write_jsonl(folder/'control_receipts.jsonl',rows)
        result=self.evaluate(fixture)
        self.assertFalse(result['arbitration']['valid'])
        self.assertEqual(result['classification'],'insufficient_control_evidence')

    def test_draw_must_choose_the_recorded_sorted_candidate(self):
        fixture=self.fixture();folder=fixture[0]['both_random']
        rows=[json.loads(l) for l in (folder/'rule_decisions.jsonl').read_text().splitlines()]
        rows[0]['draw']=.25
        self.write_jsonl(folder/'rule_decisions.jsonl',rows)
        self.assertFalse(arbitration_evidence(folder)['valid'])

    def test_structural_m1_does_not_require_the_energy_goal_to_succeed(self):
        fixture=self.fixture();folder=fixture[0]['x01']
        summary=json.loads((folder/'final_summary.json').read_text())
        summary['ue_services']['per_ue']['ue_4']['ue_energy_total_j']=1.
        (folder/'final_summary.json').write_text(json.dumps(summary))
        result=self.evaluate(fixture)
        self.assertTrue(result['structural_m1_observed'])
        self.assertFalse(result['individual_intent_case_evidence_passed'])
        self.assertEqual(result['classification'],'m1_conflict_observed')

    def test_budget_contract_is_aggregate_not_every_cohort(self):
        fixture=self.fixture();folder=fixture[0]['budget']
        with gzip.open(folder/'rt_sys_trace.jsonl.gz','rt') as stream:c=json.loads(stream.readline())['ue_services']['ue_4']['ue_task_cohorts'][0]
        a=deepcopy(c);a['mean_offload_fraction']=1.
        result=cohort_demand([a,c],dict(interval_ms=50,compute_gflop=.05),.75)
        self.assertEqual(result['expected_mec_gflops'],.75)
        self.assertEqual(result['expected_budget_violating_cohorts'],1)

    def test_all_one_winner_is_valid_finite_random_outcome(self):
        fixture=self.fixture();folder=fixture[0]['both_random']
        selections=[json.loads(l) for l in (folder/'rule_decisions.jsonl').read_text().splitlines()]
        # A single declared epoch is a valid finite coin outcome.
        receipts=[json.loads(l) for l in (folder/'control_receipts.jsonl').read_text().splitlines()]
        self.write_jsonl(folder/'rule_decisions.jsonl',selections[:1])
        self.write_jsonl(folder/'control_receipts.jsonl',receipts[:1])
        evidence=arbitration_evidence(folder)
        self.assertTrue(evidence['valid'])
        self.assertFalse(evidence['both_contenders_executed'])

    def test_frozen_data_changes_do_not_count_as_deployment(self):
        fixture=self.fixture();folder=fixture[0]['budget']
        summary=json.loads((folder/'final_summary.json').read_text())
        summary['budget_xapp']['samples']=13
        (folder/'final_summary.json').write_text(json.dumps(summary))
        self.assertFalse(self.evaluate(fixture)['valid_control_evidence'])

    def test_no_new_model_predictions_do_not_count_as_qgbdt_use(self):
        fixture=self.fixture();fixture[2]['prediction_batches']=4
        self.assertFalse(self.evaluate(fixture)['valid_control_evidence'])

    def test_original_task_demand_must_match(self):
        fixture=self.fixture();folder=fixture[0]['x01']
        cfg=json.loads((folder/'config_snapshot.json').read_text())
        cfg['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['ue_4']['interval_ms']=25
        (folder/'config_snapshot.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'mismatch'):self.evaluate(fixture)

    def test_readiness_requires_coverage_and_actual_fit(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);path=Path(tmp.name)/'model.json'
        model=dict(format='chongyu.online-qgbdt',version=1,feature_names=['candidate_offload_fraction'],
            x=[[0.]]*6+[[.5]]*6,y=[.9]*12,trained_samples=12,fit_count=1,forest=[{}],fit_parameters={},prediction_batches=0)
        path.write_text(json.dumps(model));self.assertTrue(training_readiness(path)['ready'])
        model.update(forest=[],trained_samples=0,fit_count=0)
        path.write_text(json.dumps(model));self.assertFalse(training_readiness(path)['ready'])
        model.update(forest=[{}],trained_samples=11,fit_count=1,x=model['x'][:-1])
        path.write_text(json.dumps(model));self.assertFalse(training_readiness(path)['ready'])

    def training_fixture(self):
        fixture=self.fixture();folder=fixture[0]['budget'];model=folder/'model.json'
        with gzip.open(folder/'rt_sys_trace.jsonl.gz','rt') as stream:c=json.loads(stream.readline())['ue_services']['ue_4']['ue_task_cohorts'][0]
        rows=[dict(kind='executed_policy_confirmed',cohort=1,p=.5,owner=BUDGET,produced_slot=220),
              dict(kind='measured_sample',cohort=c,p=.5,features=[.1,.5],utility=.95,produced_slot=600)]
        self.write_jsonl(folder/'budget_qgbdt_events.jsonl',rows)
        model.write_text(json.dumps(dict(x=[[struct.unpack('f',struct.pack('f',.1))[0],.5]],y=[.95])))
        return folder,model,rows

    def test_training_rows_are_actual_mature_labels_with_float32_features(self):
        folder,model,_=self.training_fixture()
        result=training_evidence(folder,model)
        self.assertEqual(result['new_measured_rows'],1)
        self.assertTrue(result['actual_mature_labels_verified'])

    def test_training_cannot_use_predictions_as_labels(self):
        folder,model,rows=self.training_fixture();rows[-1]['utility']=.99
        self.write_jsonl(folder/'budget_qgbdt_events.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'measured utility'):training_evidence(folder,model)

    def test_dry_run_creates_no_outputs_and_never_starts_jobs(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name).resolve()
        output=root/'outputs/new';args=parse_args(['--output',str(output),'--energy-model',str(root/'outputs/energy.json'),'--dry-run'])
        out=io.StringIO()
        with patch('experiments.run_mec_budget_conflict_study.ROOT',root), \
                patch('experiments.run_mec_budget_conflict_study.parse_args',return_value=args), \
                patch('experiments.run_mec_budget_conflict_study.subprocess.run') as jobs,redirect_stdout(out):main()
        jobs.assert_not_called();self.assertFalse(output.exists())
        plan=json.loads(out.getvalue());self.assertEqual(len(plan['commands']),6)
        self.assertIn('1701',plan['commands'][-1])
        self.assertIn('uniform_random',plan['commands'][-1])
        self.assertNotIn('uniform_random',plan['commands'][0])


if __name__=='__main__':unittest.main()
