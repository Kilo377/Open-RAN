"""Admit a new X12 only with its own locked-policy, real paired holdout report."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA='oran.x12-heldout-validation.v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(value,message):
    if not value:raise ValueError('Adaptive X12 release: '+message)

def number(v):return type(v)in(int,float) and math.isfinite(v)

def validate(active_release,recipe,environment,root):
    folder=Path(root)/'artifacts/xapps/chongyu'/recipe['model_release']['id']
    path=folder/'evidence/final-holdout-report.json'
    pin=active_release.get('final_report_sha256');selection_pin=active_release.get('selection_sha256')
    require(isinstance(pin,str) and re.fullmatch('[0-9a-f]{64}',pin) and sha(path)==pin,'final report checksum differs')
    selection_path=folder/'evidence/selection.json'
    require(isinstance(selection_pin,str) and re.fullmatch('[0-9a-f]{64}',selection_pin) and sha(selection_path)==selection_pin,'locked selection checksum differs')
    report=json.loads(path.read_text());selection=json.loads(selection_path.read_text())
    require(report.get('schema')==SCHEMA and report.get('role')=='holdout' and report.get('synthetic_fixture')is False,
            'a real MEC holdout report is required; prior power/throughput evidence cannot validate X12')
    model=active_release['models']['x12_mec']['sha256']
    require(report.get('selected_model_sha256')==selection.get('model_sha256')==model,'selected policy identity differs')
    require(report.get('selection_sha256')==selection_pin and selection.get('heldout_used_for_selection')is False,'holdout was used for policy selection')
    source_pin=active_release.get('source_archive_sha256')
    require(isinstance(source_pin,str) and re.fullmatch('[0-9a-f]{64}',source_pin) and report.get('source_archive_sha256')==source_pin,'study source archive identity differs')
    require(selection.get('evaluation_source_archive_sha256')==source_pin and
            isinstance(recipe.get('validated_source_sha256'),str) and re.fullmatch('[0-9a-f]{64}',recipe['validated_source_sha256']) and
            selection.get('evaluation_source_sha256')==recipe['validated_source_sha256'],
            'executed study source differs from the admitted runtime')
    seeds=report.get('registered_seeds')
    require(isinstance(seeds,list) and len(seeds)>=6 and len(set(seeds))==len(seeds) and all(type(s)is int for s in seeds) and
            selection.get('heldout_seeds')==seeds,'all six independently registered seeds are required')
    require(report.get('window')==dict(total_slots=24040,start_slot_exclusive=2040,end_slot_inclusive=22040,duration_s=10.,
            actual_terminal_followup_slots=2000,eligible_birth_exposure_s=9.9,measurement_basis='fixed_birth_cohort_actual_terminal_followup'),
            'full ten-second cohort or actual followup differs')
    require(report.get('short_window')==dict(start_slot_exclusive=2040,end_slot_inclusive=2540,duration_s=.25,
            eligible_birth_exposure_s=.15,measurement_basis='fixed_birth_cohort_actual_terminal_followup'),'startup birth cohort differs')
    comparisons=report.get('comparisons',[])
    require(len(comparisons)==len(seeds) and {r.get('seed') for r in comparisons}==set(seeds),'incomplete independent holdout comparisons')
    short,long=[],[];short_cost,long_cost=[],[]
    for row in comparisons:
        require(row.get('model_sha256')==model and row.get('role')=='holdout','comparison uses a different policy')
        prefix=row.get('prefix_certificate',{})
        require(prefix.get('passed')is True and prefix.get('off_network_prefix_sha256')==prefix.get('on_network_prefix_sha256') and
                isinstance(prefix.get('off_network_prefix_sha256'),str) and re.fullmatch('[0-9a-f]{64}',prefix['off_network_prefix_sha256']), 'native prefixes are not identical')
        execution=row.get('control_execution',{})
        require(execution.get('all_proposed_actions_verified')is True and execution.get('first_apply_slot')==2081 and
                execution.get('proposed_decisions',0)>=2 and execution.get('proposed_decisions')==execution.get('matched_normal_execution'), 'normal delayed X12 execution is unverified')
        for name,gains,costs in [('short',short,short_cost),('full',long,long_cost)]:
            interval=row.get(name,{})
            off,on=interval.get('off_ms'),interval.get('on_ms')
            require(interval.get('measurement_basis')=='fixed_birth_cohort_actual_terminal_followup' and
                    interval.get('off_followup_until_slot')==interval.get('on_followup_until_slot')==24040,
                    'actual terminal followup was not recorded symmetrically')
            require(number(off) and off>0 and number(on) and on>=0,'invalid observed task-latency mean')
            gain=100.*(off-on)/off
            require(number(interval.get('gain_percent')) and math.isclose(interval['gain_percent'],gain,rel_tol=1e-10,abs_tol=1e-9),'reported gain differs from observed means')
            require(interval.get('task_identities_equal')is True and interval.get('off_sample_count',0)>0 and interval.get('on_sample_count',0)>0,'mature offloaded task populations are unverified')
            require(all(type(interval.get(k))is int and interval[k]>=0 for k in
                        ('off_sample_count','on_sample_count','off_failed','on_failed','off_pending','on_pending')) and
                    interval['off_sample_count']+interval['off_failed']==interval['on_sample_count']+interval['on_failed'],
                    'task cohort accounting is incomplete')
            require(interval.get('off_pending',0)==interval.get('on_pending',0)==0,'unresolved MEC tasks cannot be hidden by a lower completion mean')
            cost_off,cost_on=interval.get('off_restricted_completion_ms'),interval.get('on_restricted_completion_ms')
            require(interval.get('restricted_completion_horizon_ms')==100. and number(cost_off) and 0<cost_off<=100. and
                    number(cost_on) and 0<=cost_on<=100. and number(interval.get('restricted_completion_reduction_percent')) and
                    math.isclose(interval['restricted_completion_reduction_percent'],100.*(cost_off-cost_on)/cost_off,abs_tol=1e-9),
                    'deadline-restricted completion accounting is unverified')
            costs.append(interval['restricted_completion_reduction_percent'])
            gains.append(gain)
    cases=report.get('case_evidence',[])
    selected={(r.get('seed'),r.get('actor')) for r in cases if r.get('actor') in ('OFF','selected')}
    require(selected=={(seed,actor) for seed in seeds for actor in ('OFF','selected')},'OFF/ON case evidence is incomplete')
    locked=datetime.fromisoformat(selection['locked_utc'])
    require(locked.tzinfo is not None,'selection timestamp must specify UTC')
    for case in cases:
        require(case.get('source_archive_sha256')==report['source_archive_sha256'] and case.get('complete')is True,'case source or completion differs')
        require(case.get('source_sha256')==recipe['validated_source_sha256'],'case executed another network/controller source')
        started=datetime.fromisoformat(case['started_utc'])
        require(case.get('contiguous_slots_verified')==24040 and started.tzinfo is not None and started>locked,'case was incomplete or ran before selection was locked')
        require(isinstance(case.get('trace_sha256'),str) and re.fullmatch('[0-9a-f]{64}',case['trace_sha256']),'missing retained raw trace digest')
    acceptance=report.get('acceptance',{})
    require(acceptance.get('passed')is True,'registered final acceptance has not passed')
    require(number(acceptance.get('mean_short_gain_percent')) and number(acceptance.get('mean_full_gain_percent')),'aggregate gains are missing')
    require(math.isclose(acceptance.get('mean_short_gain_percent',math.nan),sum(short)/len(short),abs_tol=1e-9) and
            math.isclose(acceptance.get('mean_full_gain_percent',math.nan),sum(long)/len(long),abs_tol=1e-9),'aggregate acceptance differs from individual seed results')
    require(sum(short)/len(short)>=5. and sum(long)/len(long)>=5. and
            sum(v>=-1e-9 for v in short)/len(short)>=.75 and sum(v>=-1e-9 for v in long)/len(long)>=.75,
            'registered mean-gain or nonregression criteria were not met')
    require(sum(short_cost)/len(short_cost)>=5. and sum(long_cost)/len(long_cost)>=5.,
            'a lower completion mean concealed worse deadline-restricted task service')
    evaluation=environment.get('model_evaluation',{})
    require(evaluation.get('source')=='actual_final_paired_report' and evaluation.get('report_sha256')==pin and
            evaluation.get('final_evaluated_actors')==['x12_mec'] and evaluation.get('all_six_newly_retested')is False,
            'inherited actor evidence must remain separate')
    return dict(window_verified=True,paired_controllers_verified=True,window_basis='actual_final_paired_report',
        observe_slots=20000,before_observe_slots=2000,baseline_slot=2040,
        paired_evaluated_actors=['mec_latency'],paired_evaluated_models=['x12_mec'],
        paired_report_sha256=pin,seeds=seeds,six_apps_newly_retested=False,
        immediate_window_slots=500,mean_short_gain_percent=sum(short)/len(short),
        immediate_measurement_basis='first150ms_birth_cohort_actual_terminal_followup',
        cohort_followup_verified=True,
        online_window_metric_verified=all(number(r['full'].get('off_window_ms')) and number(r['full'].get('on_window_ms')) and
            r['full'].get('off_window_pending')==r['full'].get('on_window_pending')==0 for r in comparisons),
        mean_full_gain_percent=sum(long)/len(long),report_url='/environment-report/')
