"""Exhaustive empirical evaluation with one independent run per configured host.

Queue order is catalog order, never a conflict/utility ranking. Each member
gets a fresh OFF baseline and an independent normal-RIC deployment run.
"""
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from threading import Thread,current_thread
import time
from uuid import uuid4

from smo.intent_language import IntentError
from smo.result_evaluation import enrich_batch

BATCH_TERMINAL={'complete','complete_with_failures','paused'}
JOB_TERMINAL={'complete','failed','cancelled'}
POLL_SECONDS=.25


class BatchEvaluator:
    def __init__(self,manager):
        self.manager=manager;self.items={};self.latest=None;self.thread=None;self.restart=False
        self.baseline_audit_cache={}
        self.root=manager.output/'batches'if manager.config else None
        if self.root:
            self.root.mkdir(exist_ok=True)
            for p in self.root.glob('*/batch.json'):
                row=json.loads(p.read_text());self.items[row['id']]=row
            if self.items:
                self.latest=max(self.items,key=lambda i:self.items[i]['created_unix_s'])
                # Reconcile a crash between persisting a child job and its queue
                # reference. Child metadata is durable before SSH launch.
                batch=self.items[self.latest]
                for index,member in enumerate(batch['members']):
                    matches=[j for j in manager.jobs.values()if j.get('batch_id')==batch['id']and j.get('batch_index')==index]
                    if matches:
                        job=max(matches,key=lambda j:j['created_unix_s'])
                        member['worker_id']=job['worker_id'];member['worker_label']=job.get('worker_label')
                        if job['id']not in member['attempts']:
                            member['attempts'].append(job['id'])
                            if member['phase']!='complete':member['job_id']=job['id']
                if batch['phase']not in BATCH_TERMINAL:
                    batch.update(worker_ids=list(manager.workers),concurrency=len(manager.workers),
                        execution='parallel_independent_runs'if len(manager.workers)>1 else'sequential_independent_runs')
                    self.save(batch)

    def active(self):
        return bool(self.latest and self.items[self.latest]['phase']not in BATCH_TERMINAL)

    def status(self):
        with self.manager.lock:
            if not self.latest:return None
            row=deepcopy(self.items[self.latest]);members=row['members']
            row['counts']={p:sum(m['phase']==p for m in members)for p in('pending','running','complete','failed','cancelled')}
            row['total']=len(members)
            result=enrich_batch(row)
            result['controlled_comparison']=self.audit_baseline(row)
            result['evaluation_origin_audit']=self.audit_origin(row)
            return result

    def audit_origin(self,row):
        """Label legacy context gaps honestly; this never changes a score."""
        completed=[m for m in row.get('members',[])if m.get('phase')=='complete'and m.get('comparison')]
        proof=row.get('context',{}).get('evaluation_origin')
        differences=[]
        before=row.get('baseline_reference',{})
        for record in row.get('records',[]):
            for target in record.get('ranges',[]):
                anchor=target.get('current');actual=before.get(target['kpi'])
                if type(anchor)in(int,float)and type(actual)in(int,float)and not math.isclose(anchor,actual,rel_tol=1e-7,abs_tol=1e-8):
                    differences.append(dict(intent_id=record['id'],kpi=target['kpi'],intent_reference=anchor,before_value=actual))
        if proof is None:
            return dict(status='legacy_unaligned',reason='legacy_batch_has_no_prepared_intent_origin',
                completed_runs=len(completed),verified_runs=0,legacy_anchor_differences=differences,
                intent_bounds_rebased=False,after_kpis_used=False)
        verified=[m for m in completed if m['comparison'].get('evaluation_origin',{}).get('origin_id')==proof['origin_id']
            and m['comparison']['evaluation_origin'].get('verified_before_activation')is True]
        status='mismatch'if len(verified)!=len(completed)else'verified'if completed else'pending'
        return dict(status=status,origin_id=proof['origin_id'],reason=None if status=='verified'else'evaluation_origin_pending'if status=='pending'else'evaluation_origin_verification_missing',
            completed_runs=len(completed),verified_runs=len(verified),legacy_anchor_differences=differences,
            intent_bounds_rebased=False,after_kpis_used=False)

    def audit_baseline(self,row):
        """Actual-experiment audit only. Never supplies inputs to risk scoring."""
        from smo.evaluation_reference import load
        evidence=load(row,self.manager.output,cache=self.baseline_audit_cache)
        return dict(status=evidence['status'],reason=evidence.get('reason'),
            reference=evidence.get('reference'),risk_inputs_used=False,
            purpose='independent_deployment_before_after_validation')

    def save(self,row):
        from smo.deployment import atomic
        with self.manager.lock:atomic(self.root/row['id']/'batch.json',row)

    def start(self,*,records,groups,observe_slots,context,snapshot,ranges,show_dashed):
        if not self.manager.config:raise IntentError('尚未配置仿真设备。')
        if not groups:raise IntentError('当前没有覆盖全部目标 KPI 的候选组合。')
        if type(observe_slots)is not int or not 1<=observe_slots<=20000:raise IntentError('窗口长度须为 1–20,000 个整数 slots。')
        signatures=[tuple(sorted(g['app_ids']))for g in groups]
        if len(signatures)!=len(set(signatures)):raise IntentError('候选目录有重复组合。')
        context=self.manager.execution_context(context,observe_slots,recipe=self.manager.runtime_recipe,
            default_origin=self.manager.runtime_origin)
        origin=context.get('evaluation_origin')
        with self.manager.lock:
            if self.active():raise IntentError('已有全部组合评估正在进行，请等待完成或先暂停。')
            if self.manager.busy_jobs():raise IntentError('已有仿真任务正在运行。')
            ident=uuid4().hex;(self.root/ident).mkdir()
            row=dict(id=ident,phase='queued',created_unix_s=time.time(),active_intent_ids=[r['id']for r in records],
                records=deepcopy(records),ranges=deepcopy(ranges),context=deepcopy(context),snapshot=deepcopy(snapshot),
                observe_slots=observe_slots,seed=self.manager.config['seed'],show_dashed=show_dashed,
                worker_source_archive_sha256=self.manager.source_archive_sha256,
                worker_source_archive_path=self.manager.source_archive_path,
                catalog_sha256=hashlib.sha256(json.dumps(groups,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                worker_ids=list(self.manager.workers),concurrency=len(self.manager.workers),execution='parallel_independent_runs'if len(self.manager.workers)>1 else'sequential_independent_runs',automatic_selection=False,ranking_performed=False,
                conflict_quantification_performed=False,pause_requested=False,
                members=[dict(group_id=g['id'],app_ids=deepcopy(g['app_ids']),selected_names=deepcopy(g['selected_names']),
                    conflicts=deepcopy(g.get('conflict_summaries',[])),intent_coverage=deepcopy(g.get('intent_coverage',{})),coverage_paths=deepcopy(g.get('coverage_paths',{})),phase='pending',job_id=None,attempts=[])for g in groups])
            if origin is not None:
                row['baseline_reference']={r['kpi']:origin['before_frame']['states'].get('kpi:'+r['kpi'],{}).get('value')for r in ranges}
                row['evaluation_origin_id']=origin['origin_id']
            if self.manager.runtime_recipe is not None:row['runtime_recipe']=deepcopy(self.manager.runtime_recipe)
            self.items[ident]=row;self.latest=ident;self.save(row);self.launch()
            return self.status()

    def launch(self):
        if self.thread and self.thread.is_alive():self.restart=True;return
        self.thread=Thread(target=self._run,daemon=True);self.thread.start()

    def record(self,batch,member,job):
        member.update(phase=job['phase'],worker_id=job.get('worker_id'),worker_label=job.get('worker_label'),data_saved=job.get('data_saved',False),error=job.get('error'))
        if job['phase']=='complete':
            member['comparison']=deepcopy(job['comparison'])
            values={r['kpi']:r['before']['value']for r in job['comparison']['rows']}
            if 'baseline_reference'not in batch:batch['baseline_reference']=values
            valid=all(type(v)in(int,float)and math.isfinite(v)and type(batch['baseline_reference'].get(k))in(int,float)for k,v in values.items())
            member['baseline_matches_reference']=all(math.isclose(v,batch['baseline_reference'][k],rel_tol=1e-7,abs_tol=1e-8)for k,v in values.items())if valid else None

    def _run(self):
        try:
            while True:
                with self.manager.lock:
                    batch=self.items[self.latest]
                    active=[]
                    for member in batch['members']:
                        if not member.get('job_id')or member['phase']not in('pending','running'):continue
                        job=self.manager.jobs.get(member['job_id'])
                        if job is None:raise RuntimeError('Retained child job unavailable; refusing to duplicate it')
                        if job['phase']in JOB_TERMINAL:
                            self.record(batch,member,job)
                            if job['phase']=='cancelled'and batch['pause_requested']:member.update(phase='pending',job_id=None)
                        else:
                            member.update(phase='running',worker_id=job['worker_id'],worker_label=job.get('worker_label'))
                            active.append(job)
                    if batch['pause_requested']:
                        batch['phase']='pausing'if active else'paused'
                        self.save(batch)
                        if not active:return
                    else:
                        for worker_id in self.manager.idle_workers(batch.get('worker_ids')):
                            member=next((m for m in batch['members']if m['phase']=='pending'),None)
                            if member is None:break
                            index=batch['members'].index(member)
                            launched=self.manager.start(records=batch['records'],group=dict(app_ids=member['app_ids'],selected_names=member['selected_names'],id=member['group_id']),
                                observe_slots=batch['observe_slots'],context=batch['context'],batch_id=batch['id'],batch_index=index,worker_id=worker_id)
                            member.update(job_id=launched['id'],phase='running',worker_id=worker_id,worker_label=launched.get('worker_label'),error=None,data_saved=False)
                            member['attempts'].append(launched['id']);active.append(self.manager.jobs[launched['id']])
                        if not active and not any(m['phase']=='pending'for m in batch['members']):
                            batch.update(phase='complete'if all(m['phase']=='complete'for m in batch['members'])else'complete_with_failures',finished_unix_s=time.time())
                            batch.pop('running_job_id',None);batch['running_job_ids']=[];self.save(batch);return
                        batch['phase']='connection_lost'if active and all(j['phase']=='connection_lost'for j in active)else'running'
                    batch['running_job_ids']=[j['id']for j in active]
                    if active:batch['running_job_id']=active[0]['id']
                    else:batch.pop('running_job_id',None)
                    self.save(batch)
                time.sleep(POLL_SECONDS)
        except Exception as error:
            with self.manager.lock:
                batch=self.items[self.latest];batch.update(phase='paused',error=str(error));self.save(batch)
        finally:
            with self.manager.lock:
                if self.thread is current_thread():self.thread=None
                restart=self.restart;self.restart=False
                if restart and self.active():self.launch()

    def pause(self,ident):
        with self.manager.lock:
            batch=self.items.get(ident)
            if not batch or ident!=self.latest or batch['phase']in BATCH_TERMINAL:raise IntentError('此批次没有正在运行的队列。')
            batch.update(pause_requested=True,phase='pausing');self.save(batch)
            errors=[]
            for job in self.manager.busy_jobs():
                if job.get('batch_id')!=ident:continue
                try:self.manager.cancel(job['id'])
                except IntentError as error:errors.append(str(error))
            if errors:batch['error']='；'.join(errors);self.save(batch)
            return self.status()

    def resume(self,ident):
        with self.manager.lock:
            batch=self.items.get(ident)
            if not batch or ident!=self.latest or batch['phase']not in ('paused','complete_with_failures'):raise IntentError('此批次无需恢复。')
            if self.manager.busy_jobs():raise IntentError('请先重新连接或停止尚未结束的仿真。')
            for member in batch['members']:
                if member['phase']in('failed','cancelled'):member.update(phase='pending',job_id=None)
            batch.update(phase='queued',pause_requested=False,worker_ids=list(self.manager.workers),concurrency=len(self.manager.workers),execution='parallel_independent_runs'if len(self.manager.workers)>1 else'sequential_independent_runs');batch.pop('error',None);self.save(batch);self.launch()
            return self.status()

    def export(self,ident,format):
        with self.manager.lock:
            if ident not in self.items:raise IntentError('批次不存在。')
            batch=deepcopy(self.items[ident])
        batch=enrich_batch(batch)
        batch['controlled_comparison']=self.audit_baseline(batch)
        batch['evaluation_origin_audit']=self.audit_origin(batch)
        if format=='json':return json.dumps(batch,ensure_ascii=False,allow_nan=False,indent=2).encode(),'application/json'
        if format!='csv':raise IntentError('导出格式须为 json 或 csv。')
        stream=io.StringIO();writer=csv.writer(stream)
        writer.writerow(['xApps','Execution status','KPI','Unit','Target lower','Target upper','Lower inclusive','Upper inclusive','Before','After','Change','After target status','Before samples','After samples','Job ID','Worker','Quantification mode','Relative change (%)','Favorable change (%)','Target progress (%)','After interval margin','Result score before','Result score after','Result score change','KPI baseline score','KPI improvement score','Common KPI scale','Scale basis','KPI score unit','No regression','Regression KPIs','KPI after normalized violation','Mean after normalized violation','Max after normalized violation','Evaluation method','Origin alignment'])
        for member in batch['members']:
            rows={r['kpi']:r for r in member.get('comparison',{}).get('rows',[])}
            for target in batch['ranges']:
                r=rows.get(target['kpi'],{});before=r.get('before')or{};after=r.get('after')or{}
                q=r.get('quantification',{})
                evaluation=member.get('comparison',{}).get('result_evaluation',{})
                metric=next((m for m in evaluation.get('metrics',[])if m['kpi']==target['kpi']),{})
                fractional=target['kpi']=='urllc_failure_fraction'
                writer.writerow([' + '.join(member['selected_names']),member['phase'],target['label'],target['unit'],target['lower'],target['upper'],target['lower_inclusive'],target['upper_inclusive'],before.get('value'),after.get('value'),r.get('delta'),after.get('status'),before.get('measurement',{}).get('sample_count'),after.get('measurement',{}).get('sample_count'),member['job_id'],member.get('worker_id')or self.manager.jobs.get(member.get('job_id'),{}).get('worker_id'),metric.get('mode',q.get('mode')),None if fractional else q.get('relative_change_percent'),None if fractional else q.get('favorable_change_percent'),metric.get('target_progress_percent'),q.get('after_margin'),evaluation.get('before',{}).get('score'),evaluation.get('after',{}).get('score'),evaluation.get('change'),metric.get('before',{}).get('score'),metric.get('improvement_score'),metric.get('scale'),metric.get('scale_basis'),metric.get('score_unit'),evaluation.get('no_regression'),' + '.join(evaluation.get('regression_kpis',[])),metric.get('after',{}).get('normalized_violation'),evaluation.get('violation',{}).get('after',{}).get('mean'),evaluation.get('violation',{}).get('after',{}).get('max'),evaluation.get('method'),batch['evaluation_origin_audit']['status']])
        return ('\ufeff'+stream.getvalue()).encode(),'text/csv'
