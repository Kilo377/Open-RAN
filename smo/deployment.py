"""Local UI job manager for an operator-configured, isolated SSH worker.

No browser-supplied command, host, path, model weights or RCP values are run.
A click starts one named combination; the job archive survives service reloads.
"""
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
from threading import RLock,Thread
import time
from uuid import uuid4

from smo.intent_language import IntentError

TERMINAL={'complete','failed','cancelled'}
REMOTE_TERMINAL={'finished','failed','cancelled'}


def atomic(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)


class DeploymentManager:
    def __init__(self,config=None):
        self.config=json.loads(Path(config).read_text())if config else None
        archive=Path(config).parent/'source.tgz'if config else None
        self.source_archive_path=str(archive.resolve())if archive and archive.exists()else None
        self.source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()if archive and archive.exists()else None
        self.runtime_recipe=None;self.runtime_origin=None;self.runtime_origin_run=None
        self.lock=RLock();self.jobs={};self.active=None;self.source=None;self.windows=None;self.run=None;self.batch=None
        self.workers={};self.watchers=set();self.rewatch=set()
        if self.config:
            c=self.config
            if 'workers'in c:
                if set(c)-{'runtime_recipe'}!={'workers','local_output','seed','label'}or not isinstance(c['workers'],list)or not c['workers']:raise ValueError('Invalid worker pool')
                entries=c['workers']
                if any(set(w)!={'id','host','socket','project_root','python','label'}for w in entries):raise ValueError('Invalid pool worker')
            else:
                if set(c)-{'runtime_recipe'}!={'host','socket','project_root','python','local_output','seed','label'}:raise ValueError('Invalid deployment worker configuration')
                entries=[dict(id='worker-0',**{k:c[k]for k in('host','socket','project_root','python','label')})]
            for w in entries:
                if not re.fullmatch(r'[\w.-]+',w['id'])or w['id']in self.workers:raise ValueError('Unique worker IDs required')
                if not re.fullmatch(r'[\w.-]+@[\w.-]+',w['host']):raise ValueError('Invalid SSH worker')
                for name in ('socket','project_root','python'):
                    if not Path(w[name]).is_absolute():raise ValueError('Deployment paths must be absolute')
                self.workers[w['id']]=deepcopy(w)
            if len({w['host']for w in entries})!=len(entries):raise ValueError('One slot per SSH host is required')
            if 'runtime_recipe'in c:
                from smo.runtime_recipe import load_recipe
                self.runtime_recipe,self.runtime_origin=load_recipe(c['runtime_recipe'],seed=c['seed'])
                self.runtime_origin_run=Path(self.runtime_recipe['validated_origin']['run'])
                if not self.source_archive_sha256:raise ValueError('Runtime recipe requires a pinned worker source.tgz')
            if not Path(c['local_output']).is_absolute():raise ValueError('Output path must be absolute')
            self.output=Path(c['local_output']);self.output.mkdir(parents=True,exist_ok=True)
            for p in self.output.glob('*/job.json'):
                row=json.loads(p.read_text());row.setdefault('worker_id',next(iter(self.workers)))
                if row['worker_id']=='worker-0'and 'worker-0'not in self.workers:row['worker_id']=next(iter(self.workers))
                if row['worker_id']not in self.workers:raise ValueError('A retained job requires its original worker configuration')
                row['worker_label']=self.workers[row['worker_id']]['label'];self.jobs[row['id']]=row
            if self.jobs:
                self.active=max(self.jobs,key=lambda i:self.jobs[i]['created_unix_s'])
                row=self.jobs[self.active]
                if row['phase']=='complete'and not row.get('batch_id'):self.attach(row)
                for job in list(self.jobs.values()):
                    if job['phase']not in TERMINAL:self.watch(job['id'])

        from smo.deployment_batch import BatchEvaluator
        self.batch=BatchEvaluator(self)
        if self.batch.active():self.batch.launch()

    def ssh(self,code,*,data=None,timeout=20,worker_id=None):
        c=self.workers[worker_id or next(iter(self.workers))]
        command=['ssh','-S',c['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=8',c['host'],
            shlex.join(['python3','-c',code])]
        return subprocess.run(command,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=True).stdout

    def status(self):
        with self.lock:
            return dict(enabled=bool(self.config),target=self.config['label']if self.config else None,
                latest=deepcopy(self.jobs.get(self.active)),running_jobs=deepcopy(self.busy_jobs()),workers=[dict(id=w['id'],label=w['label'],busy=any(j['worker_id']==w['id']for j in self.busy_jobs()))for w in self.workers.values()],concurrency=len(self.workers),latest_batch=self.batch.status()if self.batch else None,automatic_selection=False)

    def busy_jobs(self):
        return [j for j in self.jobs.values()if j['phase']not in TERMINAL]

    def idle_workers(self,allowed=None):
        busy={j['worker_id']for j in self.busy_jobs()}
        return [w for w in (allowed or list(self.workers))if w in self.workers and w not in busy]

    def persist(self,row):
        with self.lock:atomic(self.output/row['id']/'job.json',row)

    def execution_context(self,context,observe_slots,*,recipe=None,default_origin=None):
        """Only the operator configuration can choose a different BEFORE window."""
        context=deepcopy(context);origin=context.get('evaluation_origin')
        if recipe is not None:
            from smo.runtime_recipe import validate_origin
            if origin is None:origin=deepcopy(default_origin)
            if origin is None:raise IntentError('Validated runtime origin is unavailable.')
            validate_origin(recipe,origin)
            if default_origin is not None and origin['origin_id']!=default_origin['origin_id']:
                raise IntentError('评估起点与已验证环境不同。')
            context['evaluation_origin']=deepcopy(origin)
            context['runtime_environment']=dict(id=recipe['id'],baseline_slot=recipe['baseline_slot'],
                origin_observe_slots=recipe['origin_observe_slots'])
        elif origin is not None:
            from smo.evaluation_origin_guard import validate_proof
            validate_proof(origin)
            if origin['observe_slots']!=observe_slots or origin['seed']!=self.config['seed']:
                raise IntentError('评估起点与当前窗口或随机种子不同，请重新对齐评估起点。')
        return context

    def start(self,*,records,group,observe_slots,context,batch_id=None,batch_index=None,worker_id=None):
        if not self.config:raise IntentError('尚未配置仿真设备。请给 SMO 配置 deployment-config。')
        if type(observe_slots)is not int or not 1<=observe_slots<=20000:raise IntentError('窗口长度须为 1–20,000 个整数 slots。')
        with self.lock:
            if self.batch and self.batch.active()and batch_id!=self.batch.latest:raise IntentError('全部组合评估正在运行，不能插入单组任务。')
            available=self.idle_workers()
            if worker_id is not None and worker_id not in available or not available:raise IntentError('仿真设备正在运行，或指定设备尚未空闲。')
            worker_id=worker_id or available[0]
            batch=self.batch.items[batch_id]if batch_id else None
            recipe=batch.get('runtime_recipe')if batch is not None else self.runtime_recipe
            default_origin=batch.get('context',{}).get('evaluation_origin')if batch is not None else self.runtime_origin
            context=self.execution_context(context,observe_slots,recipe=recipe,default_origin=default_origin)
            origin=context.get('evaluation_origin')
            ident=uuid4().hex;folder=self.output/ident;folder.mkdir()
            row=dict(id=ident,phase='queued',created_unix_s=time.time(),app_ids=deepcopy(group['app_ids']),
                selected_names=deepcopy(group['selected_names']),active_intent_ids=[r['id']for r in records],
                records=deepcopy(records),observe_slots=observe_slots,seed=self.config['seed'],context=context,
                batch_id=batch_id,batch_index=batch_index,group_id=group.get('id'),worker_id=worker_id,worker_label=self.workers[worker_id]['label'],
                expected_source_archive_sha256=self.batch.items[batch_id].get('worker_source_archive_sha256')if batch_id else self.source_archive_sha256,output=str(folder),baseline_origin='fresh_run_all_xapps_off',automatic_selection=False)
            if origin is not None:
                row['expected_origin']=deepcopy(origin)
            if recipe is not None:
                from smo.runtime_recipe import worker_recipe
                row['runtime_recipe']=worker_recipe(recipe,self.workers[worker_id]['project_root'],origin)
                row['runtime_environment']=deepcopy(context['runtime_environment'])
            self.jobs[ident]=row;self.active=ident;self.persist(row)
            self.watch(ident,launch=True)
            return deepcopy(row)

    def watch(self,ident,launch=False):
        with self.lock:
            if ident in self.watchers:
                if not launch:self.rewatch.add(ident)
                return
            self.watchers.add(ident)
        def monitored():
            try:self._watch(ident,launch)
            finally:
                with self.lock:
                    self.watchers.discard(ident)
                    if ident in self.rewatch:
                        self.rewatch.discard(ident)
                        if self.jobs[ident]['phase']not in TERMINAL:self.watch(ident)
        Thread(target=monitored,daemon=True).start()

    def _watch(self,ident,launch):
        row=self.jobs[ident];worker_id=row['worker_id'];c=self.workers[worker_id];remote=Path(c['project_root'])/'outputs/smo_deployments'/ident
        try:
            if launch:
                request={k:row[k]for k in('id','app_ids','selected_names','records','observe_slots','seed')}
                request.update({k:row.get(k)for k in('batch_id','batch_index','group_id','worker_id','expected_source_archive_sha256')})
                if row.get('expected_origin')is not None:request['expected_origin']=deepcopy(row['expected_origin'])
                if row.get('runtime_recipe')is not None:request['runtime_recipe']=deepcopy(row['runtime_recipe'])
                code='''import json,os,subprocess,sys
from pathlib import Path
c=json.load(sys.stdin);root=Path(c['root']);job=root/'outputs/smo_deployments'/c['request']['id']
pin=c['request'].get('expected_source_archive_sha256')
if pin is not None:
    marker=root/'.smo-owned-worker.json'
    if not marker.exists() or json.loads(marker.read_text()).get('sha256')!=pin:raise ValueError('Worker source differs from the frozen batch snapshot')
if job.exists():raise ValueError('Job already exists')
job.mkdir(parents=True);p=job/'request.json';p.write_text(json.dumps(c['request']))
log=(job/'worker.log').open('ab')
proc=subprocess.Popen([c['python'],'-m','smo.deployment_worker','--request',str(p)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(proc.pid)
'''
                pid=int(self.ssh(code,data=json.dumps(dict(root=c['project_root'],python=c['python'],request=request)).encode(),worker_id=worker_id))
                row.update(worker_pid=pid,phase='initializing');self.persist(row)
                if row.get('stop_requested'):self.ssh('from pathlib import Path; Path('+repr(str(remote/'STOP'))+').touch()',worker_id=worker_id)
            errors=0;unchanged_since=time.monotonic();last_stamp=None
            while True:
                try:
                    code='from pathlib import Path; p=Path('+repr(str(remote/'status.json'))+'); print(p.read_text() if p.exists() else "{}")'
                    report=json.loads(self.ssh(code,worker_id=worker_id));errors=0
                    if report:
                        row.update({k:v for k,v in report.items()if k not in ('id','selected_app_ids','selected_names')})
                        if report.get('updated_unix_s')!=last_stamp:last_stamp=report.get('updated_unix_s');unchanged_since=time.monotonic()
                        if report['phase']in REMOTE_TERMINAL:
                            terminal=report['phase'];row['phase']='saving';self.persist(row)
                            self.download(ident,remote)
                            row['phase']='complete'if terminal=='finished'else terminal
                            if row['phase']=='complete'and not row.get('batch_id'):self.attach(row)
                            self.persist(row);return
                        self.persist(row)
                    if time.monotonic()-unchanged_since>180:
                        raise RuntimeError('Worker stopped reporting; reconnect to retain or stop this job')
                except (subprocess.SubprocessError,OSError,ValueError)as error:
                    errors+=1
                    if errors>=3:raise RuntimeError('SSH connection lost; the remote job and its data are retained. Reconnect before starting another run.')from error
                time.sleep(1.5)
        except Exception as error:
            row.update(phase='connection_lost'if isinstance(error,(RuntimeError,subprocess.SubprocessError))else'failed',error=str(error))
            self.persist(row)

    def download(self,ident,remote):
        # Archive is generated from this one owned UUID directory. Validate all
        # archive entries before writing; keep every raw recording, even on stop.
        code='import sys,tarfile; from pathlib import Path; p=Path('+repr(str(remote))+'); t=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz"); t.add(p,arcname="data"); t.close()'
        raw=self.ssh(code,timeout=120,worker_id=self.jobs[ident].get('worker_id',next(iter(self.workers))))
        folder=self.output/ident;archive=folder/'data.tar.gz';archive.write_bytes(raw)
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
            for member in t.getmembers():
                target=(folder/member.name).resolve()
                if not target.is_relative_to(folder.resolve())or member.issym()or member.islnk()or not(member.isfile()or member.isdir()):
                    raise ValueError('Invalid deployment archive entry')
            t.extractall(folder)
        row=self.jobs[ident];row.update(data_saved=True,archive_bytes=len(raw),data_path=str(folder/'data'))

    def attach(self,row):
        from smo.runtime import replay
        from smo.window_metrics import WindowIndex
        run=self.output/row['id']/'data/run'
        index=WindowIndex(json.loads((run/'config_snapshot.json').read_text()))
        source=replay(run,160,population=True,window_index=index)
        source.update(run_id=row['id'],deployment_job_id=row['id'])
        with self.lock:self.run=run;self.source=source;self.windows=index

    def cancel(self,ident):
        with self.lock:
            row=self.jobs.get(ident)
            if not row or row['phase']in TERMINAL:raise IntentError('此任务没有可停止的仿真。')
            worker_id=row['worker_id'];remote=Path(self.workers[worker_id]['project_root'])/'outputs/smo_deployments'/ident/'STOP'
            try:self.ssh('from pathlib import Path; p=Path('+repr(str(remote))+'); p.touch() if p.parent.exists() else None',worker_id=worker_id)
            except (OSError,subprocess.SubprocessError)as error:raise IntentError('SSH 未连接，停止请求尚未送达；请先重新连接。')from error
            row['stop_requested']=True;self.persist(row)
            return deepcopy(row)

    def reconnect(self):
        with self.lock:
            for row in list(self.jobs.values()):
                if row['phase']=='connection_lost':
                    row['phase']='reconnecting';row.pop('error',None);self.persist(row);self.watch(row['id'])
            return self.status()

    def view(self,ident):
        with self.lock:
            row=self.jobs.get(ident)
            if not row or row['phase']!='complete' or not row.get('data_saved'):raise IntentError('此组合尚无完整的实测结果。')
            self.attach(row)
            return dict(job_id=ident,group_id=row.get('group_id'))

    def result_file(self,ident):
        row=self.jobs.get(ident)
        if not row or not row.get('data_saved'):raise IntentError('结果仍在生成。')
        path=self.output/ident/'data/run/network-replay.html'
        if not path.is_file():raise IntentError('此次运行未生成完整的回放 HTML；原始数据已保留。')
        return path
