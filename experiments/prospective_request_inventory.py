"""Read-only request metadata handshake for a separately frozen predictor.

No simulation output, KPI or evaluation label is opened. Running/completed/
failed IDs and preserved partial directories make absence checks conservative
when a job was retried. Neither worker is paused or changed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _read(path):return json.loads(Path(path).read_text())
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _utc(epoch):return datetime.fromtimestamp(epoch,timezone.utc).isoformat()


def known_jobs(artifact):
    root=Path(artifact).resolve()
    if root.name!='v2':raise ValueError('Use the frozen V2 artifact root')
    jobs=[];workers={}
    for path in sorted((root/'deployment').glob('*/worker-config.json')):
        worker=_read(path);workers[worker['id']]=worker
        plan=_read(root/'workers'/worker['id']/'plan.json')
        if plan['project_root']!=worker['project_root']:raise ValueError('Worker source root differs')
        for case in plan['cases']:
            jobs.append(dict(job_id=case['id'],origin_key=case['origin_key'],partition='main',worker_id=worker['id'],arm=case['arm'],
                remote_output_root=worker['project_root']+'/outputs/quantification_study',
                remote_request_path=worker['project_root']+'/outputs/quantification_study/jobs/'+case['id']+'/data/request.json'))
    manifest=_read(root/'supplement/cases-manifest.json')
    for case in manifest['cases']:
        worker=workers[case['worker_id']];out=worker['project_root']+'/outputs/prospective_supplement'
        expected=out+'/jobs/'+case['id']+'/data/run'
        if case['remote_run_path']!=expected:raise ValueError('Supplement source path differs')
        jobs.append(dict(job_id=case['id'],origin_key=case['origin_key'],partition='supplement',worker_id=worker['id'],arm=case['arm'],
            remote_output_root=out,remote_request_path=out+'/jobs/'+case['id']+'/data/request.json'))
    if len(jobs)!=132 or len({j['job_id']for j in jobs})!=132 or {w:sum(j['worker_id']==w for j in jobs)for w in workers}!={'agx1':66,'agx2':66}:
        raise ValueError('Exactly132 known unique jobs on two devices are required')
    return sorted(jobs,key=lambda j:(j['worker_id'],j['partition'],j['job_id'])),workers


REMOTE_CODE=r'''
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
c=json.loads(sys.stdin.buffer.read());started=time.time();root=Path(c['project_root']);sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
from experiments.run_rcp_perturbation import source_hash
p=json.loads((root/'outputs/quantification_study/plan.json').read_text());observed=source_hash(root)
if observed!=c['source_sha256']or observed!=p['source_sha256']or p['source_archive_sha256']!=c['source_archive_sha256']:raise ValueError('Frozen physical source differs')
status={}
def ids(values):
 return [v if isinstance(v,str)else v.get('id')for v in values]
for partition,dirname in [('main','quantification_study'),('supplement','prospective_supplement')]:
 out=root/'outputs'/dirname;path=out/'study_status.json';s=json.loads(path.read_text())if path.exists()else None
 status[partition]=dict(known=s is not None,done=ids(s.get('done',[]))if s else[],running=ids(s.get('running',[]))if s else[],failed=ids(s.get('failed',[]))if s else[],worker_pid=s.get('worker_pid')if s else None)
rows=[]
for job in c['jobs']:
 ident=job['job_id'];out=Path(job['remote_output_root']);path=Path(job['remote_request_path'])
 if len(ident)!=32 or any(ch not in'0123456789abcdef'for ch in ident)or out not in(root/'outputs/quantification_study',root/'outputs/prospective_supplement')or path!=out/'jobs'/ident/'data/request.json':raise ValueError('Unknown request path')
 try:st=path.stat();exists=True
 except FileNotFoundError:st=None;exists=False
 partials=sorted(str(p)for p in(out/'partial').glob(ident+'_*')if p.is_dir())
 state=status[job['partition']];memberships=[key for key in('done','running','failed')if ident in state[key]]
 rows.append(dict(job,exists=exists,request_mtime_ns=st.st_mtime_ns if st else None,request_size_bytes=st.st_size if st else None,
  historical_partial_directories=partials,worker_status_known=state['known'],worker_status_memberships=memberships,
  ever_started_evidence=bool(exists or partials or memberships),query_state='known'if state['known']else'worker_status_unavailable'))
finished=time.time()
print(json.dumps(dict(worker_id=c['worker_id'],remote_started_unix_s=started,remote_finished_unix_s=finished,
 remote_started_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),remote_finished_utc=datetime.fromtimestamp(finished,timezone.utc).isoformat(),
 source_sha256=observed,source_archive_sha256=p['source_archive_sha256'],source_plan_sha256=p['physical_plan_sha256'],
 worker_statuses=status,jobs=rows,read_scope='known request file stat, preserved partial directory names, worker ID journals, source hashes; no KPI/output records')))
'''


def _query(worker,jobs):
    started=time.time();payload=dict(worker_id=worker['id'],project_root=worker['project_root'],source_sha256=worker['source_sha256'],
        source_archive_sha256=worker['source_archive_sha256'],jobs=jobs)
    result=subprocess.run(['ssh','-S',worker['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=10',worker['host'],
        shlex.join(['python3','-c',REMOTE_CODE])],input=json.dumps(payload).encode(),capture_output=True,timeout=90)
    if result.returncode:raise ValueError('Read-only request scan failed for '+worker['id']+': '+result.stderr.decode(errors='replace')[-1200:])
    response=json.loads(result.stdout);response.update(local_dispatched_unix_s=started,local_received_unix_s=time.time())
    response['response_sha256']=_digest(response)
    return response


def scan_requests(artifact,output_path,*,code_bundle):
    """Write one immutable complete scan; caller performs a second after scoring."""
    artifact=Path(artifact).resolve();output=Path(output_path).resolve();bundle=Path(code_bundle['path']).resolve()
    if output.exists():raise ValueError('Do not overwrite an immutable request scan')
    if not output.is_relative_to(artifact):raise ValueError('Keep the scan in the study artifact')
    if not bundle.is_file()or _sha(bundle)!=code_bundle['sha256']:raise ValueError('Frozen code bundle pin differs')
    started=time.time();mtime=bundle.stat().st_mtime_ns
    if mtime/1e9>started:raise ValueError('Code bundle must be persisted before the scan')
    jobs,workers=known_jobs(artifact)
    with ThreadPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(_query,w,[j for j in jobs if j['worker_id']==w['id']])for w in workers.values()]
        replies=[f.result()for f in futures]
    returned=[j for r in replies for j in r['jobs']]
    if len(returned)!=132 or {j['job_id']for j in returned}!={j['job_id']for j in jobs}:raise ValueError('Incomplete remote scan')
    if _sha(bundle)!=code_bundle['sha256']:raise ValueError('Code bundle changed during scan')
    finished=time.time();value=dict(schema='oran.prospective-request-scan.v1',local_started_unix_s=started,local_finished_unix_s=finished,
        local_started_utc=_utc(started),local_finished_utc=_utc(finished),frozen_code_bundle=dict(path=str(bundle),sha256=code_bundle['sha256'],mtime_ns=mtime),
        code_bundle_persisted_before_scan=True,jobs_grid_sha256=_digest(jobs),job_count=132,workers=replies,
        expected_source_pins={w['id']:dict(source_sha256=w['source_sha256'],source_archive_sha256=w['source_archive_sha256'],physical_plan_sha256=w['physical_plan_sha256'])for w in workers.values()},
        inventory_module_sha256=_sha(__file__),remote_query_code_sha256=hashlib.sha256(REMOTE_CODE.encode()).hexdigest(),after_kpi_values_read=False,remote_writes_performed=False)
    value['scan_sha256']=_digest(value);output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents a concurrent caller from replacing a scan.
    with output.open('x')as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    output.chmod(0o444)
    return dict(path=str(output),sha256=_sha(output),scan_sha256=value['scan_sha256'])


def _scan(ref):
    path=Path(ref['path']if isinstance(ref,dict)else ref)
    if isinstance(ref,dict)and _sha(path)!=ref['sha256']:raise ValueError('Request scan file pin differs')
    value=_read(path);expected=value.pop('scan_sha256',None)
    if expected!=_digest(value)or value.get('schema')!='oran.prospective-request-scan.v1'or value.get('after_kpi_values_read')is not False or value.get('remote_writes_performed')is not False:
        raise ValueError('Invalid read-only scan evidence')
    if value.get('code_bundle_persisted_before_scan')is not True:raise ValueError('Predictor must be frozen before the scan')
    value['scan_sha256']=expected
    for worker in value['workers']:
        response=dict(worker);pin=response.pop('response_sha256',None)
        if pin!=_digest(response):raise ValueError('Remote query response digest differs')
    rows=[j for w in value['workers']for j in w['jobs']]
    if len(rows)!=132 or len({j['job_id']for j in rows})!=132:raise ValueError('Incomplete scan job grid')
    return value,{j['job_id']:j for j in rows}


def paired_eligibility(before,after,*,code_bundle,scoring_started_unix_s,scoring_finished_unix_s):
    """Conservatively seal the common future subset, never infer missing starts."""
    b,bjobs=_scan(before);a,ajobs=_scan(after)
    expected=dict(path=str(Path(code_bundle['path']).resolve()),sha256=code_bundle['sha256'])
    if not Path(expected['path']).is_file()or _sha(expected['path'])!=expected['sha256']:raise ValueError('Frozen predictor bundle changed')
    for scan in(b,a):
        if {k:scan['frozen_code_bundle'][k]for k in('path','sha256')}!=expected:raise ValueError('Different frozen predictor bundle')
    if (b['jobs_grid_sha256']!=a['jobs_grid_sha256']or set(bjobs)!=set(ajobs)
        or b['expected_source_pins']!=a['expected_source_pins']or b['inventory_module_sha256']!=a['inventory_module_sha256']
        or not b['local_finished_unix_s']<=scoring_started_unix_s<=scoring_finished_unix_s<=a['local_started_unix_s']):
        raise ValueError('Two scans must bracket scoring on the identical planned grid')
    eligible=[];diagnostic=[]
    for ident in sorted(bjobs):
        old,new=bjobs[ident],ajobs[ident];reasons=[]
        if not old['worker_status_known']or not new['worker_status_known']:reasons.append('worker_history_unknown')
        if old['exists']or old['ever_started_evidence']or old['historical_partial_directories']or old['worker_status_memberships']:reasons.append('started_before_first_scan')
        if new['exists']or new['ever_started_evidence']or new['historical_partial_directories']or new['worker_status_memberships']:reasons.append('started_by_second_scan')
        if reasons:diagnostic.append(dict(job_id=ident,partition=old['partition'],origin_key=old['origin_key'],reasons=reasons))
        else:eligible.append(ident)
    return dict(schema='oran.prospective-request-eligibility.v1',before_scan_sha256=b['scan_sha256'],after_scan_sha256=a['scan_sha256'],
        jobs_grid_sha256=b['jobs_grid_sha256'],code_bundle=expected,scoring_started_unix_s=scoring_started_unix_s,
        scoring_finished_unix_s=scoring_finished_unix_s,eligible_job_ids=eligible,diagnostic_only=diagnostic,
        eligible_physical_jobs=len(eligible),diagnostic_physical_jobs=len(diagnostic),after_kpi_values_read=False)
