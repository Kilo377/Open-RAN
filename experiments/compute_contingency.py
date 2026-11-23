"""Opt-in compute contingency; never changes the frozen simulator or predictors.

Preparation/import has no remote effects. A future parity or migration needs
its own explicit root certificate. The physical source remains source-v2;
this wrapper's true SHA/host is recorded separately from the planned wrapper.
"""
from copy import deepcopy
from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

SOURCE_ROOT='/dev/shm/oran_quantification_20261122/source-v2'
SOURCE_PIN='cbe114446814007b275856e4bae92a2fe609abf4f4bcbbf8c32394a722a36af0'
ARCHIVE_PIN='4f915fd649e287b68465c4aed047a7b626959b471d532abd8a659e376e3641e1'
PLANNED_WRAPPER_PIN='da688d1e3690d00a7d8c7110f0daf8ff3ad2bbf82c21df24cd91d2c3767e2558'
REFERENCE_CASE_ID='e75458cf9ecb10e76ee7d30471678980'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_protocol(protocol,pin):
    value=deepcopy(protocol);recorded=value.pop('protocol_sha256',None)
    if recorded!=pin or digest(value)!=pin or protocol.get('schema')!='oran.compute-contingency-protocol.v1':raise ValueError('Compute contingency protocol differs')
    if (protocol['physical_source_sha256']!=SOURCE_PIN or protocol['physical_archive_sha256']!=ARCHIVE_PIN
        or protocol['original_protocol_runner_sha256']!=PLANNED_WRAPPER_PIN
        or protocol['reference_case']['id']!=REFERENCE_CASE_ID or protocol['reference_case']['arm']!='ABC'
        or protocol['logical_owner']!='agx1'or protocol['actual_execution_host']!='agx2'
        or len(protocol['migration_cases'])!=6 or len({c['id']for c in protocol['migration_cases']})!=6):
        raise ValueError('Fixed source/reference/owner/case matrix differs')
    return protocol


def idle_execution_host(original,supplement,*,original_case_count=60,supplement_case_count=6):
    """Only the already idle AGX2 may be considered; no source task is paused."""
    return all(status.get('phase')=='complete'and status.get('completed')is True and not status.get('running')
        and not status.get('failed')and len(status.get('done',[]))==count
        for status,count in((original,original_case_count),(supplement,supplement_case_count)))


def waiting_owner_eligible(status,original,case_ids,*,requests=(),partial_case_ids=(),checkpoint_ids=(),physical_children=()):
    """Rechecked while the owned waiter is frozen; any race cancels handoff."""
    expected=set(case_ids)
    return (len(expected)==6 and status.get('phase')=='waiting_for_original_matrix'and not status.get('running')
        and not status.get('done')and not status.get('failed')and status.get('completed')is False
        and original.get('completed')is not True and original.get('phase')!='complete'
        and not(expected&set(requests))and not(expected&set(partial_case_ids))and not(expected&set(checkpoint_ids))
        and not physical_children)


def runtime_trigger(*,now,stop_new_at,agx1_remaining_main,agx1_active_walls,agx2_supplement_walls):
    """Resource-only rule, frozen before parity; no KPI selects a case/host."""
    walls=list(agx1_active_walls);supp=list(agx2_supplement_walls)
    if (len(walls)<6 or len(supp)!=6 or type(agx1_remaining_main)is not int or agx1_remaining_main<1
        or any(not isinstance(x,(int,float))or isinstance(x,bool)or not math.isfinite(x)or x<=0 for x in[*walls,*supp])):
        return dict(triggered=False,reason='insufficient_runtime_or_waiting_owner_evidence')
    mean=sum(walls[-16:])/len(walls[-16:]);p90=sorted(supp)[math.ceil(.9*len(supp))-1]
    planned_finish=now+(agx1_remaining_main+6)*mean
    main_finish=now+agx1_remaining_main*mean
    offload_finish=now+900+6*p90*1.2+300
    benefit=planned_finish-max(main_finish,offload_finish)
    checks=dict(planned_finish_near_cutoff=planned_finish+900>=stop_new_at,offload_fits_before_cutoff=offload_finish<stop_new_at,
        at_least15min_critical_path_benefit=benefit>=900)
    return dict(triggered=all(checks.values()),checks=checks,planned_finish_unix_s=planned_finish,
        main_finish_unix_s=main_finish,offload_finish_unix_s=offload_finish,estimated_benefit_s=benefit,
        inputs_scope='completed wall times and remaining job counts only; no network KPI values')


def validate_certificate(certificate,protocol,pin,mode,wrapper_pin):
    verify_protocol(protocol,pin)
    if (certificate.get('schema')!='oran.compute-contingency-admission.v1'or certificate.get('root_approved')is not True
        or certificate.get('mode')!=mode or mode not in('parity','migration')or certificate.get('protocol_sha256')!=pin
        or certificate.get('actual_execution_host')!='agx2'or certificate.get('logical_owner')!='agx1'
        or certificate.get('physical_source_sha256')!=SOURCE_PIN or certificate.get('physical_archive_sha256')!=ARCHIVE_PIN
        or certificate.get('execution_wrapper_sha256')!=wrapper_pin or wrapper_pin!=protocol['execution_wrapper_sha256']
        or certificate.get('original_protocol_runner_sha256')!=PLANNED_WRAPPER_PIN):
        raise ValueError('A separate truthful compute admission is required')
    if mode=='migration'and(certificate.get('exact_parity_passed')is not True or not certificate.get('parity_proof_sha256')
        or certificate.get('runtime_trigger_passed')is not True or certificate.get('owner_handoff_verified')is not True):
        raise ValueError('Parity, budget and exclusive-owner evidence are mandatory')
    return certificate


def execution_metadata(protocol,certificate,certificate_pin,*,mode,actual_hostname=None):
    return dict(logical_owner_worker_id='agx1',actual_execution_host='agx2',actual_system_hostname=actual_hostname or socket.gethostname(),
        actual_ssh_endpoint=certificate['actual_ssh_endpoint'],physical_source_sha256=SOURCE_PIN,physical_archive_sha256=ARCHIVE_PIN,
        original_protocol_runner_sha256=PLANNED_WRAPPER_PIN,execution_wrapper_sha256=protocol['execution_wrapper_sha256'],
        contingency_protocol_sha256=protocol['protocol_sha256'],compute_admission_sha256=certificate_pin,
        parity_proof_sha256=certificate.get('parity_proof_sha256'),kind='parity_diagnostic'if mode=='parity'else'compute_migrated_supplement',
        independent_sample=False,additional_independent_replicate=False,counts_toward_original132=mode=='migration',
        original132_case_identity_changed=False)


def publish_verified_job(actual_job,actual_seal_path,logical_jobs,logical_checkpoints):
    """Copy, never move/rewrite, raw bytes; publish checkpoint only after SHA QA."""
    actual_job=Path(actual_job).resolve();seal=read(actual_seal_path);ident=seal['case']['id'];provenance=seal['execution_provenance']
    if (seal.get('completed')is not True or provenance.get('kind')!='compute_migrated_supplement'
        or provenance.get('logical_owner_worker_id')!='agx1'or provenance.get('actual_execution_host')!='agx2'
        or seal.get('source_sha256')!=SOURCE_PIN or seal.get('runner_sha256')!=provenance['execution_wrapper_sha256']
        or not provenance.get('parity_proof_sha256')):raise ValueError('No verified truthful migrated execution to publish')
    guard=read(actual_job/'data/run/evaluation_origin_verified.json')
    if (guard.get('origin_id')!=seal.get('origin_id')or any(guard.get(key)is not True for key in
        ('verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal'))):
        raise ValueError('Original OFF prefix guard did not pass')
    pins=seal['files_sha256'];destination=Path(logical_jobs)/ident;checkpoint=Path(logical_checkpoints)/(ident+'.json')
    for relative,pin in pins.items():
        source=(actual_job/relative).resolve()
        if not source.is_relative_to(actual_job)or not source.is_file()or sha(source)!=pin:raise ValueError('Actual raw byte pin differs')
    if checkpoint.exists():
        if checkpoint.read_bytes()!=Path(actual_seal_path).read_bytes():raise ValueError('Do not overwrite an existing logical checkpoint')
        if any(not(destination/r).is_file()or sha(destination/r)!=p for r,p in pins.items()):raise ValueError('Published logical raw files changed')
        return dict(published=True,existing=True,job_id=ident)
    if destination.exists():
        for path in destination.rglob('*'):
            if not path.is_file():continue
            relative=str(path.relative_to(destination))
            if relative not in pins or sha(path)!=pins[relative]:raise ValueError('Logical owner already has different/unknown raw data')
    for relative,pin in pins.items():
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary=target.with_name(target.name+'.compute-publish-tmp')
            if temporary.exists():raise ValueError('Retain incomplete publication temporary files for review')
            shutil.copyfile(actual_job/relative,temporary)
            if sha(temporary)!=pin:raise ValueError('Copy verification failed')
            os.replace(temporary,target)
        if sha(target)!=pin:raise ValueError('Logical owner copy pin differs')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    with checkpoint.open('xb')as stream:stream.write(Path(actual_seal_path).read_bytes())
    return dict(published=True,existing=False,job_id=ident,logical_owner='agx1',actual_execution_host='agx2',files=len(pins))


def run_executor(plan_path,certificate_path,certificate_pin):
    """Future explicit execution only. The preparation workflow never calls it."""
    plan_path=Path(plan_path).resolve();p=read(plan_path);protocol=verify_protocol(p['protocol'],p['protocol_sha256']);root=Path(SOURCE_ROOT)
    if sha(certificate_path)!=certificate_pin:raise ValueError('Compute admission document pin differs')
    certificate=validate_certificate(read(certificate_path),protocol,p['protocol_sha256'],p['mode'],sha(__file__))
    out=plan_path.parent;expected=root/'outputs/compute_contingency'/p['mode']
    if out!=expected:raise ValueError('Independent owned compute output required')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
    from experiments.quantification_study import command,completed,isolated_environment,validate_owned
    from experiments.context_identification import external_simulations,resources
    from experiments.run_rcp_perturbation import atomic
    original_out=root/'outputs/quantification_study';supp_out=root/'outputs/prospective_supplement';original=read(original_out/'plan.json');supp=read(supp_out/'plan.json')
    validate_owned(root,original)
    if not idle_execution_host(read(original_out/'study_status.json'),read(supp_out/'study_status.json')):
        raise ValueError('AGX2 original60 plus its own6 supplemental jobs must finish first')
    if not all(completed(original_out,c)for c in original['cases'])or not all(completed(supp_out,c)for c in supp['cases']):raise ValueError('Host completion lacks full raw SHA seals')
    if external_simulations():raise ValueError('Do not compete with another physical simulation')
    metadata=execution_metadata(protocol,certificate,certificate_pin,mode=p['mode']);lock=(out/'executor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if p['mode']=='migration':
        if sha(out/'owner-claim.json')!=certificate['owner_claim_sha256']:raise ValueError('Exclusive owner claim differs')
        claim=read(out/'owner-claim.json')
        if claim.get('old_waiter_stopped')is not True or set(claim.get('claimed_job_ids',[]))!={c['id']for c in protocol['migration_cases']}:raise ValueError('Six original cases were not exclusively handed off')
        if sha(out/'parity-proof.json')!=certificate['parity_proof_sha256']or read(out/'parity-proof.json').get('passed')is not True:
            raise ValueError('Exact parity did not pass')
    cases=[protocol['parity_case']]if p['mode']=='parity'else protocol['migration_cases']
    for name in('jobs','checkpoints'):(out/name).mkdir(exist_ok=True)
    env=isolated_environment(root);status=dict(phase='initializing',completed=False,done=[],failed=[],running=[],execution_provenance=metadata,
        source_sha256=SOURCE_PIN,source_archive_sha256=ARCHIVE_PIN,executor_pid=os.getpid(),mode=p['mode'])
    def publish(**values):status.update(values,updated_unix_s=time.time(),free=resources(root));atomic(out/'executor-status.json',status)
    deadline=datetime.fromisoformat(protocol['deadline_utc']).timestamp();stop_new_at=deadline-3600
    for case in cases:
        if completed(out,case):status['done'].append(case['id']);continue
        validate_owned(root,original)
        if sha(__file__)!=protocol['execution_wrapper_sha256']:raise ValueError('Actual compute wrapper changed')
        if time.time()+900>stop_new_at:publish(phase='budget_complete',completed=True);return
        if external_simulations():raise ValueError('An unrelated simulator started; retain the queue without competition')
        free=resources(root)
        if free['available_memory_bytes']<8*1024**3 or free['free_work_bytes']<5*1024**3 or free['free_system_bytes']<256*1024**2:
            publish(phase='storage_guard',completed=True);return
        job=out/'jobs'/case['id'];data=job/'data';run=data/'run'
        if job.exists():raise ValueError('Retain an existing unsealed compute run; no implicit retry')
        data.mkdir(parents=True)
        origin_ref=protocol['origins'][case['origin_key']];origin=out/'origins'/(case['origin_key']+'.json')
        if sha(origin)!=origin_ref['sha256']:raise ValueError('Original AGX1 OFF proof bytes differ')
        (data/'expected_origin.json').write_bytes(origin.read_bytes())
        if p['mode']=='migration':
            # The local coordinator preclaims these identical request bytes on
            # the logical owner before the actual host can start any job.
            request=out/'claimed-requests'/(case['id']+'.json')
            if sha(request)!=certificate['claimed_request_sha256'][case['id']]:raise ValueError('Logical admission request differs')
            (data/'request.json').write_bytes(request.read_bytes())
        else:atomic(data/'request.json',dict(case=case,reference_case_id=REFERENCE_CASE_ID,execution_provenance=metadata,independent_sample=False,observe_slots=2000))
        atomic(data/'timeline.json',[dict(at_ms=1020.,operation='set',apps=[a.removeprefix('xapp:')for a in case['app_ids']],cell='*')])
        args=command(root,out,case,original['physical']);atomic(data/'command.json',dict(argv=args,execution_provenance=metadata))
        started=time.time();running=dict(case=case,pid=None,slot=0,started_unix_s=started)
        with(data/'simulation.log').open('wb')as log:
            child=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running['pid']=child.pid;publish(phase='running',running=[running])
            while child.poll()is None:
                try:running['slot']=read(run/'live.json').get('slot',0)
                except(OSError,ValueError):pass
                publish()
                if time.time()>deadline-180 or(out/'STOP').exists():
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM)
                time.sleep(2)
        try:
            result=read(run/'run_status.json');guard=read(run/'evaluation_origin_verified.json')
            if child.returncode or not result.get('completed')or not result.get('replay',{}).get('audits_passed')or result['source_sha256']!=SOURCE_PIN:
                raise ValueError('Actual compute run failed; raw files retained')
            if not guard.get('verified_before_activation')or guard['origin_id']!=case['origin_id']:raise ValueError('Original OFF prefix did not reproduce')
            seal=dict(schema='oran.compute-parity-run.v1'if p['mode']=='parity'else'oran.prospective-supplement-run.v1',completed=True,case=case,
                worker_id='agx1',logical_owner_worker_id='agx1',actual_execution_host='agx2',source_sha256=SOURCE_PIN,source_archive_sha256=ARCHIVE_PIN,
                runner_sha256=protocol['execution_wrapper_sha256'],original_protocol_runner_sha256=PLANNED_WRAPPER_PIN,
                execution_provenance=metadata,origin_id=guard['origin_id'],finished_unix_s=time.time(),wall_s=time.time()-started,
                supplement_plan_sha256=protocol['supplement_plan_sha256'],physical_plan_sha256=protocol['parent_physical_plan_sha256'],
                after_kpis_read_for_development=False,files_sha256={str(f.relative_to(job)):sha(f)for f in job.rglob('*')if f.is_file()})
            atomic(out/'checkpoints'/(case['id']+'.json'),seal);status['done'].append(case['id']);publish(running=[])
        except Exception as error:
            status['failed'].append(dict(case_id=case['id'],error=str(error)));publish(phase='failed_retained',running=[],completed=True);return
    publish(phase='complete',running=[],completed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker-plan',type=Path,required=True);parser.add_argument('--certificate',type=Path,required=True);parser.add_argument('--certificate-sha256',required=True)
    args=parser.parse_args();run_executor(args.worker_plan,args.certificate,args.certificate_sha256)
