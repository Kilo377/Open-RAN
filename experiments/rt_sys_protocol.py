"""Held-out-map protocol. Generate plans or execute explicitly labeled probes.

This protocol does not certify the uncalibrated SYS EESM model or make an
algorithm-generalization claim. Both methods consume identical physical and
traffic seeds. Reflective evaluation requires nonempty frozen train memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phy.rt_sys.scenario import build_case
from phy.rt_sys.contracts import digest


def plan(proof=False):
    jobs=[]
    maps=('simple_street_canyon','munich')
    for fold,train_scene in enumerate(maps):
        for phase in ('train','validation','test'):
            scene=maps[1-fold] if phase=='test' else train_scene
            seeds={'train':(1,2,3),'validation':(11,12),'test':(101,102,103)}[phase]
            deployments={'train':(0,1),'validation':(2,),'test':(0,1,2)}[phase]
            counts=(12,24,36) if phase=='test' else (12,24)
            loads=(.5,1.,2.)
            if proof:seeds=seeds[:1];deployments=deployments[:1];counts=(4,);loads=(1.,)
            for seed in seeds:
                for deployment in deployments:
                    for ues in counts:
                        for load in loads:
                            for method in ('reflective','fixed_plan'):
                                jobs.append(dict(fold=fold,phase=phase,scene=scene,seed=seed,deployment_id=deployment,
                                    ues=ues,cells=2 if proof else 3,traffic_load=load,slots=400 if proof else 20000,
                                    warmup_ms=20 if proof else 2000,selection_mode=method))
    return jobs


def validate(jobs):
    for fold in (0,1):
        subset=[j for j in jobs if j['fold']==fold]
        byphase={p:[j for j in subset if j['phase']==p] for p in ('train','validation','test')}
        assert {j['scene'] for j in byphase['train']}=={j['scene'] for j in byphase['validation']}
        assert {j['scene'] for j in byphase['train']}.isdisjoint(j['scene'] for j in byphase['test'])
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            assert {j['seed'] for j in byphase[a]}.isdisjoint(j['seed'] for j in byphase[b])
        assert {j['deployment_id'] for j in byphase['train']}.isdisjoint(j['deployment_id'] for j in byphase['validation'])
        groups={}
        for job in subset:
            m,_,_=build_case(job['scene'],job['cells'],job['ues'],job['seed'],job['slots'],
                deployment_id=job['deployment_id'],traffic_load=job['traffic_load'])
            key=tuple((k,v) for k,v in job.items() if k!='selection_mode')
            groups.setdefault(key,[]).append(digest(m))
        assert all(len(hashes)==2 and len(set(hashes))==1 for hashes in groups.values())
    return {'held_out_maps':True,'disjoint_seeds':True,'held_out_validation_deployment':True,
        'paired_physical_and_traffic_inputs':True,'jobs':len(jobs)}


def execute(jobs,root,device,allow_uncalibrated):
    if not allow_uncalibrated:
        raise RuntimeError('Formal protocol PHY/HARQ/CSI validation is still pending; only explicitly labeled integration probes are enabled')
    root=root.resolve()
    if not root.is_relative_to(ROOT/'outputs'):raise ValueError('Protocol artifacts must stay under copy outputs/')
    if any((root/f'fold{fold}'/'trained_memory.json').exists() for fold in (0,1)):
        raise ValueError('Use a fresh protocol output directory; existing memory would silently warm-start training')
    root.mkdir(parents=True,exist_ok=True)
    (root/'plan.json').write_text(json.dumps({'jobs':jobs,'audit':validate(jobs),'phy_calibrated':False},indent=2))
    records=[]
    for index,job in enumerate(jobs):
        model=root/f'fold{job["fold"]}'/'trained_memory.json'
        model.parent.mkdir(exist_ok=True)
        out=root/f'job_{index:04d}_{job["phase"]}_{job["selection_mode"]}'
        out.mkdir(exist_ok=True)
        command=[sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--device',device,'--output',str(out)]
        for key,value in job.items():
            if key!='fold':command.extend(['--'+key.replace('_','-'),str(value)])
        if job['selection_mode']=='reflective':command.extend(['--model-state',str(model)])
        before=hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None
        with (out/'process.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record={'job':job,'returncode':result.returncode,'model_before_sha256':before,
            'model_after_sha256':hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None}
        records.append(record)
        (root/'protocol_status.json').write_text(json.dumps({'completed':len(records)==len(jobs) and result.returncode==0,
            'uncalibrated_phy_probe':True,'runs':records},indent=2))
        if result.returncode:raise RuntimeError(f'Protocol case failed; inspect {out}/process.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proof',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-uncalibrated',action='store_true');p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--output',type=Path,default=ROOT/'outputs/rt_sys_protocol')
    args=p.parse_args();jobs=plan(args.proof)
    if args.execute:execute(jobs,args.output,args.device,args.allow_uncalibrated)
    else:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'plan.json').write_text(json.dumps({'jobs':jobs,'audit':validate(jobs),'phy_calibrated':False},indent=2))
        print(json.dumps(validate(jobs),indent=2))
