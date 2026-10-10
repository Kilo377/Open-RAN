"""Verify actual RT and SYS GPU calls before accepting a Spark environment."""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def environment_info():
    import torch
    import mitsuba as mi
    import sionna.rt
    from importlib.metadata import version
    driver=''
    if platform.system()=='Linux' and torch.cuda.is_available():
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    return {'driver_report':driver, 'compute_capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None, 'python':platform.python_version(),'machine':platform.machine(),'system':platform.system(),
        'torch':torch.__version__,'sionna':version('sionna'),'sionna_rt':version('sionna-rt'),
        'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,
        'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',
        'rt_variant':mi.variant()}


def check(device='cuda',require_spark=True):
    info=environment_info()
    if require_spark and (info['machine'] not in ('aarch64','arm64') or info['system']!='Linux'):
        raise RuntimeError('Spark acceptance requires ARM64 Linux')
    if device=='cuda' and (not info['cuda_available'] or not info['rt_variant'].startswith('cuda')):
        raise RuntimeError('Both PyTorch and RT must execute on CUDA')
    if require_spark and ('GB10' not in info['gpu_name'] or info['compute_capability'][0]<10):
        raise RuntimeError('Spark acceptance requires the GB10 Blackwell GPU')
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.abstraction import SionnaAbstraction
    import numpy as np
    manifest,_,_=build_case(total_slots=2,device=device)
    provider=RtChannelProvider(manifest,ROOT/'channel_cache',device=device)
    snapshot=provider.snapshot(1)
    if not np.any(abs(snapshot.cfr)>0): raise RuntimeError('RT produced no connected links')
    model=SionnaAbstraction(device)
    result=model.evaluate(np.ones((14,12,1))*10,'qam64',0,1,1,'preflight')
    if device=='cuda' and not str(model.phy.device).startswith('cuda:'):
        raise RuntimeError('SYS PHYAbstraction is not executing on CUDA')
    info.update(rt_actual_call=True,sys_actual_call=True,sys_probe=result,cache_key=provider.key)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cpu-local',action='store_true'); p.add_argument('--lock',action='store_true')
    args=p.parse_args(); result=check('cpu' if args.cpu_local else 'cuda',not args.cpu_local)
    target=ROOT/'outputs/rt_sys/preflight';target.mkdir(parents=True,exist_ok=True)
    (target/'environment.json').write_text(json.dumps(result,indent=2))
    if args.lock:
        if args.cpu_local: raise ValueError('Only verified Spark GPU environment may produce the Spark lock')
        lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        # A locally cached wheel must not make the lock depend on this user's
        # absolute home path. Preserve its exact official artifact and hash.
        from importlib.metadata import distribution
        from urllib.parse import quote,unquote,urlsplit
        raw=distribution('torch').read_text('direct_url.json')
        if raw:
            direct=json.loads(raw)
            if direct['url'].startswith('file:'):
                filename=unquote(urlsplit(direct['url']).path.rsplit('/',1)[-1])
                sha=direct['archive_info'].get('hashes',{}).get('sha256')
                if not sha:raise ValueError('Installed Torch wheel hash missing')
                artifact='torch @ https://download.pytorch.org/whl/cu130/'+quote(filename)+'#sha256='+sha
                lock='\n'.join(artifact if row.startswith('torch @ ') else row for row in lock.splitlines())+'\n'
        (ROOT/'requirements-spark.lock.txt').write_text(lock)
    print(json.dumps(result,indent=2))
