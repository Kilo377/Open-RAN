"""Finite, fail-closed acceptance pipeline for one frozen source fingerprint."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sha=source_fingerprint(ROOT);start=time.perf_counter()
    status=dict(completed=False,passed=False,source_sha256=sha,stage='starting')
    def save():
        status['elapsed_s']=time.perf_counter()-start
        temp=out/'pipeline_status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'pipeline_status.json')
    save()
    jobs=[('pytest',['-m','pytest','-q','tests','--disable-warnings','--maxfail=1']),
        ('gate',['scripts/validation/validate_no_probe_environment.py','--stage','gate','--output',str(out/'gate')]),
        ('olla',['scripts/validation/validate_olla_traffic.py','--device','cuda','--output',str(out/'olla')]),
        ('control_lifecycle',['scripts/validation/validate_control_lifecycle.py','--feedback-mode','nr_direct','--output',str(out/'control_lifecycle')]),
        ('xapp_io',['scripts/validation/validate_xapp_io.py','--output',str(out/'xapp_io')]),
        ('moving_handover',['scripts/validation/validate_moving_handover.py','--output',str(out/'moving_handover')]),
        ('matrix',['scripts/validation/validate_no_probe_environment.py','--stage','matrix','--workers','3','--output',str(out/'matrix')])]
    try:
        for stage,argv in jobs:
            if source_fingerprint(ROOT)!=sha:raise RuntimeError('Source fingerprint changed')
            status['stage']=stage;save();print(f'START {stage}',flush=True)
            with (out/f'{stage}.log').open('w') as log:
                subprocess.run([sys.executable,*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'PASS {stage}',flush=True)
        assert source_fingerprint(ROOT)==sha
        status.update(completed=True,passed=True,stage='complete');save()
        files=[]
        for file in sorted(out.rglob('*')):
            if not file.is_file() or file.name=='data_manifest.json':continue
            h=hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
            files.append(dict(path=str(file.relative_to(out)),bytes=file.stat().st_size,sha256=h.hexdigest()))
        (out/'data_manifest.json').write_text(json.dumps(dict(files=files,count=len(files),bytes=sum(f['bytes'] for f in files),source_sha256=sha),indent=2))
    except BaseException as error:
        status.update(stage=status['stage'],error=repr(error),completed=False,passed=False);save()
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
