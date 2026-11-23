"""Wait for complete audited numerical evidence, then stage its UI manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--base-manifest',type=Path,required=True)
    args=parser.parse_args();study=args.study.resolve();repo=Path(__file__).resolve().parents[1]
    expected=sum(len(json.loads((study/(w+'-plan.json')).read_text())['cases'])for w in('agx1','agx2'))
    prefix=[str(Path.home()/'.local/bin/uv'),'run','--offline','--with','numpy','python']
    previous=-1;deadline=time.time()+6*3600;status=study/'analysis-pipeline-status.json'
    def save(**data):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),updated_unix_s=time.time(),**data),indent=2)+'\n');tmp.replace(status)
    def run(command,log):
        p=subprocess.run(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with(study/log).open('a')as stream:stream.write(p.stdout)
        if p.returncode:
            save(phase='processing_failed_retained',error=p.stdout[-2500:]);raise SystemExit(p.stdout)
    while time.time()<deadline:
        jobs={f.parent for name in('analysis-ready.json','mirror-verified.json')for f in study.glob('workers/*/jobs/*/'+name)}
        count=len(jobs)
        if count!=previous:
            save(phase='auditing_new_raw',mirrored=count,expected=expected)
            print(f'Auditing complete parameter measurements {count}/{expected}',flush=True)
            run([*prefix,'-m','experiments.fit_current_environment_gains','--study',str(study),'--audit-only'],'numerical-audit.log')
            previous=count;save(phase='waiting_for_raw',mirrored=count,expected=expected)
        if count==expected:
            save(phase='fitting',mirrored=count,expected=expected)
            run([*prefix,'-m','experiments.fit_current_environment_gains','--study',str(study)],'final-processing.log')
            run([*prefix,'scripts/publish-current-environment-calibration.py','--study',str(study),'--base-manifest',str(args.base_manifest)],'final-processing.log')
            save(phase='ready_for_ui_verification',mirrored=count,expected=expected,manifest=str(study/'ui/scenario-manifest.json'))
            print('Current-environment gains are ready for actual UI verification.',flush=True);return
        time.sleep(30)
    save(phase='watch_deadline',mirrored=previous,raw_retained=True)
    raise SystemExit('Bounded wait ended; raw measurements remain retained.')


if __name__=='__main__':main()
