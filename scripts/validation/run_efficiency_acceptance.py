"""Ordered Spark acceptance pipeline; never copies raw results to a client."""
import argparse,json,subprocess,sys,time,shutil,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phy.rt_sys.contracts import source_fingerprint
from scripts.validation.validate_efficiency import read


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);sha=source_fingerprint(ROOT)
    state=dict(source_sha256=sha,completed=False,stage='starting',stages=[],started_epoch=time.time())
    source_files=[p for directory in ('core','phy','ran','ric','state','models','traffic','experiments','configs','scripts','scenarios','tests')
        for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml')]
    tree={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (out/'source_and_validation_sha256.json').write_text(json.dumps(tree,indent=2))
    if not (out/'executed_sources.tar.gz').exists():
        with tarfile.open(out/'executed_sources.tar.gz','w:gz') as archive:
            for p in source_files:archive.add(p,arcname=str(p.relative_to(ROOT)))
    def save(validate=True):
        if validate:
            assert source_fingerprint(ROOT)==sha,'Source changed during acceptance'
            assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in tree.items()),'Source or validation script changed during acceptance'
        (out/'pipeline_status.json').write_text(json.dumps(state,indent=2))
    def execute(label,script,argv):
        target=out/label;state['stage']=label;save()
        if (target/'validation.json').exists():
            report=read(target/'validation.json')
            assert report.get('source_sha256',sha)==sha
        else:
            with (out/f'{label}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/script),*argv,'--output',str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=read(target/'validation.json')
        assert all(report.get(key,True) for key in ('completed','passed','structural_passed','equivalence_passed')),(label,'Refusing a failed/incomplete stage')
        state['stages'].append(dict(stage=label,path=str(target),completed=True));save();return report
    try:
        gate=read(a.gate/'validation.json')
        assert gate['completed'] and gate['equivalence_passed'] and gate['source_sha256']==sha
        state['gate']=str(a.gate);save()
        # Gate timing is a budget estimate, never the acceptance speed claim.
        gate_rows=[c for c in gate['cases'] if c['slots']==128]
        estimated=sum(c['timing']['total_s']*512/128*3 for c in gate_rows)
        estimate=dict(serial_performance_estimate_s=estimated,free_disk_bytes=shutil.disk_usage(out).free,
            maximum_baseline_cases=18,baseline_slots_per_case=4000,moving_cases=8,moving_slots_per_case=4000,
            note='Performance repeats run serially; correctness matrices may use three processes.')
        (out/'initial_expansion_estimate.json').write_text(json.dumps(estimate,indent=2))
        performance=execute('performance','scripts/validation/validate_efficiency.py',
            ['--stage','performance','--reference-root',str(a.reference_root)])
        # Never expand a mode with a >5% median regression in any fixture.
        labels=['experiment_scalar','experiment_batched']
        eligible=[label for label in labels if all(r['speedups'][label]>=1/1.05 for r in performance['performance'])]
        if not eligible:raise RuntimeError('Both exact experiment paths regress >5%; inspect before expanding')
        label=max(eligible,key=lambda label:sum(r['speedups'][label] for r in performance['performance']))
        execution=label.split('_')[1];state['selected_execution']=execution
        state['performance_target_met']=performance['performance_target_met'];save()
        estimates=[]
        for r in performance['performance']:
            multiplicity=3 if r['ues']==4 else 6
            estimates.append(dict(scene=r['scene'],ues=r['ues'],cases=multiplicity,
                estimated_case_s=r['medians_s'][label]*4000/(512-128),
                interpretation='Serial steady-rate extrapolation; concurrency has its own throughput.'))
        (out/'baseline_expansion_estimate.json').write_text(json.dumps(dict(estimates=estimates,
            workers=3,free_disk_bytes=shutil.disk_usage(out).free),indent=2))
        for scene in ('simple_street_canyon','munich'):
            for execution_mode in ('scalar','batched'):
                folder=out/'profiles'/f'{scene}_{execution_mode}'
                if not (folder/'diagnostic_profile.json').exists():
                    folder.parent.mkdir(exist_ok=True)
                    with (out/f'profile_{scene}_{execution_mode}.log').open('w') as log:
                        subprocess.run([sys.executable,'scripts/validation/profile_exact_nr.py','--scene',scene,
                            '--execution',execution_mode,'--output',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        execute('baseline','scripts/validation/validate_baseline_dynamic.py',['--stage','baseline','--workers','3','--execution',execution])
        execute('controlled_olla','scripts/validation/validate_dynamic_olla.py',[])
        execute('moving_gate','scripts/validation/validate_baseline_dynamic.py',['--stage','moving_gate','--workers','1','--execution',execution])
        execute('moving','scripts/validation/validate_baseline_dynamic.py',['--stage','moving','--gate',str(out/'moving_gate'),'--workers','3','--execution',execution])
        execute('handover_events','scripts/validation/validate_paper_handover_rt.py',[])
        state.update(completed=True,stage='completed',elapsed_s=time.time()-state['started_epoch']);save()
        reference_data=a.reference_root/'outputs/efficiency_reference'
        if not (reference_data/'data_manifest.json').exists():
            subprocess.run([sys.executable,'scripts/validation/verify_artifact_manifest.py','--root',str(reference_data),
                '--manifest',str(reference_data/'data_manifest.json'),'--create',
                '--source-sha',read(next(reference_data.glob('*/run_status.json')))['source_sha256']],check=True,cwd=ROOT)
        subprocess.run([sys.executable,'scripts/validation/verify_artifact_manifest.py','--root',str(reference_data),
            '--manifest',str(reference_data/'data_manifest.json'),'--strict-inventory',
            '--receipt',str(out/'reference_data_verification.json')],check=True,cwd=ROOT)
        # Manifest after every writer/process log is closed; driver log stays outside inventory.
        subprocess.run([sys.executable,'scripts/validation/verify_artifact_manifest.py','--root',str(out),
            '--manifest',str(out/'data_manifest.json'),'--create','--source-sha',sha],check=True,cwd=ROOT)
        subprocess.run([sys.executable,'scripts/validation/verify_artifact_manifest.py','--root',str(out),
            '--manifest',str(out/'data_manifest.json'),'--strict-inventory','--receipt',str(out.parent/(out.name+'_verification.json'))],check=True,cwd=ROOT)
    except BaseException as e:
        state.update(stage='failed',error=f'{type(e).__name__}: {e}',elapsed_s=time.time()-state['started_epoch'])
        save(validate=False);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    main(p.parse_args())
