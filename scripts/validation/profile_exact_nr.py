"""Short diagnostic only: CUDA events for codec stages, one final fence."""
import argparse,json,os,sys,time
from collections import defaultdict
from functools import wraps
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main(a):
    import torch
    from phy.rt_sys.reference import NrReferenceAbstraction
    from experiments.run_rt_sys import run_case,parse_args
    from phy.rt_sys import channel
    from core.simulator import Simulator
    from core.state_updater import StateUpdater
    from core.measurement_updater import MeasurementUpdater
    stages=defaultdict(lambda:dict(calls=0,host_wall_s=0.));events=defaultdict(list)
    def wrap(function,label,gpu=False):
        @wraps(function)
        def call(*args,**kwargs):
            if gpu:
                start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
            start=time.perf_counter()
            try:return function(*args,**kwargs)
            finally:
                stages[label]['calls']+=1;stages[label]['host_wall_s']+=time.perf_counter()-start
                if gpu:end_event.record();events[label].append((start_event,end_event))
        return call
    waveform=NrReferenceAbstraction.waveform
    def observed(self,*args,**kw):
        wave=waveform(self,*args,**kw)
        if not getattr(wave,'_diagnostic_wrapped',False):
            for name,label in (('encoder','encoding'),('mapper','mapping'),('demapper','demapping'),('decoder','decoding')):
                setattr(wave,name,wrap(getattr(wave,name),label,True))
            wave._diagnostic_wrapped=True
        return wave
    NrReferenceAbstraction.waveform=observed
    channel.reconstruct_paths=wrap(channel.reconstruct_paths,'cfr_reconstruction')
    channel.RtChannelProvider.snapshot=wrap(channel.RtChannelProvider.snapshot,'path_read_or_generation_and_cfr')
    channel.SionnaRayTracer.generate_paths=wrap(channel.SionnaRayTracer.generate_paths,'rt_path_generation')
    Simulator._run_managed_cell_schedulers=wrap(Simulator._run_managed_cell_schedulers,'scheduling')
    StateUpdater.apply_phy_result=wrap(StateUpdater.apply_phy_result,'state_update')
    MeasurementUpdater.update=wrap(MeasurementUpdater.update,'measurement_update')
    from ric.controller_manager import ControllerManager
    ControllerManager.update=wrap(ControllerManager.update,'control_observation_and_decision')
    sys.argv=['run_rt_sys.py','--scene',a.scene,'--cells','3','--ues','24','--slots','128','--seed','512',
        '--config',str(ROOT/'configs/rt_sys/efficiency_experiment.json'),'--nr-execution',a.execution,
        '--initial-queue-bytes','0','--ue-speed-mps','0','--output',str(a.output)]
    run_case(parse_args());torch.cuda.synchronize()
    for label,items in events.items():stages[label]['cuda_stream_elapsed_s']=sum(s.elapsed_time(e) for s,e in items)/1000
    (a.output/'diagnostic_profile.json').write_text(json.dumps(dict(stages=dict(stages),
        clock='Host wall and same-stream CUDA events; one final synchronization.',
        interpretation='Instrumented diagnostic, not the serial speed benchmark. Inclusive stages and stream idle gaps must not be added as exclusive kernel time.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=['simple_street_canyon','munich'],required=True)
    p.add_argument('--execution',choices=['scalar','batched'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    main(a)
