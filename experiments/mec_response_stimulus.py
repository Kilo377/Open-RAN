"""Fixed scheduler-only calibration stimuli through normal RIC admission.

Run this file outside the pinned worker source tree. Only this process replaces
the fixed-test adapter; no project/global files or learned xApps are modified.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys


def validate(spec):
    if set(spec)!={'schema','name','switch_slot','scheduler_mode','ue_scheduler_weights'}or spec['schema']!='oran.mec-response-stimulus.v1':raise ValueError('Invalid fixed stimulus')
    if spec['switch_slot']!=560:raise ValueError('Matched supplement uses slot 560')
    if set(spec['scheduler_mode'])!={'cell_0','cell_1','cell_2'}or any(v not in('round_robin','latency_aware','qos_deadline')for v in spec['scheduler_mode'].values()):raise ValueError('Invalid cell modes')
    if set(spec['ue_scheduler_weights'])!={'ue_0','ue_1','ue_2','ue_4','ue_6','ue_8'}or any(type(v)not in(int,float)or not .5<=v<=4 for v in spec['ue_scheduler_weights'].values()):raise ValueError('Invalid weights')
    return deepcopy(spec)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--owned-root',type=Path,required=True)
    own,args=parser.parse_known_args();root=own.owned_root.resolve();sys.path.insert(0,str(root))
    from ric.near_rt.xapp_base import XAppBase
    from ric.xapp_contract import XAppContract,validate_action
    from models.action import ConfigAction,ControlDecision
    import experiments.fixed_control_timeline as adapter
    fields=frozenset(('scheduler_mode','ue_scheduler_weights','latency_bias'))
    class FixedSchedulerStimulus(XAppBase):
        def __init__(self,path,interval=5):
            super().__init__('fixed_control_timeline',100,list(fields),contract=XAppContract((),fields))
            self.spec=validate(json.loads(Path(path).read_text()));self.inputs=[];self.sent=False
        def run(self,observation,policy_snapshot):
            if observation.slot_id<self.spec['switch_slot']or self.sent:return ControlDecision()
            if observation.slot_id!=self.spec['switch_slot']:raise ValueError('Stimulus missed matched RIC boundary')
            self.sent=True;actions=[]
            for cell in observation.cell_observations:
                cid=cell.cell_id;users=[u.ue_id for u in observation.ue_observations if u.serving_cell_id==cid]
                mode=self.spec['scheduler_mode'][cid]
                actions.append(ConfigAction(source_name=self.name,target_cell_id=cid,scheduler_mode=mode,
                    ue_scheduler_weights={u:self.spec['ue_scheduler_weights'][u]for u in users},
                    latency_bias=1. if mode=='latency_aware'else None))
            problems=[p for action in actions for p in validate_action(action,self.contract,observation)]
            if problems:raise ValueError('Fixed stimulus violates admission contract: '+str(problems))
            return ControlDecision(config_actions=actions)
    adapter.FixedControlTimeline=FixedSchedulerStimulus
    sys.argv=[str(root/'experiments/run_rt_sys.py'),*args]
    runpy.run_path(str(root/'experiments/run_rt_sys.py'),run_name='__main__')


if __name__=='__main__':main()
