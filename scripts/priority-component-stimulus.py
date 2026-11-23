"""One fixed UE-weight component through normal RIC validation and delay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--owned-root',type=Path,required=True);own,args=p.parse_known_args()
    root=own.owned_root.resolve();sys.path.insert(0,str(root))
    from ric.near_rt.xapp_base import XAppBase
    from ric.xapp_contract import XAppContract,validate_action
    from models.action import ConfigAction,ControlDecision
    import experiments.fixed_control_timeline as adapter
    class FixedPriorityProbe(XAppBase):
        def __init__(self,path,interval=5):
            fields=frozenset(('scheduler_mode','ue_scheduler_weights'))
            super().__init__('fixed_control_timeline',100,list(fields),contract=XAppContract((),fields))
            self.spec=json.loads(Path(path).read_text());self.sent=False;self.inputs=[]
            if set(self.spec)!={'switch_slot','scheduler_mode','ue_scheduler_weights'}:raise ValueError('Invalid fixed probe')
        def run(self,observation,policy_snapshot):
            if self.sent or observation.slot_id<self.spec['switch_slot']:return ControlDecision()
            if observation.slot_id!=self.spec['switch_slot']:raise ValueError('Fixed probe missed decision boundary')
            self.sent=True;actions=[]
            for cell in observation.cell_observations:
                members=[u.ue_id for u in observation.ue_observations if u.serving_cell_id==cell.cell_id]
                actions.append(ConfigAction(source_name=self.name,target_cell_id=cell.cell_id,scheduler_mode=self.spec['scheduler_mode'],
                    ue_scheduler_weights={u:self.spec['ue_scheduler_weights'][u]for u in members}))
            failures=[v for a in actions for v in validate_action(a,self.contract,observation)]
            if failures:raise ValueError('Probe failed admission contract: '+str(failures))
            self.inputs.append(dict(slot=observation.slot_id,spec=deepcopy(self.spec)))
            return ControlDecision(config_actions=actions)
    adapter.FixedControlTimeline=FixedPriorityProbe
    sys.argv=[str(root/'experiments/run_rt_sys.py'),*args]
    runpy.run_path(str(root/'experiments/run_rt_sys.py'),run_name='__main__')


if __name__=='__main__':main()
