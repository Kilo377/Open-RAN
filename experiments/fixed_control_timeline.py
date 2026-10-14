"""External fixed test stimuli through normal RIC admission/delay/receipts.

No observations choose actions; this adapter is not a research xApp policy.
"""
import json
from pathlib import Path
from ric.near_rt.xapp_base import XAppBase
from ric.xapp_contract import XAppContract,SUPPORTED_ACTIONS
from phy.rt_sys.paper_controls import COMMANDS
from models.action import ConfigAction,CommandAction,ControlDecision


class FixedControlTimeline(XAppBase):
    def __init__(self,path,interval=5):
        super().__init__('fixed_control_timeline',100,list(SUPPORTED_ACTIONS),
            contract=XAppContract((),SUPPORTED_ACTIONS,commands=frozenset(COMMANDS)))
        entries=json.loads(Path(path).read_text())
        if not isinstance(entries,list):raise ValueError('Timeline must be a list')
        self.entries={};self.inputs=[]
        for e in entries:
            slot=e['slot']
            if type(slot) is not int or slot<1 or slot%interval or slot in self.entries:raise ValueError('Unique positive timeline slots must align with xApp period')
            self.entries[slot]=e

    def run(self,observation,policy_snapshot):
        slot=observation.slot_id;e=self.entries.get(slot,{})
        self.inputs.append(dict(slot=slot,kpis=getattr(observation,'kpi_records',[])))
        config=ConfigAction(**e.get('config',{}),source_name=self.name)
        if e.get('config'):
            config.effective_from_slot=slot;config.effective_until_slot=slot+4
        commands=[]
        for i,value in enumerate(e.get('commands',[])):
            cmd=CommandAction(**value,source_name=self.name,control_request_id=f'timeline:{slot}:{i}')
            cmd.valid_until_slot=slot+4;commands.append(cmd)
        return ControlDecision(config_action=config,commands=commands)
