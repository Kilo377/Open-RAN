"""Fixed non-RT acceptance stimuli; never chooses an algorithm from KPIs."""
from copy import deepcopy
import json
from pathlib import Path
from models.policy import PolicyUpdate
from ric.non_rt.rapp_base import RAppBase


class FixedPolicyTimeline(RAppBase):
    def __init__(self,path,interval=10):
        super().__init__('fixed_policy_timeline')
        rows=json.loads(Path(path).read_text());self.entries={};self.done=set()
        if not isinstance(rows,list):raise ValueError('Policy timeline must be a list')
        for row in rows:
            slot=row['slot']
            if type(slot) is not int or slot<1 or slot%interval or slot in self.entries:raise ValueError('Unique policy slots must align with rApp period')
            if set(row)-{'slot','xapp_bindings','rule_based_parameters'}:raise ValueError('Unknown policy timeline field')
            self.entries[slot]=row

    def run(self,observation,policy_snapshot=None):
        slot=observation.slot_id
        row=self.entries.get(slot,{}) if slot not in self.done else {}
        self.done.add(slot)
        return PolicyUpdate(source_name=self.name,**{k:deepcopy(v) for k,v in row.items() if k!='slot'})
