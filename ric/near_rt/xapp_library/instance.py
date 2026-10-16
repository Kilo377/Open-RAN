"""One deployable identity, with isolated per-cell decision engines."""
from collections import Counter
from copy import deepcopy
import time
import math
from models.action import ConfigAction,ControlDecision
from ric.near_rt.xapp_base import XAppBase
from ric.xapp_contract import XAppContract
from .catalog import IMPLEMENTATIONS
from .algorithms import Algorithm


class XAppInstance(XAppBase):
    library_instance=True
    rule_based_v1=True  # Reuse scoped arbitration/leases, not legacy selection.

    def __init__(self,instance_id,implementation_id,cell_ids,parameters=None,priority=None,slot_duration_ms=.5,initial_bindings=None,context_cell_ids=None):
        if not isinstance(instance_id,str) or not instance_id.strip():raise ValueError('Nonempty instance_id required')
        if implementation_id not in IMPLEMENTATIONS:raise ValueError('Unknown implementation: '+str(implementation_id))
        if not cell_ids or len(set(cell_ids))!=len(cell_ids):raise ValueError('Explicit unique allowed cells required')
        self.spec=deepcopy(IMPLEMENTATIONS[implementation_id]);self.category_id=self.spec['category_id']
        self.implementation_id=implementation_id;self.kind=self.category_id;self.version=self.spec['version']
        self.allowed_cells=tuple(cell_ids);self.parameters=deepcopy(parameters or {});self.dt=slot_duration_ms
        self.initial_bindings=deepcopy(initial_bindings);self.children={};self.active_cells=set();self.tokens={}
        self.statistics=Counter();self.last_decisions=[];self.decision_seconds=0.
        self.ret_last_action=None
        contract=XAppContract(tuple(self.spec['required_kpis']),frozenset(self.spec['writable_actions']),
            commands=frozenset(self.spec['commands']),cell_ids=self.allowed_cells,partial_entities=True,
            include_context_entities=True,command_target_cell_ids=tuple(context_cell_ids or self.allowed_cells),
            max_age_slots=math.ceil(self.parameters.get('max_age_ms',self.spec['defaults']['max_age_ms'])/slot_duration_ms))
        super().__init__(instance_id,self.spec['priority'] if priority is None else priority,self.spec['writable_actions'],contract)
        self.validate_parameters(self.parameters)

    def new_child(self,cid,parameters=None):
        child=Algorithm(self.category_id,self.spec['algorithm'],name=self.name,priority=self.priority,
            parameters=self.parameters if parameters is None else parameters,cell_ids=(cid,),slot_duration_ms=self.dt)
        from dataclasses import replace
        child.contract=replace(child.contract,required_kpis=tuple(self.spec['required_kpis']))
        return child

    def validate_parameters(self,overrides):
        if not isinstance(overrides,dict):raise ValueError('Instance parameters must be a mapping')
        self.new_child(self.allowed_cells[0],{**self.parameters,**overrides})

    def runtime_contract(self,policy):
        from dataclasses import replace
        overrides=(getattr(policy,'rule_based_parameters',{}) or {}).get(self.name,{})
        age=overrides.get('max_age_ms',self.parameters.get('max_age_ms',self.spec['defaults']['max_age_ms']))
        return replace(self.contract,max_age_slots=math.ceil(age/self.dt))

    def select(self,cells,tokens):
        cells=set(cells)
        for cid in self.active_cells ^ cells:self.children.pop(cid,None)
        self.active_cells=cells;self.tokens=dict(tokens)

    def reset(self):
        self.children.clear()
        self.ret_last_action=None

    def run(self,observation,policy_snapshot):
        started=time.perf_counter()
        self.last_decisions=[];actions=[];commands=[]
        cells=sorted(self.active_cells)
        try:
            overrides=(getattr(policy_snapshot,'rule_based_parameters',{}) or {}).get(self.name,{})
            if overrides:self.validate_parameters(overrides)
            for cid in cells:
                if cid in self.children:self.children[cid].capture_reports(observation,policy_snapshot)
        except (ValueError,TypeError) as error:
            self.last_decisions=[dict(reason='nonrt_policy_rejected',detail=str(error),**self.identity(cells[0]))]
            self.statistics['nonrt_policy_rejected']+=1;self.decision_seconds+=time.perf_counter()-started
            return ControlDecision(config_action=ConfigAction(source_name=self.name,target_cell_id=cells[0]))
        if self.category_id=='ret_coverage' and cells:
            overrides=(getattr(policy_snapshot,'rule_based_parameters',{}) or {}).get(self.name,{})
            interval=overrides.get('min_action_ms',self.parameters.get('min_action_ms',self.spec['min_action_ms']))
            if self.ret_last_action is not None and (observation.slot_id-self.ret_last_action)*self.dt<interval:
                self.last_decisions=[dict(reason='action_dwell',**self.identity(cells[0]))]
                self.statistics['action_dwell']+=1
                self.decision_seconds+=time.perf_counter()-started
                return ControlDecision(config_action=ConfigAction(source_name=self.name))
            index=int(observation.slot_id*self.dt/max(interval,1))%len(cells)
            cells=cells[index:]+cells[:index]
            pending=[cid for cid in cells if cid in self.children and self.children[cid].state.get(cid,{}).get('pending')]
            if pending:cells=pending
        for cid in cells:
            if cid not in self.children:self.children[cid]=self.new_child(cid)
            child=self.children[cid]
            decision=child.run(observation,policy_snapshot)
            child.decision_seconds=0.
            for row in child.last_decisions:
                row=deepcopy(row);row.update(self.identity(cid))
                self.last_decisions.append(row);self.statistics[row['reason']]+=1
            for a in decision.config_actions:
                if a.target_cell_id!=cid:continue
                a._library_provenance=self.identity(cid);actions.append(a)
                if self.category_id=='ret_coverage':self.ret_last_action=observation.slot_id
            for command in decision.commands:
                command._library_provenance=self.identity(cid);commands.append(command)
            if self.category_id=='ret_coverage' and actions:break
        self.decision_seconds+=time.perf_counter()-started
        return ControlDecision(config_actions=actions or [ConfigAction(source_name=self.name,target_cell_id=next(iter(sorted(self.active_cells)),None))],commands=commands)

    def identity(self,cid):
        return dict(instance_id=self.name,implementation_id=self.implementation_id,category_id=self.category_id,
            implementation_version=self.version,cell_id=cid,selection_version=self.tokens.get(cid,0))

    def descriptor(self):
        return {**deepcopy(self.spec),'instance_id':self.name,'allowed_cell_ids':list(self.allowed_cells),
            'parameters':deepcopy(self.parameters),'active_cell_ids':sorted(self.active_cells)}
