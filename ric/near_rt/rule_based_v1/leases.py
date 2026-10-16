"""Re-arbitrate live rule requests and restore only their owned fields on withdrawal."""
from copy import deepcopy
from models.action import ConfigAction,DroppedControlItem


class RuleLeases:
    def __init__(self):
        self.requests={};self.baseline={};self.owned={};self.was_enabled={}

    def reconcile(self,manager,observation,winners,scopes,received,dropped=None):
        apps={i.xapp.name:i.xapp for i in manager.e2_interfaces if getattr(i.xapp,'rule_based_v1',False)}
        paper=getattr(observation,'paper_report',None) or {}
        for cid,item in paper.get('values',{}).get('cell',{}).items():
            if item.get('executed_config') and cid not in self.baseline:self.baseline[cid]=deepcopy(item['executed_config'])
        enabled={name for name in apps if manager._is_xapp_enabled(name)}
        for name,app in apps.items():
            if self.was_enabled.get(name,False) and name not in enabled and hasattr(app,'reset'):app.reset()
        self.was_enabled={name:name in enabled for name in apps}
        for key in list(self.requests):
            if key[0] not in enabled or (getattr(apps.get(key[0]),'library_instance',False) and not manager.instance_selection.enabled(key[0],key[1])):del self.requests[key]
        for action in received:
            name=action.source_name
            if name not in enabled or action.target_cell_id is None:continue
            if getattr(apps[name],'library_instance',False):manager.instance_selection.consume_hold(action,apps[name])
            for field in apps[name].controlled_fields:
                family={'slice_weight_adjustment':'slice_aware','latency_bias':'latency_aware'}.get(field)
                if family and action.scheduler_mode is not None and action.scheduler_mode!=family:
                    self.requests.pop((name,action.target_cell_id,field),None)
                    continue
                value=getattr(action,field,None)
                if getattr(apps[name],'library_instance',False) and field=='slice_weight_adjustment' and not value:continue
                if value is not None:self.requests[name,action.target_cell_id,field]=(apps[name].priority,deepcopy(value),name)
        live={}
        # A RET request owns the analog subspace, including while waiting for execution.
        ret_cells={cid for cid,cfg in self.baseline.items() if cfg.get('ret_enabled')}
        current=paper.get('values',{}).get('cell',{})
        ret_cells={cid for cid,item in current.items() if item.get('executed_config',{}).get('ret_enabled')}
        for (name,cid,field),value in self.requests.items():
            if field=='ret_enabled':
                if value[1]:ret_cells.add(cid)
                else:ret_cells.discard(cid)
        suppressed=set()
        for (name,cid,field),value in self.requests.items():
            key=(cid,field)
            if cid in ret_cells and field in ('codebook_id','beam_profile_id'):
                suppressed.add(key);continue
            if key not in live or (value[0],value[2])>(live[key][0],live[key][2]):live[key]=value
        for cid,field in suppressed:
            scope='cell:'+cid;winners.setdefault(scope,{}).pop(field,None)
            cached=manager.current_action_by_scope.get(scope)
            if cached:setattr(cached,field,None)
        for (cid,field),value in live.items():
            scope='cell:'+cid;scopes[scope]=('cell',cid)
            old=winners.setdefault(scope,{}).get(field)
            if old is None or value[0]>old[0] or old[2] in apps:winners[scope][field]=value
            self.owned[cid,field]=value[2]
        for (cid,field),owner in list(self.owned.items()):
            if (cid,field) in live:continue
            if (cid,field) in suppressed:
                scope='cell:'+cid
                winners.setdefault(scope,{}).pop(field,None)
                cached=manager.current_action_by_scope.get(scope)
                if cached:setattr(cached,field,None)
                del self.owned[cid,field]
                continue
            scope='cell:'+cid;scopes[scope]=('cell',cid);values=winners.setdefault(scope,{})
            if field in values:del self.owned[cid,field];continue
            base=deepcopy(self.baseline.get(cid,{}).get(field))
            if field=='prb_power_w' and base is None:
                # Clear the persisted vector too; no silent normalization.
                values['clear_prb_power']=(0,True,'rule_withdrawal')
                current=manager.current_action_by_scope.get(scope)
                if current:current.prb_power_w=None
            elif base is not None:values[field]=(0,base,'rule_withdrawal')
            del self.owned[cid,field]
        # Keep held requests for later re-arbitration, but do not report a
        # parameter as executable under a scheduler family that ignores it.
        for scope,values in winners.items():
            if 'scheduler_mode' not in values:continue
            mode=values['scheduler_mode'][1]
            for field,required in (('slice_weight_adjustment','slice_aware'),('latency_bias','latency_aware')):
                if mode==required or field not in values:continue
                _,value,owner=values[field]
                if owner not in apps:continue
                del values[field]
                cached=manager.current_action_by_scope.get(scope)
                if cached:setattr(cached,field,{} if field=='slice_weight_adjustment' else None)
                if dropped is not None and value:
                    dropped.append(DroppedControlItem(kind='field_write',source=owner,
                        reason='inactive_scheduler_family:'+str(mode),field_name=field,value=deepcopy(value)))
        if getattr(getattr(manager,'instance_selection',None),'mode','exclusive')=='conflict_experiment':
            for (name,cid,field),(priority,value,_) in self.requests.items():
                winner=winners.get('cell:'+cid,{}).get(field)
                app=apps[name]
                row=dict(reason='request_arbitrated',field=field,requested_value=deepcopy(value),
                    winner=winner[2] if winner else None,arbitrated_value=deepcopy(winner[1]) if winner else None,
                    status='selected' if winner and winner[2]==name else 'overridden' if winner else 'suppressed',
                    slot_id=observation.slot_id,**app.identity(cid))
                manager.rule_decisions=getattr(manager,'rule_decisions',[])+[row]
