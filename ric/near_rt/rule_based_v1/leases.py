"""Re-arbitrate live rule requests and restore only their owned fields on withdrawal."""
from copy import deepcopy
import math
from models.action import ConfigAction,DroppedControlItem
from .random_arbiter import SCHEDULER_BUNDLE_FIELDS, SCHEDULER_RESERVED_FIELDS


class RuleLeases:
    UE_SERVICE_FIELDS = frozenset({'ue_drx_configs', 'ue_offload_fraction', 'ue_scheduler_weights'})

    def __init__(self):
        self.requests={};self.baseline={};self.owned={};self.was_enabled={}
        self.ue_service_baseline={field:{} for field in self.UE_SERVICE_FIELDS}
        self.ue_service_owned={}
        self.ue_service_notes=set()
        self.request_metadata={}
        self.random_outputs=[]
        self.scheduler_requests={}
        self.scheduler_managed_cells=set()
        self.scheduler_controlled_uids=set()
        self.random_scheduler_outputs=[]

    def _service_note(self,dropped,owner,cid,field,uid,reason):
        key=(owner,cid,field,uid,reason)
        if key in self.ue_service_notes:return
        self.ue_service_notes.add(key)
        if dropped is not None:
            dropped.append(DroppedControlItem(kind='field_write',source=owner,
                reason=reason,field_name=field,value={'ue_id':uid,'cell_id':cid}))

    def _service_context(self,manager,observation,paper):
        association={u.ue_id:u.serving_cell_id for u in getattr(observation,'ue_observations',[])}
        for uid,item in paper.get('values',{}).get('ue',{}).items():
            if item.get('association',{}).get('cell_id') is not None:
                association[uid]=item['association']['cell_id']
        executed={}
        for cid,item in paper.get('values',{}).get('cell',{}).items():
            for field in self.UE_SERVICE_FIELDS:
                values=item.get('executed_config',{}).get(field)
                if not isinstance(values,dict):continue
                for uid,value in values.items():
                    self.ue_service_baseline[field].setdefault(uid,deepcopy(value))
                    if association.get(uid)==cid:executed[field,uid]=deepcopy(value)
        # Persistent cached actions must not keep re-applying an old source
        # cell's UID map after handover. Policies already on the UE stay put.
        for cached in getattr(manager,'current_action_by_scope',{}).values():
            cid=getattr(cached,'target_cell_id',None)
            if cid is None:continue
            for field in self.UE_SERVICE_FIELDS:
                values=getattr(cached,field,None)
                if isinstance(values,dict):
                    kept={uid:deepcopy(value) for uid,value in values.items() if association.get(uid)==cid}
                    setattr(cached,field,kept or None)
        return association,executed

    def _service_filter(self,apps,association,name,cid,field,values,dropped):
        if not isinstance(values,dict):return {}
        result={}
        allowed=getattr(apps[name],'allowed_cells',())
        for uid,value in values.items():
            reason=( 'ue_service_request_moved_cell' if association.get(uid)!=cid else
                     'ue_service_request_outside_allowed_cells' if cid not in allowed else
                     'ue_service_baseline_unavailable' if uid not in self.ue_service_baseline[field] else None)
            if reason:self._service_note(dropped,name,cid,field,uid,reason)
            else:result[uid]=deepcopy(value)
        return result

    def _service_restore(self,manager,apps,enabled,association,executed,winners,scopes,dropped):
        # Ownership follows the UID across HO; it is not a stale source-cell
        # request. Only winning maps acquire ownership, preserving arbitration.
        for scope,fields in winners.items():
            cid=scopes.get(scope,(None,None))[1]
            for field in self.UE_SERVICE_FIELDS:
                if field == 'ue_offload_fraction' and getattr(manager, 'direct_conflict_arbiter', None) is not None:continue
                if field=='ue_scheduler_weights' and cid in self.scheduler_managed_cells:continue
                value=fields.get(field)
                if value is None or value[2] not in apps:continue
                for uid in value[1]:self.ue_service_owned[field,uid]=(value[2],cid)
        for (field,uid),(owner,source) in list(self.ue_service_owned.items()):
            if field == 'ue_offload_fraction' and getattr(manager, 'direct_conflict_arbiter', None) is not None:continue
            if field=='ue_scheduler_weights' and uid in self.scheduler_controlled_uids:continue
            app=apps.get(owner)
            selected=getattr(manager,'instance_selection',None)
            active=owner in enabled and (not getattr(app,'library_instance',False) or
                (selected is not None and selected.enabled(owner,source)))
            if active:continue
            cid=association.get(uid)
            base=self.ue_service_baseline[field].get(uid)
            if cid is None or app is None or cid not in getattr(app,'allowed_cells',()):
                self._service_note(dropped,owner,cid,field,uid,'ue_service_restore_deferred_outside_allowed_cells')
                continue
            if (field,uid) in executed and executed[field,uid]==base:
                del self.ue_service_owned[field,uid]
                continue
            scope='cell:'+cid;scopes[scope]=('cell',cid);fields=winners.setdefault(scope,{})
            existing=fields.get(field)
            if existing is not None and existing[2]!='rule_withdrawal':
                # Never merge separately owned whole maps or attach a restore
                # to another xApp's queued provenance. Wait until it is free.
                if isinstance(existing[1],dict) and uid in existing[1]:
                    del self.ue_service_owned[field,uid]
                else:self._service_note(dropped,owner,cid,field,uid,'ue_service_restore_deferred_field_owned')
                continue
            values=deepcopy(existing[1]) if existing else {}
            values[uid]=deepcopy(base)
            fields[field]=(0,values,'rule_withdrawal')
            # Keep ownership until the delayed executed-state report confirms
            # restoration, allowing rejected HO-time restores to retry.

    def _random_offload(self,manager,apps,enabled,association,executed,winners,scopes,slot):
        arbiter=manager.direct_conflict_arbiter
        candidates={}
        for (name,cid,field),(priority,values,_) in self.requests.items():
            if field!='ue_offload_fraction':continue
            meta=self.request_metadata.get((name,cid,field),{})
            if not arbiter.request_valid(meta,slot):continue
            provenance=meta.get('provenance')
            if provenance is not None and not manager.instance_selection.valid(provenance):continue
            for uid,value in values.items():
                candidates.setdefault(uid,[]).append(dict(source=name,value=value,priority=priority,**deepcopy(meta)))
        # This mode owns UID entries, not whole maps. Clear only this cached
        # field so an old cell map cannot be reissued alongside the selection.
        for cached in manager.current_action_by_scope.values():cached.ue_offload_fraction=None
        for fields in winners.values():fields.pop('ue_offload_fraction',None)
        for key in list(self.owned):
            if key[1]=='ue_offload_fraction':del self.owned[key]
        self.random_outputs=[]
        uids=set(candidates)|set(arbiter.current)|{uid for field,uid in self.ue_service_owned if field=='ue_offload_fraction'}
        owners={}
        for uid in sorted(uids):
            cid=association.get(uid)
            baseline=self.ue_service_baseline['ue_offload_fraction'].get(uid)
            if cid is None or not any(cid in app.allowed_cells for app in apps.values()):
                arbiter.current.pop(uid,None)
                continue
            result,event=arbiter.choose(uid,cid,slot,candidates.get(uid,[]),baseline)
            if event:manager.rule_decisions=getattr(manager,'rule_decisions',[])+[event]
            if result is None:continue
            owner=result['winner']
            scope='cell:'+cid;scopes[scope]=('cell',cid)
            entry=winners.setdefault(scope,{}).setdefault('ue_offload_fraction',(0,{},None))
            entry[1][uid]=deepcopy(result['selected_value'])
            owners.setdefault(cid,set()).add(owner)
            if owner=='rule_withdrawal':
                # A delayed observation can still show native p=.5 after a
                # newer p=1 acknowledgement. Never suppress its restoration.
                actual=getattr(manager,'_mitigation_executed_values',{}).get(uid,executed.get(('ue_offload_fraction',uid)))
                if actual==baseline:
                    self.ue_service_owned.pop(('ue_offload_fraction',uid),None)
                    prior=getattr(manager,'_mitigation_executed_owners',{}).pop(uid,None)
                    if prior is not None:
                        manager.rule_decisions.append(dict(reason='direct_conflict_baseline_confirmed',slot_id=slot,
                            ue_id=uid,cell_id=cid,field='ue_offload_fraction',value=baseline,previous_owner=prior))
                    continue
            else:self.ue_service_owned['ue_offload_fraction',uid]=(owner,cid)
            self.random_outputs.append(deepcopy(result))
        for cid,values in owners.items():
            owner=next(iter(values)) if len(values)==1 else None
            self.owned[cid,'ue_offload_fraction']=owner
            field=winners['cell:'+cid]['ue_offload_fraction']
            winners['cell:'+cid]['ue_offload_fraction']=(0,field[1],owner)

    @staticmethod
    def _scheduler_value(action,members):
        mode=getattr(action,'scheduler_mode',None)
        weights=getattr(action,'ue_scheduler_weights',None)
        if not isinstance(mode,str) or mode.strip() not in ('round_robin','latency_aware','qos_deadline') or not isinstance(weights,dict):
            raise ValueError('uniform_random scheduler bundle requires a mode and full served-UID map')
        if set(weights)!=set(members):
            raise ValueError('uniform_random scheduler bundle requires the full current served-UID map')
        max_weight=4 if mode.strip()=='qos_deadline' else 2
        if any(type(w) not in (int,float) or not math.isfinite(w) or not .5<=w<=max_weight for w in weights.values()):
            raise ValueError('invalid uniform_random scheduler bundle weights')
        if action.latency_bias not in (None,1.) or action.slice_weight_adjustment:
            raise ValueError('uniform_random scheduler bundle does not support non-neutral scheduler tuning')
        expected_hint='latency_bias' if mode.strip()=='latency_aware' else 'balanced'
        if action.scheduler_policy_hint not in (None,expected_hint):
            raise ValueError('uniform_random scheduler bundle has an incompatible policy hint')
        return dict(scheduler_mode=mode.strip(),ue_scheduler_weights=deepcopy(weights))

    def _prepare_scheduler_requests(self,manager,apps,enabled,association,received,slot):
        arbiter=getattr(manager,'scheduler_conflict_arbiter',None)
        if arbiter is None:return
        selection=getattr(manager,'instance_selection',None)
        for key,request in list(self.scheduler_requests.items()):
            name,cid=key;provenance=request.get('provenance')
            members={uid for uid,cell in association.items() if cell==cid}
            reason=('scheduler_source_withdrawn' if name not in enabled or
                (getattr(apps.get(name),'library_instance',False) and not selection.enabled(name,cid)) else
                'scheduler_source_replaced' if provenance is not None and not selection.valid(provenance) else
                'scheduler_request_expired' if request.get('effective_until_slot') is not None and slot>request['effective_until_slot'] else
                'scheduler_membership_changed' if set(request['value']['ue_scheduler_weights'])!=members else None)
            if reason:
                del self.scheduler_requests[key]
                manager.rule_decisions=getattr(manager,'rule_decisions',[])+[dict(reason='direct_conflict_scheduler_request_invalidated',
                    invalidation_reason=reason,slot_id=slot,field='scheduler_bundle',source=name,cell_id=cid,
                    request_id=request.get('request_id'))]
        for action in received:
            name=action.source_name;app=apps.get(name);cid=action.target_cell_id
            if app is None or name not in enabled or not SCHEDULER_BUNDLE_FIELDS<=set(app.controlled_fields):continue
            if action.scheduler_mode is None and action.ue_scheduler_weights is None:continue
            if action.apply_scope!='cell' or cid is None or cid not in app.allowed_cells:
                raise ValueError('uniform_random scheduler bundle requires an explicit allowed cell scope')
            if getattr(app,'library_instance',False) and not selection.enabled(name,cid):
                raise ValueError('uniform_random scheduler bundle requires a selected source in that cell')
            members={uid for uid,cell in association.items() if cell==cid}
            value=self._scheduler_value(action,members)
            meta=dict(request_id=getattr(action.metadata,'decision_id',None),
                effective_from_slot=action.effective_from_slot,effective_until_slot=action.effective_until_slot,
                provenance=deepcopy(app.identity(cid)) if getattr(app,'library_instance',False) else None)
            self.scheduler_requests[name,cid]=dict(source=name,priority=app.priority,value=value,
                framework_tuning={'latency_bias':1.} if value['scheduler_mode']=='latency_aware' else {},**meta)
            for field in SCHEDULER_RESERVED_FIELDS:self.requests.pop((name,cid,field),None)
            manager.rule_decisions=getattr(manager,'rule_decisions',[])+[dict(reason='direct_conflict_request_received',
                field='scheduler_bundle',source=name,cell_id=cid,slot_id=slot,requested_value=deepcopy(value),**deepcopy(meta))]
        self.scheduler_managed_cells.update(cid for (_,cid),r in self.scheduler_requests.items() if arbiter.request_valid(r,slot))
        allowed={cid for app in apps.values() if SCHEDULER_BUNDLE_FIELDS<=set(app.controlled_fields) for cid in app.allowed_cells}
        self.scheduler_managed_cells.update(association[uid] for uid in self.scheduler_controlled_uids
                                            if uid in association and association[uid] in allowed)
        for action in received:
            if getattr(action,'_scheduler_bundle_candidate',False):continue
            cid=action.target_cell_id if action.target_cell_id is not None else getattr(manager,'_mitigation_default_cell_id',None)
            overlaps=cid in self.scheduler_managed_cells or (action.apply_scope=='cluster' and self.scheduler_managed_cells)
            if overlaps and any(getattr(action,f,None) not in (None,{}) for f in SCHEDULER_RESERVED_FIELDS):
                raise ValueError('uniform_random scheduler bundle cannot mix standalone scheduler tuning')

    def _random_scheduler(self,manager,association,executed,winners,scopes,slot):
        arbiter=manager.scheduler_conflict_arbiter
        self.random_scheduler_outputs=[]
        for cid in sorted(self.scheduler_managed_cells):
            members=sorted(uid for uid,cell in association.items() if cell==cid)
            cfg=self.baseline.get(cid,{})
            if not cfg or any(uid not in self.ue_service_baseline['ue_scheduler_weights'] for uid in members):continue
            candidates=[deepcopy(r) for (_,cell),r in self.scheduler_requests.items() if cell==cid and arbiter.request_valid(r,slot)]
            baseline=dict(scheduler_mode=cfg.get('scheduler_mode','round_robin'),
                ue_scheduler_weights={uid:deepcopy(self.ue_service_baseline['ue_scheduler_weights'][uid]) for uid in members})
            tuning={}
            if baseline['scheduler_mode']=='latency_aware':tuning['latency_bias']=cfg.get('latency_bias') if cfg.get('latency_bias') is not None else 1.
            if baseline['scheduler_mode']=='slice_aware':tuning['slice_weight_adjustment']=deepcopy(cfg.get('slice_weight_adjustment',{}))
            record,event=arbiter.choose(cid,slot,candidates,baseline,members,tuning)
            if event:manager.rule_decisions=getattr(manager,'rule_decisions',[])+[event]
            if record is None:continue
            # No cached field or separately owned tuning action may accompany
            # the atomic selection, including while it waits for E2 execution.
            scope='cell:'+cid;scopes[scope]=('cell',cid);fields=winners.setdefault(scope,{})
            for field in SCHEDULER_RESERVED_FIELDS:
                fields.pop(field,None);self.owned.pop((cid,field),None)
                cached=manager.current_action_by_scope.get(scope)
                if cached:setattr(cached,field,{} if field=='slice_weight_adjustment' else None)
            owner=record['winner']
            if owner=='rule_withdrawal':
                actual=getattr(manager,'_scheduler_mitigation_executed',{}).get(cid,{})
                actual_mode=actual.get('value',{}).get('scheduler_mode',cfg.get('scheduler_mode'))
                uid_values=getattr(manager,'_scheduler_mitigation_uid_values',{})
                actual_weights={uid:uid_values.get(uid,executed.get(('ue_scheduler_weights',uid))) for uid in members}
                actual_tuning=actual.get('framework_tuning',tuning)
                if actual_mode==baseline['scheduler_mode'] and actual_weights==baseline['ue_scheduler_weights'] and actual_tuning==tuning:
                    self.scheduler_managed_cells.discard(cid)
                    self.scheduler_controlled_uids.difference_update(members)
                    getattr(manager,'_scheduler_mitigation_executed',{}).pop(cid,None)
                    for uid in members:getattr(manager,'_scheduler_mitigation_uid_values',{}).pop(uid,None)
                    arbiter.invalidate(cid)
                    manager.rule_decisions.append(dict(reason='direct_conflict_scheduler_baseline_confirmed',slot_id=slot,
                        cell_id=cid,field='scheduler_bundle',value=baseline))
                    continue
            else:
                self.scheduler_controlled_uids.update(members)
                for field in SCHEDULER_BUNDLE_FIELDS:self.owned[cid,field]=owner
            for field,value in record['selected_value'].items():fields[field]=(0,deepcopy(value),owner)
            self.random_scheduler_outputs.append(deepcopy(record))

    def reconcile(self,manager,observation,winners,scopes,received,dropped=None,slot_id=None):
        apps={i.xapp.name:i.xapp for i in manager.e2_interfaces if getattr(i.xapp,'rule_based_v1',False)}
        paper=getattr(observation,'paper_report',None) or {}
        association,service_executed=self._service_context(manager,observation,paper)
        for cid,item in paper.get('values',{}).get('cell',{}).items():
            if item.get('executed_config') and cid not in self.baseline:self.baseline[cid]=deepcopy(item['executed_config'])
        enabled={name for name in apps if manager._is_xapp_enabled(name)}
        random_arbiter=getattr(manager,'direct_conflict_arbiter',None)
        now=None
        if random_arbiter is not None or getattr(manager,'scheduler_conflict_arbiter',None) is not None:
            clock=getattr(observation,'slot_id',None) if slot_id is None else slot_id
            if clock is None:
                raise ValueError('Timed conflict arbitration requires an actual slot_id')
            now=int(clock)
        self._prepare_scheduler_requests(manager,apps,enabled,association,received,now)
        for name,app in apps.items():
            if self.was_enabled.get(name,False) and name not in enabled and hasattr(app,'reset'):app.reset()
        self.was_enabled={name:name in enabled for name in apps}
        for key in list(self.requests):
            if key[0] not in enabled or (getattr(apps.get(key[0]),'library_instance',False) and not manager.instance_selection.enabled(key[0],key[1])):del self.requests[key]
            elif key[2] in self.UE_SERVICE_FIELDS:
                priority,values,owner=self.requests[key]
                values=self._service_filter(apps,association,key[0],key[1],key[2],values,dropped)
                if values:self.requests[key]=(priority,values,owner)
                else:del self.requests[key]
        for action in received:
            name=action.source_name
            if random_arbiter is not None and action.ue_offload_fraction is not None and name not in apps:
                raise ValueError('uniform_random requires lease-backed UE offload xApps')
            if name not in enabled or action.target_cell_id is None:continue
            if getattr(apps[name],'library_instance',False):manager.instance_selection.consume_hold(action,apps[name])
            for field in apps[name].controlled_fields:
                if (name,action.target_cell_id) in self.scheduler_requests and field in SCHEDULER_RESERVED_FIELDS:continue
                family={'slice_weight_adjustment':'slice_aware','latency_bias':'latency_aware'}.get(field)
                if family and action.scheduler_mode is not None and action.scheduler_mode!=family:
                    self.requests.pop((name,action.target_cell_id,field),None)
                    continue
                value=getattr(action,field,None)
                if field in self.UE_SERVICE_FIELDS and value is not None:
                    value=self._service_filter(apps,association,name,action.target_cell_id,field,value,dropped)
                    if not value:
                        self.requests.pop((name,action.target_cell_id,field),None)
                        continue
                if getattr(apps[name],'library_instance',False) and field=='slice_weight_adjustment' and not value:continue
                if value is not None:
                    key=(name,action.target_cell_id,field)
                    self.requests[key]=(apps[name].priority,deepcopy(value),name)
                    if random_arbiter is not None and field=='ue_offload_fraction':
                        self.request_metadata[key]=dict(request_id=getattr(action.metadata,'decision_id',None),
                            effective_from_slot=action.effective_from_slot,effective_until_slot=action.effective_until_slot,
                            provenance=deepcopy(apps[name].identity(action.target_cell_id)) if getattr(apps[name],'library_instance',False) else None)
                        manager.rule_decisions=getattr(manager,'rule_decisions',[])+[
                            dict(reason='direct_conflict_request_received',slot_id=now,source=name,field=field,
                                ue_id=uid,cell_id=action.target_cell_id,requested_value=deepcopy(item),
                                **deepcopy(self.request_metadata[key])) for uid,item in value.items()]
        if random_arbiter is not None:
            self.request_metadata={key:value for key,value in self.request_metadata.items() if key in self.requests}
            from .random_arbiter import overlaps_unequally
            held={}
            for (name,cid,field),(_,value,_) in self.requests.items():
                if field=='ue_offload_fraction':continue
                if cid in self.scheduler_managed_cells and field in SCHEDULER_RESERVED_FIELDS:
                    raise ValueError('uniform_random scheduler bundle cannot mix standalone scheduler tuning')
                for other_name,other_value in held.setdefault((cid,field),[]):
                    if other_name!=name and overlaps_unequally(value,other_value):
                        raise ValueError('uniform_random does not support direct conflict on '+field)
                held[cid,field].append((name,value))
        # Rebuild these two field winners from sanitized held requests, rather
        # than retaining the raw per-cell map collected before this reconcile.
        for fields in winners.values():
            for field in self.UE_SERVICE_FIELDS:
                if field in fields and fields[field][2] in apps:del fields[field]
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
            if random_arbiter is not None and field=='ue_offload_fraction':continue
            if cid in self.scheduler_managed_cells and field in SCHEDULER_RESERVED_FIELDS:continue
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
            if field in self.UE_SERVICE_FIELDS:
                del self.owned[cid,field]
                continue
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
        self._service_restore(manager,apps,enabled,association,service_executed,winners,scopes,dropped)
        if random_arbiter is not None:
            self._random_offload(manager,apps,enabled,association,service_executed,winners,scopes,now)
        if getattr(manager,'scheduler_conflict_arbiter',None) is not None:
            self._random_scheduler(manager,association,service_executed,winners,scopes,now)
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
