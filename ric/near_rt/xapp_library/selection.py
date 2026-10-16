"""Atomic per-cell/category activation; no selection decision algorithm."""
from copy import deepcopy


class InstanceSelection:
    def __init__(self,manager):
        self.manager=manager;self.bindings=None;self.generation={};self.receipts=[];self.catalog={};self.version=0;self.holds={}

    def apps(self):
        return {i.xapp.name:i.xapp for i in self.manager.e2_interfaces if getattr(i.xapp,'library_instance',False)}

    @property
    def mode(self):
        modes={app.selection_mode for app in self.apps().values()}
        if len(modes)>1:raise ValueError("Inconsistent library selection modes")
        return next(iter(modes),"exclusive")

    def validate(self,bindings,parameters=None,legacy_names=(),active_sets=None):
        if self.mode=="conflict_experiment":return self.validate_sets(bindings,active_sets,parameters,legacy_names)
        if active_sets is not None:raise ValueError("xapp_active_sets requires conflict_experiment mode")
        apps=self.apps();known=set(cid for a in apps.values() for cid in a.allowed_cells)
        if any(name in apps for name in legacy_names):raise ValueError('Library instances require xapp_bindings, not legacy enabled/disabled lists')
        if not isinstance(bindings,dict):raise ValueError('xapp_bindings must be a complete mapping')
        for cid,categories in bindings.items():
            if cid not in known or not isinstance(categories,dict):raise ValueError('Unknown cell or malformed category bindings: '+str(cid))
            for category,ident in categories.items():
                if ident not in apps:raise ValueError('Unknown instance: '+str(ident))
                app=apps[ident]
                if category!=app.category_id or cid not in app.allowed_cells:raise ValueError('Category/cell mismatch: '+str(ident))
                if not app.contract.available:raise ValueError('Unavailable backend capability: '+str(ident))
        for ident,values in (parameters or {}).items():
            if ident not in apps:raise ValueError('Unknown instance parameter target: '+str(ident))
            apps[ident].validate_parameters(values)
        return deepcopy(bindings)

    def default_bindings(self):
        apps=self.apps();explicit=[a.initial_bindings for a in apps.values() if a.initial_bindings is not None]
        if explicit:
            if any(x!=explicit[0] for x in explicit):raise ValueError('Inconsistent initial instance bindings')
            return self.validate(explicit[0])
        result={}
        for name,app in apps.items():
            for cid in app.allowed_cells:
                if app.category_id in result.setdefault(cid,{}):raise ValueError('Overlapping same-category instances require explicit bindings')
                result[cid][app.category_id]=name
        return self.validate(result)

    def sync(self,slot,observation):
        apps=self.apps()
        if not apps:return
        snapshot=self.manager.policy_store.get_snapshot()
        if self.mode=='conflict_experiment':return self.sync_sets(slot,snapshot)
        requested=snapshot.xapp_bindings
        new=self.default_bindings() if requested is None else self.validate(requested)
        old=self.bindings or {};changes=[]
        if old!=new:self.version+=1
        for cid in sorted(set(old)|set(new)):
            for category in sorted(set(old.get(cid,{}))|set(new.get(cid,{}))):
                before=old.get(cid,{}).get(category);after=new.get(cid,{}).get(category)
                if before==after:continue
                key=(cid,category);self.generation[key]=self.version
                changes.append(dict(slot_id=slot,kind='xapp_selection',cell_id=cid,category_id=category,
                    previous_instance_id=before,instance_id=after,selection_version=self.generation[key],status='applied'))
                leases=getattr(self.manager,'rule_leases',None)
                if leases:
                    for request in list(leases.requests):
                        if request[0]==before and request[1]==cid:del leases.requests[request]
                    # Switching holds executed state until the new instance has
                    # valid observations. Explicit disable uses normal withdrawal.
                    if after:
                        cached=self.manager.current_action_by_scope.get('cell:'+cid)
                        for (cell,field),owner in list(leases.owned.items()):
                            if cell==cid and owner==before:
                                self.holds.setdefault(key,{})[field]=owner
                                if cached:setattr(cached,field,{} if field=='slice_weight_adjustment' else None)
                                del leases.owned[cell,field]
                    else:
                        for field,owner in self.holds.pop(key,{}).items():leases.owned.setdefault((cid,field),owner)
        self.bindings=deepcopy(new);self.receipts.extend(changes)
        for name,app in apps.items():
            cells={cid for cid,categories in new.items() if categories.get(app.category_id)==name}
            app.select(cells,{cid:self.generation.get((cid,app.category_id),0) for cid in cells})
        self.manager.policy_store.snapshot.xapp_selection_version=self.version

    def valid(self,provenance):
        cid=provenance['cell_id'];category=provenance['category_id']
        if self.mode=='conflict_experiment':
            ident=provenance['instance_id']
            return self.bindings is not None and ident in self.bindings.get(cid,{}).get(category,[]) and self.generation.get((cid,category,ident))==provenance['selection_version']
        return self.bindings is not None and self.bindings.get(cid,{}).get(category)==provenance['instance_id'] and self.generation.get((cid,category))==provenance['selection_version']

    def consume_hold(self,action,app):
        key=(action.target_cell_id,app.category_id,app.name) if self.mode=='conflict_experiment' else (action.target_cell_id,app.category_id);held=self.holds.get(key,{})
        for field in list(held):
            family={'slice_weight_adjustment':'slice_aware','latency_bias':'latency_aware'}.get(field)
            if self.manager._action_field_has_value(field,getattr(action,field,None)) or (family and action.scheduler_mode is not None and action.scheduler_mode!=family):held.pop(field)
        if not held:self.holds.pop(key,None)

    def enabled(self,name,cid=None):
        app=self.apps().get(name)
        return bool(app and (app.active_cells if cid is None else cid in app.active_cells))

    def validate_sets(self,bindings,active_sets,parameters=None,legacy_names=()):
        if bindings is not None:raise ValueError('conflict_experiment uses xapp_active_sets, not xapp_bindings')
        apps=self.apps();known={c for a in apps.values() for c in a.allowed_cells}
        if any(n in apps for n in legacy_names):raise ValueError('Library instances require xapp_active_sets')
        if not isinstance(active_sets,dict):raise ValueError('xapp_active_sets must be a complete mapping')
        result={}
        for cid,categories in active_sets.items():
            if cid not in known or not isinstance(categories,dict):raise ValueError('Unknown cell or malformed active sets')
            result[cid]={}
            for category,ids in categories.items():
                from .catalog import category_catalog
                if category not in category_catalog():raise ValueError('Unknown category in active sets')
                if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(ids)!=len(set(ids)):raise ValueError('Unique instance list required')
                goals=set()
                for ident in ids:
                    app=apps.get(ident)
                    if app is None or app.category_id!=category or cid not in app.allowed_cells or not app.contract.available:raise ValueError('Invalid instance category, scope or capability')
                    objective=app.spec['objective_id']
                    if objective in goals:raise ValueError('Duplicate objective in same cell/category')
                    goals.add(objective)
                result[cid][category]=sorted(ids)
        for ident,values in (parameters or {}).items():
            if ident not in apps:raise ValueError('Unknown parameter instance')
            apps[ident].validate_parameters(values)
        return result

    def sync_sets(self,slot,snapshot):
        apps=self.apps();requested=snapshot.xapp_active_sets
        if requested is None:
            presets=[a.initial_active_sets for a in apps.values() if a.initial_active_sets is not None]
            if presets and any(p!=presets[0] for p in presets):raise ValueError('Inconsistent initial active sets')
            requested=presets[0] if presets else {}
        new=self.validate(None,active_sets=requested);old=self.bindings or {}
        if old!=new:self.version+=1
        for cid in sorted(set(old)|set(new)):
            for category in sorted(set(old.get(cid,{}))|set(new.get(cid,{}))):
                before=set(old.get(cid,{}).get(category,[]));after=set(new.get(cid,{}).get(category,[]))
                leases=getattr(self.manager,'rule_leases',None)
                for ident in sorted(before ^ after):
                    key=(cid,category,ident);self.generation[key]=self.version
                    self.receipts.append(dict(slot_id=slot,kind='xapp_selection',cell_id=cid,category_id=category,
                        instance_id=ident,status='applied',operation='added' if ident in after else 'removed',selection_version=self.version))
                    if ident in before and leases:
                        for req in list(leases.requests):
                            if req[0]==ident and req[1]==cid:del leases.requests[req]
                        # Retained instances keep their requests/history; reconcile restores
                        # baseline only if no legal request remains for the field.
                        cached=self.manager.current_action_by_scope.get('cell:'+cid)
                        for (cell,field),owner in list(leases.owned.items()):
                            if cell==cid and owner==ident and cached:setattr(cached,field,{} if field=='slice_weight_adjustment' else None)
        self.bindings=deepcopy(new)
        for ident,app in apps.items():
            cells={cid for cid,groups in new.items() if ident in groups.get(app.category_id,[])}
            app.select(cells,{cid:self.generation[(cid,app.category_id,ident)] for cid in cells})
        self.manager.policy_store.snapshot.xapp_selection_version=self.version
