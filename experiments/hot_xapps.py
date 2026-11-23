"""Warm, frozen multi-xApp activation through the ordinary RIC lease lifecycle.

Instances/models are initialized once. Selection changes neither reset the
network nor directly write RAN parameters; versioned admission, arbitration,
delayed execution and withdrawal remain responsible for those parameters.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import uuid

from experiments.live_xapp_test import LiveXAppSession, atomic_json

ROOT=Path(__file__).resolve().parents[1]
RELEASE='20261117'
KINDS=dict(x01='energy',x02='ap',x08='throughput',x12_mec='latency',x12_urllc='urllc',x13='budget')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(parser):
    parser.add_argument('--live-xapps',action='store_true',help='Warm-load trained Chongyu models and independently toggle multiple xApps during simulation')
    parser.add_argument('--xapps-start',default='',help='Comma-separated initially enabled IDs, e.g. x01,x13; default all OFF; all enables all six')
    parser.add_argument('--xapp-model-dir',type=Path,help='Versioned six-model installation under outputs/; default installs the bundled release')
    parser.add_argument('--xapp-toggle-timeline',type=Path,help='Optional simulation-time activation timeline JSON; requires --live-xapps')


def aliases(release):
    norm=lambda s:s.strip().lower().replace('-','').replace('_','')
    result={}
    for key,entry in release['models'].items():
        for value in (key,entry['id'],entry['implementation_id']):result[norm(value)]=entry['implementation_id']
    return result


def resolve_ids(values,lookup):
    if isinstance(values,str):values=[v.strip()for v in values.split(',')if v.strip()]
    if not isinstance(values,list)or any(not isinstance(v,str)for v in values):raise ValueError('xApps must be a list of registered IDs')
    if values==['all']:return sorted(set(lookup.values()))
    result=[]
    for value in values:
        key=value.strip().lower().replace('-','').replace('_','')
        if key not in lookup:raise ValueError('Unknown trained xApp: '+value)
        result.append(lookup[key])
    if len(result)!=len(set(result)):raise ValueError('Duplicate xApp IDs')
    return sorted(result)


def load_timeline(path,lookup,slot_ms,total_slots):
    if path is None:return []
    entries=json.loads(Path(path).read_text())
    if not isinstance(entries,list):raise ValueError('xApp timeline must be a list')
    result=[];previous=-1
    for i,item in enumerate(entries):
        if (not isinstance(item,dict)or set(item)-{'at_ms','operation','apps','cell'} or
                not {'at_ms','operation'}<=set(item)):
            raise ValueError('Timeline entries require at_ms, operation, and optional apps/cell')
        at=item['at_ms'];operation=item['operation']
        if type(at)not in(int,float)or not math.isfinite(at)or at<0:
            raise ValueError('Timeline time must be finite and nonnegative')
        slot=round(at/slot_ms)
        if not math.isclose(slot*slot_ms,at,abs_tol=1e-8)or not previous<=slot<total_slots:
            raise ValueError('Timeline times must be ordered, align to slots and precede run end')
        if operation not in ('enable','disable','set','disable_all'):raise ValueError('Unknown timeline operation')
        ids=resolve_ids(item.get('apps',[]),lookup)
        if (operation in ('enable','disable')and not ids)or(operation=='disable_all'and ids):
            raise ValueError('enable/disable require apps; disable_all takes no apps')
        cell=item.get('cell','*')
        if not isinstance(cell,str):raise ValueError('Timeline cell must be a string')
        result.append(dict(run_id=None,request_id='timeline:'+str(i),operation=operation,
            apps=ids,cell=cell,scheduled_slot=slot,source='simulation_timeline'))
        previous=slot
    return result


def configure(configs,manifest,args,output,root=ROOT):
    root=Path(root).resolve();output=Path(output).resolve()
    if args.xapp_pool!='chongyu' or getattr(args,'phase','integration')!='integration' or getattr(args,'control_mode','baseline')!='baseline':
        raise ValueError('--live-xapps requires a Chongyu integration baseline')
    if getattr(args,'live_xapp_test',False):raise ValueError('Choose --live-xapps or the legacy --live-xapp-test')
    if not output.is_relative_to(root/'outputs'):raise ValueError('Live xApp output must remain under outputs/')
    if {u['ue_id']for u in manifest['ues']}!={'ue_0','ue_1','ue_2','ue_4','ue_6','ue_8'}:
        raise ValueError('The released live controllers require the current six-UE scenario')
    spec=importlib.util.spec_from_file_location('oran_model_installer',root/'scripts/orin/install-trained-xapps.py')
    installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    target=getattr(args,'xapp_model_dir',None)
    model_dir=(Path(target) if target else root/'outputs/trained_xapps'/RELEASE)
    model_dir=(model_dir if model_dir.is_absolute()else root/model_dir).resolve()
    if not model_dir.is_relative_to(root/'outputs'):raise ValueError('Model directory must remain under outputs/')
    if target is None and not model_dir.exists():installer.install(root/'artifacts/xapps/chongyu'/RELEASE,model_dir,root)
    release=installer.load_release(model_dir)
    action_set=getattr(args,'energy_xapp_action_set_version','legacy9')
    if action_set not in ('legacy9','extended24','balanced42'):
        raise ValueError('Unknown energy action set')
    released_action_set=release['models']['x01'].get('runtime_parameters',{}).get('action_set_version','legacy9')
    if action_set!=released_action_set:
        raise ValueError('The selected model release and energy action set differ')
    if action_set!='legacy9':
        from smo.model_release import energy_catalogue,catalogue_sha256
        entry=release['models']['x01']
        if (entry.get('action_catalogue')!=energy_catalogue(action_set)
                or entry.get('action_catalogue_sha256')!=catalogue_sha256(action_set)
                or entry.get('implementation_variant')!=action_set):
            raise ValueError('The model release must pin its exact reviewed action variant')
    lookup=aliases(release);initial=resolve_ids(getattr(args,'xapps_start',''),lookup)
    for key,kind in KINDS.items():
        mode=getattr(args,kind+'_xapp_mode','off')
        if mode not in ('off','frozen'):raise ValueError('Warm model switching uses frozen models; collect online training in a separate run')
        if mode=='frozen':initial.append(release['models'][key]['implementation_id'])
        explicit=getattr(args,kind+'_xapp_model',None)
        if explicit is not None and Path(explicit).resolve()!=(model_dir/release['models'][key]['path']).resolve():
            raise ValueError('Use one pinned release/model directory for live xApps')
    initial=sorted(set(initial));cells=[c['cell_id']for c in manifest['cells']]
    timeline=load_timeline(getattr(args,'xapp_toggle_timeline',None),lookup,manifest['slot_duration_ms'],manifest['total_slots'])
    if any(e['cell']!='*'and e['cell']not in cells for e in timeline):raise ValueError('Unknown timeline cell')
    services=configs['phy']['rt_sys'].get('ue_services',{})
    if services.get('enabled')and services.get('mec',{}).get('enabled'):services['task_cohort_ms']=100
    mitigation=getattr(args,'direct_conflict_mitigation',None)or'uniform_random'
    window=getattr(args,'mitigation_window_ms',100)
    seed=getattr(args,'mitigation_seed',None)
    seed=args.seed if seed is None else seed
    if mitigation not in ('priority','uniform_random')or type(seed)is not int or seed<0 or type(window)not in(int,float)or not math.isfinite(window)or window<=0:
        raise ValueError('Invalid live conflict mitigation configuration')
    apps=[];pins={}
    for key,entry in release['models'].items():
        kind=KINDS[key];model=(model_dir/entry['path']).resolve()
        params=dict(mode='frozen',model_path=str(model),log_path=str(output/(kind+'_rf_events.jsonl')),
            window_ms=getattr(args,kind+'_xapp_window_ms',100),seed=args.seed)
        if key=='x01':
            params['deploy_drx']=getattr(args,'energy_xapp_deploy_drx',True)
            if action_set!='legacy9':params['action_set_version']=action_set
        if key=='x08':params.update(log_path=str(output/'throughput_mlp_events.jsonl'),policy=getattr(args,'throughput_xapp_policy','throughput'))
        if key=='x12_urllc':params.update(policy_version='qos_v2',balanced_start_windows=0)
        if key=='x12_mec':params['complete_scheduler_bundle']=True
        if key=='x13':
            params.update(log_path=str(output/'budget_qgbdt_events.jsonl'),budget_gflops=getattr(args,'budget_xapp_gflops',.75),
                task_profile={k:services.get('mec',{}).get('tasks_by_ue',{}).get('ue_4',{}).get(k,v)
                              for k,v in entry['runtime_parameters']['task_profile'].items()})
        ident=entry['implementation_id']
        apps.append(dict(type='pool_xapp',pool='chongyu',implementation_id=ident,instance_id=ident,
            cell_ids=cells,slot_duration_ms=manifest['slot_duration_ms'],parameters=params,
            selection_mode='conflict_experiment',initial_active_sets={}))
        if key=='x01' and action_set!='legacy9':apps[-1]['implementation_variant']=action_set
        pins[ident]=dict(path=str(model),sha256=entry['sha256'],release=release['release'],mode='frozen',label=entry['id'])
        if key=='x01':
            pins[ident].update(action_set_version=action_set,implementation_variant=action_set)
    configs['ric'].update(xapp_pool='chongyu',rapps=[],run_rapps_on_start=False,freeze_learning=True,
        xapp_interval_slots=max(1,round(20/manifest['slot_duration_ms'])),xapps=apps,
        direct_conflict_mitigation=mitigation,mitigation_seed=seed,mitigation_window_ms=window)
    configs['phy']['rt_sys'].setdefault('paper_interfaces',{}).update(enabled=True,rule_observations=True)
    timeline_path=getattr(args,'xapp_toggle_timeline',None)
    args._hot_xapp_spec=dict(initial=initial,aliases=lookup,models=pins,timeline=timeline,
        timeline_source=dict(path=str(Path(timeline_path).resolve()),sha256=digest(timeline_path))if timeline_path else None)
    args.xapp_model_dir=model_dir
    return 'live_multi_xapps'


class HotXAppSession(LiveXAppSession):
    def __init__(self,simulator,args,output,state_path=None):
        self.hot_spec=deepcopy(args._hot_xapp_spec);self.timeline=deepcopy(self.hot_spec['timeline'])
        self.multi=True;self.command_sequence=0
        super().__init__(simulator,args,output,state_path)
        checkpoint_dir=self.output/'xapp_model_checkpoints';checkpoint_dir.mkdir()
        for ident,entry in self.hot_spec['models'].items():
            if digest(entry['path'])!=entry['sha256']:raise ValueError('Released model changed before warm start')
            target=checkpoint_dir/(ident+'.json');shutil.copy2(entry['path'],target)
            entry['checkpoint']=str(target.relative_to(self.output))
        source=self.hot_spec.get('timeline_source')
        if source:
            if digest(source['path'])!=source['sha256']:raise ValueError('Activation timeline changed after configuration')
            shutil.copy2(source['path'],self.output/'xapp_switch_timeline.json')
        for e in self.timeline:e['run_id']=self.run_id
        self.apply_command('set',self.hot_spec['initial'],'*',0)
        atomic_json(self.output/'xapp_hot_start.json',dict(schema='oran.hot-xapps.v1',
            models=self.hot_spec['models'],initial=self.hot_spec['initial'],timeline=self.timeline,timeline_source=source,
            policy='Versioned active sets with normal arbitration and delayed RAN actuation'))

    def _selected(self):
        return {name for groups in (self.manager.instance_selection.bindings or {}).values()
                for ids in groups.values()for name in ids}

    def apply_command(self,operation,apps,cell,slot):
        from models.policy import PolicyUpdate
        if cell!='*'and cell not in self.baseline:raise ValueError('Unknown target cell')
        if operation not in ('enable','disable','set','disable_all'):raise ValueError('Unsupported multi-xApp operation')
        if not isinstance(apps,list)or len(apps)!=len(set(apps))or any(i not in self.apps for i in apps):
            raise ValueError('Unique registered xApp IDs are required')
        if operation in ('enable','disable')and not apps:raise ValueError('Choose at least one xApp')
        if operation=='disable_all'and apps:raise ValueError('disable_all takes no apps')
        if operation in ('enable','set'):
            for name in apps:
                reason=self.unavailable_reason(self.apps[name])
                if reason:raise ValueError(name+': '+reason)
        selection=self.manager.instance_selection;new=deepcopy(selection.bindings or {})
        for cid in sorted(self.baseline)if cell=='*'else[cell]:
            groups=new.setdefault(cid,{})
            if operation in ('set','disable_all'):groups.clear()
            for name in apps:
                category=self.apps[name].category_id;ids=set(groups.get(category,[]))
                if operation in ('enable','set'):ids.add(name)
                else:ids.discard(name)
                if ids:groups[category]=sorted(ids)
                else:groups.pop(category,None)
            if not groups:new.pop(cid,None)
        if new==(selection.bindings or {})and self.segments:return 'unchanged'
        # Validate the complete new set before admission: atomic all-or-nothing.
        for ident in ('throughput_scheduler_mlp','latency_urllc_rf'):
            app=self.apps.get(ident)
            if app is None:continue
            chosen={cid for cid,groups in new.items()if ident in groups.get(app.category_id,[])}
            if chosen and chosen!=set(app.allowed_cells):
                raise ValueError(ident+' requires the whole network; choose All cells')
        selection.validate(None,active_sets=new)
        accepted=self.manager.policy_store.apply_update(PolicyUpdate(source_name='live_xapp_ui',xapp_active_sets=new),slot_id=slot)
        if accepted is False:raise ValueError('Selection policy was rejected')
        selection.sync(slot,None)
        active=sorted(self._selected())
        if apps:self.app=self.apps[apps[-1]]
        self.segments.append(dict(label='ON / '+', '.join(self.hot_spec['models'][n]['label']for n in active)if active else'OFF / baseline',
            app_id=None,app_ids=active,active_sets=deepcopy(new),enabled=bool(active),cell=cell,
            start_slot=slot,first_counted_slot=slot+self.settle_slots+1,
            slots=0,skipped_slots=0,delivered_bytes=0,queue_sum=0,hol_sum=0.,decodes=0,actual_decodes=0,
            nacks=0,completed_packets=0,completion_delay_sum_ms=0.))
        return 'selection_admitted; awaiting normal control tick and delayed execution'

    def change(self,enabled,cell,slot,app_id=None):
        if app_id is None and not enabled:return self.apply_command('disable_all',[],cell,slot)
        app_id=app_id or(self.app.name if self.app else None)
        return self.apply_command('enable'if enabled else'disable',[app_id],cell,slot)

    def activate(self,request):
        return self.change(True,request['cell'],self.simulator.state.slot_id,request['app'])

    def record_receipt(self,request,status,reason):
        event=dict(request_id=request.get('request_id'),operation=request.get('operation'),
            app=request.get('app'),apps=request.get('apps'),cell=request.get('cell'),
            scheduled_slot=request.get('scheduled_slot'),source=request.get('source','live_command'),
            received_at_slot=self.simulator.state.slot_id,status=status,reason=reason,
            active_sets=deepcopy(self.manager.instance_selection.bindings or {}))
        self.receipts.append(event)
        with self.events.open('a')as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')

    def _consume(self,request):
        status='rejected'
        try:
            if not isinstance(request,dict):raise ValueError('Malformed command')
            if request.get('run_id')!=self.run_id:raise ValueError('Stale run')
            legacy={'run_id','request_id','operation','app','cell'}
            multi={'run_id','request_id','operation','apps','cell'}
            scheduled=multi|{'scheduled_slot','source'}
            if set(request)not in(legacy,multi,scheduled):raise ValueError('Malformed command')
            if not isinstance(request['request_id'],str)or not request['request_id']:raise ValueError('Missing request ID')
            op=request['operation']
            if op=='activate':op='enable'
            values=request.get('apps',[request['app']]if'app'in request else[])
            ids=resolve_ids(values,self.hot_spec['aliases'])
            reason=self.apply_command(op,ids,request['cell'],self.simulator.state.slot_id)
            status='accepted'
        except(ValueError,TypeError,KeyError)as e:reason=str(e)
        self.record_receipt(request if isinstance(request,dict)else{},status,reason)

    def before_step(self):
        while self.timeline and self.timeline[0]['scheduled_slot']<=self.simulator.state.slot_id:
            self._consume(self.timeline.pop(0))
        if self.command_dir:
            for path in sorted(self.command_dir.glob('*.json'))[:32]:
                try:
                    if path.stat().st_size>8192:raise ValueError('Command too large')
                    request=json.loads(path.read_text())
                except(OSError,ValueError)as e:self.record_receipt({},'rejected',str(e))
                else:self._consume(request)
                path.unlink(missing_ok=True)

    def snapshot(self):
        result=super().snapshot()
        result.update(multi_activation=True,active_app_ids=sorted(self._selected()),
            active_sets=deepcopy(self.manager.instance_selection.bindings or {}),
            model_artifacts=deepcopy(self.hot_spec['models']),timeline_remaining=len(self.timeline))
        result['selected_cells']=sorted(cid for cid,groups in result['active_sets'].items()if any(groups.values()))
        result['last_decisions']=[deepcopy(row)for app in self.apps.values()for row in app.last_decisions]
        return result

    def verify_models(self):
        for entry in self.hot_spec['models'].values():
            if digest(entry['path'])!=entry['sha256']:raise RuntimeError('Frozen xApp model artifact changed during live run')
            if digest(self.output/entry['checkpoint'])!=entry['sha256']:raise RuntimeError('Run-local model checkpoint changed')


def submit_many(state_path,data,operation,apps=(),cell='*'):
    x=data['xapp_test']
    if not x.get('multi_activation'):raise ValueError('This run does not support multiple live xApps')
    if data.get('stage')!='simulating':raise ValueError('This run has finished; start a new live run')
    if operation not in ('enable','disable','set','disable_all'):raise ValueError('Unknown operation')
    if not isinstance(apps,(list,tuple))or any(not isinstance(a,str)for a in apps):raise ValueError('Choose registered xApp IDs')
    directory=Path(state_path).with_suffix('.commands')
    if not directory.is_dir():raise ValueError('No live command channel')
    request_id=uuid.uuid4().hex
    atomic_json(directory/f'{time.time_ns():020d}_{request_id}.json',dict(run_id=x['run_id'],
        request_id=request_id,operation=operation,apps=list(apps),cell=cell))
    return request_id
