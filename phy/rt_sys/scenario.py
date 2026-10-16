"""Build deterministic deployment manifests and legacy simulator input configs."""
from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np
from core.config_loader import load_all_configs, load_yaml_file
from phy.nr_reference import lookup_nr_max_prbs
from scenarios.scenario_registry import deep_merge_dict

ROOT=Path(__file__).resolve().parents[2]


def build_case(scene='simple_street_canyon',num_cells=2,num_ues=4,seed=1,total_slots=40,
               path_update_ms=None,device='cuda',config_path=None,*,deployment_id=None,traffic_load=1,initial_queue_bytes=None,ue_speed_mps=None):
    base=json.loads((Path(config_path) if config_path else ROOT/'configs/rt_sys/default.json').read_text())
    catalog=json.loads((ROOT/'configs/rt_sys/scene_catalog.json').read_text())
    if scene not in catalog or not 1<=num_cells<=len(catalog[scene]['cells']) or num_ues<1 or total_slots<1:
        raise ValueError('Invalid scene/cell/UE/slot configuration')
    initial_queue_bytes=int(base.get('initial_queue_bytes',0) if initial_queue_bytes is None else initial_queue_bytes)
    if initial_queue_bytes<0:raise ValueError('Initial queue must be nonnegative')
    radio=deepcopy(base['radio']); dt=radio['slot_duration_ms']; duration=dt*total_slots/1000
    if deployment_id is not None and deployment_id not in (0,1,2):raise ValueError('Deployment ID must be 0, 1, or 2')
    if not np.isfinite(traffic_load) or traffic_load<=0:raise ValueError('Positive finite traffic load required')
    nprb,_=lookup_nr_max_prbs(bandwidth_hz=radio['bandwidth_hz'],subcarrier_spacing_hz=radio['subcarrier_spacing_hz'],carrier_frequency_hz=radio['carrier_frequency_hz'])
    fft=2**math.ceil(math.log2(math.ceil(nprb*12*1.25)))
    ns=14*max(1,round(dt/(1000*15/radio['subcarrier_spacing_hz'])))
    ues=[]
    # Phase offsets select distinct locations on the same validated road segment.
    for i in range(num_ues):
        route=np.asarray(catalog[scene]['routes'][i%len(catalog[scene]['routes'])],float)
        phase=((i//len(catalog[scene]['routes']))+.5)/max(1,math.ceil(num_ues/len(catalog[scene]['routes'])))
        fraction=min(.8,phase*.8) if deployment_id is None else .05+.25*deployment_id+.2*phase
        start=route[0]+fraction*(route[1]-route[0])
        direction=route[1]-start
        distance=float(np.linalg.norm(direction))
        speed=float(base.get('ue_speed_mps',1.5) if ue_speed_mps is None else ue_speed_mps)
        if speed<0:raise ValueError('UE speed must be nonnegative')
        times=[0.];points=[start.tolist()];target=route[1];now=start.copy()
        while times[-1]<duration-1e-12:
            distance=float(np.linalg.norm(target-now))
            if speed==0:
                times.append(duration);points.append(now.tolist());break
            if distance<1e-9:
                target=route[0] if np.allclose(target,route[1]) else route[1];continue
            remaining=duration-times[-1];step=min(remaining,distance/speed)
            now=now+(target-now)*(speed*step/distance)
            times.append(times[-1]+step);points.append(now.tolist())
            if step<remaining:target=route[0] if np.allclose(target,route[1]) else route[1]
        ues.append({'ue_id':f'ue_{i}','trajectory':{'times_s':times,'positions_m':points}})
    manifest={'scene':scene,'cells':deepcopy(catalog[scene]['cells'][:num_cells]),'ues':ues,
              'rng_seed':seed,'total_slots':total_slots,'path_update_ms':path_update_ms if path_update_ms is not None else base['path_update_ms'],
              'chunk_slots':base['chunk_slots'],'ray_solver':base['ray_solver'],
              'cache_format':base.get('cache_format','dense_v1'),
              'num_symbols':ns,'fft_size':fft,**radio}
    if 'max_geometry_displacement_m' in base:
        manifest['max_geometry_displacement_m']=base['max_geometry_displacement_m']
    if deployment_id is not None:
        manifest['deployment_id']=deployment_id
        for cell in manifest['cells']:cell['orientation_rad'][0]+=.1*deployment_id
    configs=load_all_configs(str(ROOT/'configs'))
    for name in ('sim','ric','ue'):
        override=ROOT/'configs'/f'{name}_intent_orchestration.yaml'
        if override.exists(): configs[name]=deep_merge_dict(configs[name],load_yaml_file(str(override)))
    configs['sim'].update(base['simulation']); configs['sim']['scheduler_type']='rt_sys_scheduler'; configs['sim'].update(total_slots=total_slots,slot_duration_ms=dt,rng_seed=seed)
    configs['ric']['run_forever']=False
    library=base.get('xapp_library',{})
    if library.get('enabled'):
        if base.get('rule_based_v1',{}).get('enabled'):raise ValueError('Select one library config format')
        known=[c['cell_id'] for c in manifest['cells']]
        bindings=library.get('bindings')
        if bindings is None and 'initial_selection' in library:
            bindings={cid:deepcopy(library['initial_selection']) for cid in known}
        configs['ric'].update(rapps=[],run_rapps_on_start=False,freeze_learning=True,xapps=[])
        for item in library.get('instances',[]):
            scope=item.get('cell_ids',known)
            if not scope or any(cid not in known for cid in scope):raise ValueError('Invalid instance cell scope')
            configs['ric']['xapps'].append(dict(type='xapp_instance',**item,cell_ids=scope,
                initial_bindings=bindings,slot_duration_ms=dt,context_cell_ids=known))
        if library.get('monitor',True):configs['ric']['xapps'].append(dict(type='rule_kpi_monitor'))
    rule=base.get('rule_based_v1',{})
    if rule.get('enabled'):
        from ric.near_rt.rule_based_v1 import CATALOG
        kinds=rule.get('applications',[])
        if len(set(kinds))!=len(kinds) or any(k not in CATALOG for k in kinds):raise ValueError('Invalid/duplicate rule applications')
        configs['ric'].update(rapps=[],xapps=[dict(type='rule_based_v1',kind=k,name='rule_'+k,
            parameters={**rule.get('parameters',{}).get('common',{}),**rule.get('parameters',{}).get(k,{})},
            cell_ids=rule.get('cell_ids',[]),slot_duration_ms=dt) for k in kinds],
            run_rapps_on_start=False,freeze_learning=True)
        if rule.get('monitor',True):configs['ric']['xapps'].append(dict(type='rule_kpi_monitor',name='rule_kpi_monitor',priority=0))
    for rapp in configs['ric'].get('rapps',[]):
        rapp['llm_enabled']=False
        rapp['persistent_memory_enabled']=False
        rapp['intent_text']=base['intent']
    configs['cell']['managed_cell_ids']=[c['cell_id'] for c in manifest['cells']]
    configs['cell']['cell_id']=manifest['cells'][0]['cell_id']
    configs['cell']['neighbor_cell_positions']={c['cell_id']:dict(zip(('x_m','y_m','z_m'),c['position_m'])) for c in manifest['cells']}
    configs['cell']['cell_radio_profiles']={c['cell_id']:{**radio,'tx_power_dbm':30,'gnb_num_tx_ant':radio['tx_rows']*radio['tx_cols']} for c in manifest['cells']}
    originals=deepcopy(configs['ue']['ues'])
    # Interleave slice templates so even the small smoke test includes URLLC.
    slices=list(dict.fromkeys(item['slice_id'] for item in originals))
    grouped=[[item for item in originals if item['slice_id']==sid] for sid in slices]
    originals=[group[k] for k in range(max(map(len,grouped))) for group in grouped if k<len(group)]
    configs['ue']['ues']=[]
    configs['ric']['persistent_memory_enabled']=False
    configs['ric']['memory_store_path']=str(ROOT/'outputs/rt_sys/model_state/xapp_selection.json')
    for i,u in enumerate(ues):
        item=deepcopy(originals[i%len(originals)])
        item.update(ue_id=u['ue_id'],queue_bytes=initial_queue_bytes,serving_beam_id=0,is_indoor=False,serving_cell_id=manifest['cells'][i%num_cells]['cell_id'],
                    num_rx_ant=radio['num_rx_ant'],velocity_mps=0,shadowing_db=0,pathloss_db=100)
        if configs['sim'].get('scheduler_enable_spatial_multiplexing'):
            # One stream per UE leaves the configured two-stream budget for MU.
            item.update(rank=1,ri=1)
        item.update(dict(zip(('x_m','y_m','z_m'),u['trajectory']['positions_m'][0])))
        for key in ('arrival_rate_bps','burst_size_bytes'):
            if key in item:item[key]=item[key]*traffic_load
        for lc in item.get('logical_channels',[]):
            for key in ('arrival_rate_bps','burst_size_bytes'):
                if key in lc:lc[key]=lc[key]*traffic_load
        lcs=item.get('logical_channels',[])
        if lcs:
            weights=[max(0,int(lc.get('queue_bytes',0))) for lc in lcs]
            total=sum(weights);remaining=initial_queue_bytes
            for j,lc in enumerate(lcs):
                amount=remaining if j==len(lcs)-1 else int(initial_queue_bytes*(weights[j]/total if total else 1/len(lcs)))
                lc['queue_bytes']=amount;remaining-=amount
        configs['ue']['ues'].append(item)
    configs['beamforming']['antenna'].update({**radio,'gnb_num_tx_ant':radio['tx_rows']*radio['tx_cols'],
        'default_ue_num_rx_ant':radio['num_rx_ant'],'gnb_panel_layout':f'{radio["tx_rows"]}x{radio["tx_cols"]}'})
    configs['phy']={'phy_type':'sionna_rt_sys','rt_sys':{**base['rt_sys'],'initial_association':'strongest_rt_reference','device':device,'rng_seed':seed,**radio}}
    if rule.get('enabled'):
        configs['phy']['rt_sys'].setdefault('paper_interfaces',{}).update(enabled=True,rule_observations=True,
            beam_reports='beam_management' in rule.get('applications',[]))
    if library.get('enabled'):
        configs['phy']['rt_sys'].setdefault('paper_interfaces',{}).update(enabled=True,rule_observations=True,
            beam_reports=any(i['implementation_id'].startswith('beam_management:') for i in library.get('instances',[])))
    return manifest,configs,base['intent']
