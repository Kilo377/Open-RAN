"""Streaming, self-contained 3D/KPI replay for a completed RT network run."""
from pathlib import Path
import base64
import hashlib
import html
import json
import math
import os
import re
import uuid
import numpy as np
from phy.rt_sys.contracts import has_tb_feedback

ASSETS = Path(__file__).with_name('replay_assets')
ENERGY_SCOPE = ('Radiated RF output integrated in simulation time; excludes PA losses, '
                'RF circuits, baseband, and host electrical consumption.')

def read_json(path):
    return json.loads(Path(path).read_text())

def label_ms(value):
    return f'{value:g} ms' if value < 1000 else f'{value/1000:g} s'

def duration_label(seconds):
    if seconds < 1:return f'{seconds*1000:g} ms'
    if seconds < 120:return f'{seconds:g} s'
    if seconds < 7200:return f'{seconds/60:g} min'
    return f'{seconds/3600:g} h'

def replay_window_slots(total_slots, slot_ms, max_frames=500, window_ms=None):
    if type(max_frames) is not int or max_frames < 1:raise ValueError('Replay max_frames must be a positive integer')
    if not math.isfinite(slot_ms) or slot_ms <= 0:raise ValueError('Positive slot duration required')
    if window_ms is not None:
        if not math.isfinite(window_ms) or window_ms <= 0:raise ValueError('Replay window must be positive')
        return max(1,round(window_ms/slot_ms))
    target=max(slot_ms,total_slots*slot_ms/max_frames)
    power=10**math.floor(math.log10(target))
    nice=next(f*power for f in (1,2,5,10) if f*power>=target*(1-1e-12))
    return max(1,math.ceil(nice/slot_ms-1e-10))

def write_scene_mesh(provider, path):
    """Use the already-loaded scene; never run a new ray-tracing solve."""
    vertices=[];faces=[];offset=0
    for obj in provider.backend.scene.objects.values():
        params=provider.backend.mi.traverse(obj.mi_mesh)
        v=np.asarray(params['vertex_positions'],dtype=np.float32).reshape(-1,3)
        f=np.asarray(params['faces'],dtype=np.int32).reshape(-1,3)
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    if not vertices:raise ValueError('Scene contains no render mesh')
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,vertices=np.concatenate(vertices),faces=np.concatenate(faces))
    os.replace(tmp,path)
    return path

def render_mesh(path,manifest):
    with np.load(path,allow_pickle=False) as mesh:v=mesh['vertices'];faces=mesh['faces']
    if v.ndim!=2 or v.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not len(v) or not len(faces):raise ValueError('Invalid replay scene mesh')
    if not np.isfinite(v).all() or faces.min()<0 or faces.max()>=len(v):raise ValueError('Invalid mesh coordinates or indices')
    points=np.asarray([c['position_m'] for c in manifest['cells']]+[p for u in manifest['ues'] for p in u['trajectory']['positions_m']])
    low=points[:,:2].min(0)-80;high=points[:,:2].max(0)+80
    triangles=v[faces]
    mask=np.all(triangles[:,:,:2].max(1)>=low,axis=1)&np.all(triangles[:,:,:2].min(1)<=high,axis=1)
    if mask.any():faces=faces[mask]
    used,inv=np.unique(faces,return_inverse=True);v=v[used];faces=inv.reshape(-1,3)
    quant=.1;vertices=np.rint(v/quant).astype('<i2' if np.max(abs(v))/quant<32767 else '<i4')
    indices=faces.astype('<u2' if len(v)<65536 else '<u4')
    center=(points[:,:2].min(0)+points[:,:2].max(0))/2
    span=max(float(np.ptp(points[:,:2],axis=0).max()),60.)
    return dict(vertices=base64.b64encode(vertices.tobytes()).decode(),indices=base64.b64encode(indices.tobytes()).decode(),
        vertex_bits=vertices.dtype.itemsize*8,index_bits=indices.dtype.itemsize*8,vertex_count=len(v),face_count=len(faces),quantization_m=quant),dict(
        center=[float(center[0]),12.,float(-center[1])],distance=span*2.8,ground_size=max(span*2.5,300.))

def build_data(folder, *, max_frames=500, window_ms=None, mesh_path=None):
    folder=Path(folder);config=read_json(folder/'config_snapshot.json');summary=read_json(folder/'final_summary.json')
    status=read_json(folder/'run_status.json');initial=read_json(folder/'initial_conditions.json');manifest=config['manifest']
    if not status.get('completed'):raise ValueError('Replay requires a completed simulation; source files were preserved')
    total=int(summary['total_slots']);slot_ms=float(manifest['slot_duration_ms']);dt=slot_ms/1000
    if total!=manifest['total_slots'] or total<1:raise ValueError('Configured and completed slot counts differ')
    window_slots=replay_window_slots(total,slot_ms,max_frames,window_ms)
    ids=[u['ue_id'] for u in manifest['ues']];cell_ids=[c['cell_id'] for c in manifest['cells']]
    ci={cid:i for i,cid in enumerate(cell_ids)};initial_users={u['ue_id']:u for u in initial['ues']}
    counts={uid:[0,0,0,0] for uid in ids};actual=sampled=0;energy=0.;last={}
    frames=[dict(t=0.,slot=0,window_ms=0,g=0.,cg=0.,q=sum(initial_users[u].get('queue_bytes',0) for u in ids),overflow=0,delay=None,nack=None,
        decodes=0,nacks=0,users=[[ci[initial_users[u]['serving_cell_id']],initial_users[u].get('queue_bytes',0),0,0,None,None,None,0,0,None,0,0,False] for u in ids],
        connections=[initial_users[u].get('connection_state','connected') for u in ids],cdelay=None,deadline_misses=0,
        link=[[0,0,0,0] for _ in ids],cells=[[0,0,0] for _ in cell_ids],hol_max_ms=[0 for _ in ids],offered_mbps=0.,rf_power_w=0.,rf_energy_j=0.,xapps=initial.get('xapp_bindings',{}))]
    previous_completed=previous_latency=previous_offered=0;last_mcs={u:None for u in ids}
    first_overflow=None;window_n=0;bindings_recorded=False
    def empty_window():
        return {u:dict(bytes=0,decodes=0,nacks=0,sinr_sum=0.,sinr_n=0,hol=0.) for u in ids},{c:dict(bytes=0,power=0.,occupancy=0.) for c in cell_ids}
    window,cells=empty_window()
    with (folder/'rt_sys_trace.jsonl').open() as stream:
        for expected,line in enumerate(stream,1):
            row=json.loads(line);slot=row['slot_id'];bindings_recorded=bindings_recorded or 'xapp_bindings' in row
            if slot!=expected:raise ValueError(f'Replay trace missing/repeated slot: expected {expected}, got {slot}')
            if {u['ue_id'] for u in row['ues']}!=set(ids):raise ValueError('Trace UE population differs from manifest')
            window_n+=1
            for cid,c in row['cells'].items():
                cells[cid]['power']+=c['average_used_w'];cells[cid]['occupancy']+=c['occupied_re']/c['grid_re']
                energy+=c['average_used_w']*dt
            for u in row['ues']:
                uid=u['ue_id'];last[uid]=u;w=window[uid]
                w['bytes']+=u['delivered_bytes'];cells[u['cell_id']]['bytes']+=u['delivered_bytes'];w['hol']=max(w['hol'],u['packet_hol_delay_ms'])
                if has_tb_feedback(u):
                    c=counts[uid];index=2 if u.get('is_retx') else 0;c[index]+=1;c[index+1]+=int(not u['ack'])
                    w['decodes']+=1;w['nacks']+=int(not u['ack']);actual+=bool(u.get('actual_decode'));sampled+=not bool(u.get('actual_decode'))
                if u.get('scheduled') and u.get('ack_source')!='no_path':
                    value=u.get('raw_sinr_db',u['sinr_db'])
                    if value is not None and math.isfinite(value):w['sinr_sum']+=value;w['sinr_n']+=1
                    last_mcs[uid]=u['mcs']
            if first_overflow is None and sum(u['packet_overflow_bytes'] for u in last.values()):first_overflow=slot*dt
            if slot%window_slots and slot!=total:continue
            seconds=window_n*dt;user_rows=[]
            for uid in ids:
                u=last[uid];w=window[uid]
                user_rows.append([ci[u['cell_id']],u['packet_pending_bytes'],u['packet_hol_delay_ms'],w['bytes']*8/seconds/1e6,
                    w['sinr_sum']/w['sinr_n'] if w['sinr_n'] else None,last_mcs[uid],100*w['nacks']/w['decodes'] if w['decodes'] else None,
                    u['packet_delivered_bytes'],u['packet_overflow_bytes'],u['packet_mean_completion_latency_ms'],u['packet_deadline_missed_packets'],
                    u['packet_late_completed_packets'],u.get('ack_source')=='no_path' and not w['sinr_n']])
            delivered=sum(u['packet_delivered_bytes'] for u in last.values());offered=sum(u['packet_offered_bytes'] for u in last.values())
            completed=sum(u['packet_completed_packets'] for u in last.values());latency=sum(u['packet_completion_latency_ms_sum'] for u in last.values())
            cumulative=[sum(c[i] for c in counts.values()) for i in range(4)]
            n=sum(w['decodes'] for w in window.values());bad=sum(w['nacks'] for w in window.values())
            frames.append(dict(t=slot*dt,slot=slot,window_ms=window_n*slot_ms,connections=[row.get('connection_states',{}).get(u,'connected') for u in ids],
                cdelay=latency/completed if completed else None,deadline_misses=sum(u['packet_deadline_missed_packets'] for u in last.values()),
                g=sum(w['bytes'] for w in window.values())*8/seconds/1e6,cg=delivered*8/(slot*dt)/1e6,
                q=sum(u['packet_pending_bytes'] for u in last.values()),overflow=sum(u['packet_overflow_bytes'] for u in last.values()),
                delay=(latency-previous_latency)/(completed-previous_completed) if completed>previous_completed else None,
                nack=100*bad/n if n else None,decodes=cumulative[0]+cumulative[2],nacks=cumulative[1]+cumulative[3],
                users=user_rows,link=[counts[u].copy() for u in ids],cells=[[cells[c]['bytes']*8/seconds/1e6,cells[c]['occupancy']/window_n,cells[c]['power']/window_n] for c in cell_ids],
                hol_max_ms=[window[u]['hol'] for u in ids],offered_mbps=(offered-previous_offered)*8/seconds/1e6,
                rf_power_w=sum(c['power'] for c in cells.values())/window_n,rf_energy_j=energy,xapps=row.get('xapp_bindings',{})))
            previous_completed=completed;previous_latency=latency;previous_offered=offered;window_n=0;window,cells=empty_window()
    if expected!=total or frames[-1]['slot']!=total:raise ValueError('Incomplete replay trace')
    metrics=summary['packet_metrics'];totals={k:sum(m[k] for m in metrics.values()) for k in (
        'initial_bytes','offered_bytes','delivered_bytes','pending_bytes','overflow_bytes','completed_packets','late_completed_packets','deadline_missed_packets','completion_latency_ms_sum')}
    totals['mean_completion_latency_ms']=totals['completion_latency_ms_sum']/totals['completed_packets'] if totals['completed_packets'] else None
    for uid,m in metrics.items():
        for k in ['delivered_bytes','pending_bytes','overflow_bytes']:
            if last[uid]['packet_'+k]!=m[k]:raise ValueError(f'Replay/final packet counter mismatch: {uid}/{k}')
        if m['initial_bytes']+m['offered_bytes']!=m['delivered_bytes']+m['pending_bytes']+m['overflow_bytes']:raise ValueError('Packet byte conservation failed')
    expected_energy=summary.get('paper_interfaces',{}).get('rf_energy_j')
    if expected_energy is not None and not math.isclose(sum(expected_energy.values()),energy,abs_tol=1e-7,rel_tol=1e-9):raise ValueError('Replay RF-energy audit failed')
    network=summary.get('link_statistics',{}).get('network',{})
    if network.get('actual_decode_attempts',actual)!=actual:raise ValueError('Replay actual-decode audit failed')
    if 'transmission_attempts' in network and network['transmission_attempts']!=actual+sampled:raise ValueError('Replay feedback audit failed')
    profiles={u['ue_id']:u for u in config['configs']['ue']['ues']}
    users=[dict(id=u,slice=profiles[u].get('slice_id',''),budget_ms=profiles[u]['latency_budget_ms'],queue_cap=profiles[u].get('max_queue_bytes'),
        arrival_model=profiles[u].get('arrival_model'),arrival_rate_bps=profiles[u].get('arrival_rate_bps')) for u in ids]
    phy=config['configs']['phy']['rt_sys'];model='fast' if phy.get('feedback_mode')=='fast' or summary.get('diagnostic_model',{}).get('model')=='hybrid_sys' else 'full'
    candidates=[folder/'replay.mesh.npz',folder/'live.mesh.npz',folder/'scene.mesh.npz']
    mesh_file=Path(mesh_path) if mesh_path else next((p for p in candidates if p.is_file()),None)
    if mesh_file is None:raise ValueError('No scene mesh found; use --mesh or export during a simulation')
    mesh,view=render_mesh(mesh_file,manifest);duration=total*dt
    # Historical fast outputs used the full grid and predate profile metadata.
    fast_profile=phy.get('fast_profile','reference') if model=='fast' else None
    return dict(meta=dict(scene=manifest['scene'],model=model,fast_profile=fast_profile,completed=True,slots=total,slot_ms=slot_ms,simulation_s=duration,
        active_prbs=summary['n_prbs_total'],window_ms=window_slots*slot_ms,warmup_s=config.get('warmup_ms',0)/1000,wall_s=status.get('elapsed_s'),
        source_sha256=status.get('source_sha256'),source_run=folder.name,playback_speed=duration/20,
        first_overflow_s=first_overflow,actual_decodes=actual,sampled_attempts=sampled,xapp_pool=config['configs']['ric'].get('xapp_pool'),
        xapp_bindings_recorded=bindings_recorded,configured_xapps=len(config['configs']['ric'].get('xapps',[]))),manifest=manifest,mesh=mesh,view=view,frames=frames,users=users,totals=totals,energy_scope=ENERGY_SCOPE)

def configuration_rows(data,config):
    m=data['manifest'];r=config['configs'];cell=r['cell'];profile=cell.get('cell_radio_profiles',{})
    powers=[profile.get(c['cell_id'],{}).get('tx_power_dbm',cell.get('tx_power_dbm',30)) for c in m['cells']]
    speed=[]
    for u in m['ues']:
        tr=u['trajectory'];speed.extend(np.linalg.norm(np.diff(tr['positions_m'],axis=0)/np.diff(tr['times_s'])[:,None],axis=1).tolist())
    lo,hi=min(speed),max(speed);speed_label=f'{hi:g} m/s' if math.isclose(lo,hi) else f'{lo:g}–{hi:g} m/s'
    power=f'{powers[0]:g} dBm ({10**((powers[0]-30)/10):g} W)' if len(set(powers))==1 else 'Per-cell configuration'
    sim=r['sim'];rows=[('Cells',len(m['cells']),'UEs',len(m['ues'])),
        ('Carrier frequency',f"{m['carrier_frequency_hz']/1e9:g} GHz",'Channel bandwidth',f"{m['bandwidth_hz']/1e6:g} MHz"),
        ('Subcarrier spacing',f"{m['subcarrier_spacing_hz']/1000:g} kHz",'Active PRBs',data['meta']['active_prbs']),
        ('Initial Tx power / cell',power,'Antennas / link',f"{m['tx_rows']*m['tx_cols']} Tx × {m['num_rx_ant']} Rx"),
        ('UE speed',speed_label,'Slot duration',label_ms(m['slot_duration_ms'])),
        ('Simulation duration',f"{duration_label(data['meta']['simulation_s'])} ({m['total_slots']:,} slots)",'MU-MIMO','Enabled' if sim.get('scheduler_enable_spatial_multiplexing') else 'Disabled'),
        ('Model',data['meta']['model'].capitalize(),'xApp pool',data['meta'].get('xapp_pool') or 'N/A'),
        ('Control policy','Orchestrated' if config.get('control_mode')=='orchestrated' else 'Native baseline','xApps','Off' if not data['meta']['configured_xapps'] else 'Configured'),
        ('Max. reflection depth',m['ray_solver']['max_depth'],'Path update interval',label_ms(m['path_update_ms']))]
    if data['meta']['model']=='fast':
        efficient=data['meta'].get('fast_profile') in ('efficient','system')
        phy=r['phy']['rt_sys']
        algebra_device=phy.get('algebra_device','cuda' if phy.get('algebra_backend')=='cupy' else phy.get('device','N/A'))
        time_sampling='Per symbol' if not efficient else ('1 / gain update / stream pattern' if phy.get('system_phy_enabled') else '1 / slot / stream pattern')
        rows.extend([('Fast profile',data['meta'].get('fast_profile','reference').capitalize(),'PHY frequency sampling','4 SC / PRB' if efficient else 'Per RE'),
            ('PHY time sampling',time_sampling,'Linear algebra',algebra_device)])
        if phy.get('system_phy_enabled') or phy.get('reference_gain_update_ms'):
            rows.append(('Spatial gain update',label_ms(phy.get('system_gain_update_ms',data['meta']['slot_ms'])) if phy.get('system_phy_enabled') else 'Per slot',
                'Reference gain update',label_ms(phy.get('reference_gain_update_ms') or data['meta']['slot_ms'])))
    return ''.join('<tr>'+''.join(f'<td'+((' class="value" id="xapps-state"' if row[i-1]=='xApps' else ' class="value"') if i%2 else '')+'>'+html.escape(str(value))+'</td>' for i,value in enumerate(row))+'</tr>' for row in rows)

def export_replay(folder,output=None,*,provider=None,max_frames=500,window_ms=None,mesh_path=None):
    folder=Path(folder);target=Path(output) if output else folder/'network-replay.html'
    if mesh_path is None and provider is not None:
        mesh_path=write_scene_mesh(provider,folder/'replay.mesh.npz')
    data=build_data(folder,max_frames=max_frames,window_ms=window_ms,mesh_path=mesh_path)
    config=read_json(folder/'config_snapshot.json');template=(ASSETS/'template.html').read_text();app=(ASSETS/'replay.js').read_text()
    duration=data['meta']['simulation_s'];speed=data['meta']['playback_speed']
    choices=sorted({speed/2,speed,speed*2,1.})
    options=''.join(f'<option value="{v:.12g}"'+(' selected' if v==speed else '')+f'>{v:.5g}×</option>' for v in choices)
    replacements=dict(__REPLAY_DATA__=json.dumps(data,separators=(',',':'),ensure_ascii=True,allow_nan=False).replace('<',chr(92)+'u003c'),
        __THREE_MODULE__=base64.b64encode((ASSETS/'three.module.min.js').read_bytes()).decode(),__APP_CODE__=app,
        __CONFIGURATION_ROWS__=configuration_rows(data,config),__WINDOW_LABEL__=label_ms(data['meta']['window_ms']),
        __DURATION_LABEL__=duration_label(duration),__MEAN_GOODPUT__=f"{data['totals']['delivered_bytes']*8/duration/1e6:.2f}",
        __DURATION_MS__=f'{duration*1000:.12g}',__SLOT_MS__=str(data['meta']['slot_ms']),__SPEED_OPTIONS__=options,
        __WARMUP_LABEL__=f"Warmup: {duration_label(data['meta']['warmup_s'])}." if data['meta']['warmup_s'] else 'No warmup interval.',
        __ZERO_TIME__='0.000 ms' if duration<1 else '0.000 s')
    template=re.sub(r'__[A-Z_]+__',lambda match:replacements.get(match.group(),match.group()),template)
    template=template.replace('</head>','<!-- Three.js r170 license:\n'+(ASSETS/'THREE-LICENSE.txt').read_text()+'\n-->\n</head>')
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp');temporary.write_text(template);os.replace(temporary,target)
    report=dict(completed=True,html=str(target.resolve()),html_bytes=target.stat().st_size,html_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        self_contained=True,model=data['meta']['model'],fast_profile=data['meta']['fast_profile'],slots=data['meta']['slots'],simulation_s=duration,frames=len(data['frames']),window_ms=data['meta']['window_ms'],
        actual_decodes=data['meta']['actual_decodes'],sampled_attempts=data['meta']['sampled_attempts'],audits_passed=True)
    (folder/'replay_export.json').write_text(json.dumps(report,indent=2))
    return report
