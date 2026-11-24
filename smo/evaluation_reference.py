"""Read-only common BEFORE evidence for an independent-run evaluation batch.

Never use an AFTER KPI, change an intent bound, or execute a controller here.
Older recordings may precede the baseline endpoint; that gap stays explicit.
"""
from copy import deepcopy
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import tarfile

from smo.runtime import records, replay
from smo.xapp_preview import PreviewStore

ORIGIN_SCHEMA='oran.evaluation-origin.v1'
_DYNAMICS_GROUPS=('core','phy','ran','ric','state','models','traffic','configs')
_DYNAMICS_EXPERIMENTS={f'experiments/{name}.py'for name in('hot_xapps','live_xapp_test','online_rf_xapps',
    'urllc_scenarios','rcp_perturbation','rt_live','scenario_scan_utils','fixed_policy_timeline',
    'fixed_control_timeline','xapp_preview','steady_state','rt_diagnostics')}


def _is_dynamics(name):
    p=Path(name)
    return p.suffix in('.py','.json','.yaml','.yml')and (p.parts[0]in _DYNAMICS_GROUPS or name in _DYNAMICS_EXPERIMENTS)


def _files_digest(files):
    if not files:raise ValueError('evaluation_origin_dynamics_files_missing')
    digest=hashlib.sha256()
    for name,data in sorted(files.items()):digest.update(name.encode());digest.update(data)
    return digest.hexdigest()


def dynamics_fingerprint(root):
    root=Path(root);files={}
    for group in _DYNAMICS_GROUPS:
        for path in (root/group).rglob('*'):
            if path.is_file()and _is_dynamics(str(path.relative_to(root))):files[str(path.relative_to(root))]=path.read_bytes()
    for name in _DYNAMICS_EXPERIMENTS:
        path=root/name
        if path.is_file():files[name]=path.read_bytes()
    return _files_digest(files)


def archived_dynamics_fingerprint(path,expected_archive_sha256):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_archive_sha256:
        raise ValueError('evaluation_origin_source_archive_mismatch')
    files={}
    with tarfile.open(path,'r:*')as archive:
        for item in archive:
            name=item.name.removeprefix('./')
            if not item.isfile()or not _is_dynamics(name):continue
            if Path(name).is_absolute()or'..'in Path(name).parts or name in files:
                raise ValueError('evaluation_origin_source_archive_invalid')
            files[name]=archive.extractfile(item).read()
    return _files_digest(files)


def origin_configuration(config):
    """Network recipe, independent of output paths and audit-only wrappers.

    The entire source hash remains in provenance. A change to the evaluation
    wrapper need not invalidate an otherwise identical reproducible prefix;
    its controls, counters, exogenous arrivals and model pins are also checked.
    """
    result=configuration(config)
    for key in ('source_sha256','platform_profile','validation','record_compression'):
        result.pop(key,None)
    for app in result.get('configs',{}).get('ric',{}).get('xapps',[]):
        params=app.get('parameters',{})
        model=params.get('model_path')
        if model:
            # Released adaptive inference uses the very same pinned model
            # file. Its project root may move between isolated workers.
            # Keep distinct research overrides strict; do not normalize them.
            if params.get('adaptive_policy_path')==model:
                params['adaptive_policy_path']=Path(model).name
            params['model_path']=Path(model).name
    ric=result.get('configs',{}).get('ric',{})
    if ric.get('persistent_memory_enabled')is False:ric.pop('memory_store_path',None)
    bank=result.get('configs',{}).get('phy',{}).get('rt_sys',{}).get('reference',{}).get('probability_bank',{})
    bank.pop('bank_file',None)
    # In-memory manifest/config values may contain tuples; archived JSON has
    # lists. Canonicalize both representations before a semantic comparison.
    return json.loads(json.dumps(result,allow_nan=False))


def trajectory_horizon_compatibility(actual,expected,baseline_slot):
    """Prove horizon truncation without discarding any actual geometry.

    Check every slot and every segment boundary over the complete shorter run,
    including the certified BEFORE prefix. The maximum declared speed is also
    retained because it affects the native geometry-block spacing.
    """
    result=dict(geometry_equal=False,method='every_slot_and_segment_boundary_position_velocity_check',
        baseline_slot=baseline_slot,position_absolute_tolerance_m=1e-10,velocity_absolute_tolerance_mps=1e-10,
        relative_tolerance=1e-12)
    def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
    def prepared(value):
        if not isinstance(value,dict)or set(value)!={'times_s','positions_m'}:raise ValueError('Unsupported trajectory declaration')
        times=value['times_s'];points=value['positions_m']
        if not isinstance(times,list)or not isinstance(points,list)or len(times)<2 or len(times)!=len(points):raise ValueError('Invalid trajectory')
        if any(type(t)not in(int,float)or not math.isfinite(t)for t in times)or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Invalid trajectory times')
        if any(not isinstance(p,list)or len(p)!=3 or any(type(x)not in(int,float)or not math.isfinite(x)for x in p)for p in points):raise ValueError('Invalid XYZ trajectory')
        velocities=[[(b-a)/(t1-t0)for a,b in zip(p0,p1)]for t0,t1,p0,p1 in zip(times,times[1:],points,points[1:])]
        return times,points,velocities
    def sample(trajectory,t):
        times,points,velocities=trajectory
        if t<times[0]-1e-12 or t>times[-1]+1e-12:raise ValueError('Trajectory outside common horizon')
        i=min(max(bisect_right(times,t)-1,0),len(times)-2);velocity=velocities[i]
        return [p+v*(t-times[i])for p,v in zip(points[i],velocity)],velocity
    try:
        if type(baseline_slot)is not int or baseline_slot<1:return result
        dt=actual.get('slot_duration_ms');other_dt=expected.get('slot_duration_ms')
        if type(dt)not in(int,float)or dt<=0 or not close(dt,other_dt):return result
        totals=[actual.get('total_slots'),expected.get('total_slots')]
        if any(type(n)is not int or n<baseline_slot for n in totals):return result
        count=min(totals);end=count*dt/1000
        left=actual.get('ues');right=expected.get('ues')
        if not isinstance(left,list)or not isinstance(right,list)or len(left)!=len(right)or not left:return result
        ids=[u.get('ue_id')for u in left]
        if any(not isinstance(i,str)for i in ids)or len(set(ids))!=len(ids)or ids!=[u.get('ue_id')for u in right]:return result
        checks=0;maximum_position_difference=0.;maximum_velocity_difference=0.
        for a,b in zip(left,right):
            at=prepared(a.get('trajectory'));bt=prepared(b.get('trajectory'))
            if not close(max(math.sqrt(sum(x*x for x in v))for v in at[2]),max(math.sqrt(sum(x*x for x in v))for v in bt[2])):return result
            times=sorted({slot*dt/1000 for slot in range(count+1)}|{t for t in at[0]+bt[0]if 0<=t<=end})
            for t in times:
                ap,av=sample(at,t);bp,bv=sample(bt,t);checks+=1
                maximum_position_difference=max(maximum_position_difference,max(abs(x-y)for x,y in zip(ap,bp)))
                maximum_velocity_difference=max(maximum_velocity_difference,max(abs(x-y)for x,y in zip(av,bv)))
                if any(not close(x,y)for x,y in zip(ap+av,bp+bv)):return result
        result.update(geometry_equal=True,ue_ids=ids,checked_through_slot=count,checked_before_slots=baseline_slot,
            position_velocity_checks=checks,maximum_position_difference_m=maximum_position_difference,
            maximum_velocity_difference_mps=maximum_velocity_difference,maximum_speed_equal=True)
    except (ValueError,TypeError,KeyError,OverflowError):return result
    return result


def prefix_configuration_compatibility(actual,expected,*,baseline_slot=None):
    """Compare a genuine BEFORE prefix while retaining the original v1 proof.

    The run horizon does not alter an already recorded prefix. Two relocated
    operator profiles are equivalent only while their SHA-256 and every other
    recorded profile field/configuration value are identical. Actual counters,
    exogenous arrivals, models and dynamics are checked separately by the guard.
    """
    actual=origin_configuration(actual);expected=origin_configuration(expected)
    ignored=[]
    a=actual.get('manifest',{});b=expected.get('manifest',{})
    geometry=None
    declarations_differ=a.get('ues')!=b.get('ues')
    declared=isinstance(a.get('ues'),list)and any(isinstance(u,dict)and 'trajectory'in u for u in a['ues'])
    if baseline_slot is not None and (declarations_differ or declared):
        geometry=trajectory_horizon_compatibility(a,b,baseline_slot)
        if geometry['geometry_equal']:
            for left,right in zip(a['ues'],b['ues']):left['trajectory']=deepcopy(right['trajectory'])
            if declarations_differ:ignored.append('manifest.ues.trajectory.proven_equivalent_shorter_horizon')
    geometry_equal=geometry is None or geometry['geometry_equal']
    if type(a.get('total_slots'))is int and type(b.get('total_slots'))is int and min(a['total_slots'],b['total_slots'])>0:
        if a['total_slots']!=b['total_slots']:ignored.append('manifest.total_slots')
        a.pop('total_slots');b.pop('total_slots')
    if geometry_equal:
        sa=actual.get('configs',{}).get('sim',{});sb=expected.get('configs',{}).get('sim',{})
        if type(sa.get('total_slots'))is int and type(sb.get('total_slots'))is int and min(sa['total_slots'],sb['total_slots'])>0:
            if sa['total_slots']!=sb['total_slots']:ignored.append('configs.sim.total_slots')
            sa.pop('total_slots');sb.pop('total_slots')
    for left,right,name in ((a.get('traffic_profile'),b.get('traffic_profile'),'manifest.traffic_profile.path'),
                           (actual.get('ue_services_profile'),expected.get('ue_services_profile'),'ue_services_profile.path')):
        if not isinstance(left,dict)or not isinstance(right,dict):continue
        pin=left.get('sha256')
        if not isinstance(pin,str)or len(pin)!=64 or any(c not in '0123456789abcdef'for c in pin)or pin!=right.get('sha256'):continue
        if left.get('path')!=right.get('path'):ignored.append(name)
        left.pop('path',None);right.pop('path',None)
    return dict(configuration_equal=geometry_equal and comparable(actual,expected),
        method='same_pinned_profile_content_and_native_prefix_with_run_metadata_compatibility',
        ignored_metadata_fields=ignored,profile_hashes_and_configuration_values_preserved=True,
        trajectory_horizon_compatibility=geometry)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def origin_frame(frame,window):
    """One common BEFORE window replaces point/cumulative KPI readouts."""
    result=deepcopy(frame)
    result['states']={k:v for k,v in result['states'].items()if not k.startswith('kpi:')}
    result['states'].update(deepcopy(window['states']))
    result['window_ms']=window['window_slots']*window.get('slot_duration_ms',.5)
    result['evaluation_window']=dict(start_slot=window['window_start_slot'],end_slot=window['slot'],slots=window['window_slots'])
    return result


def _exogenous(row):
    return dict(slot=row['slot_id'],traffic_arrivals=row.get('traffic_arrivals'),
        task_births=[{k:v for k,v in e.items()if k not in('latency_ms',)}for e in row.get('ue_service_events',[])if e.get('kind')=='task_generated'])


def _model_pins(run):
    path=Path(run)/'xapp_hot_start.json'
    if not path.is_file():raise ValueError('evaluation_origin_model_pins_missing')
    entries=json.loads(path.read_text()).get('models',{})
    if not entries:raise ValueError('evaluation_origin_model_pins_missing')
    result={}
    for name,item in entries.items():
        pin=item.get('sha256');relative=item.get('checkpoint')
        if not relative or not isinstance(pin,str):raise ValueError('evaluation_origin_model_pins_missing')
        path=(Path(run)/relative).resolve()
        if not path.is_relative_to(Path(run).resolve())or not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:
            raise ValueError('evaluation_origin_model_pin_mismatch')
        result[name]=pin
    return result


def prepare_origin(batch, output, *, observe_slots=500,source_archive=None):
    """Reuse a retained, genuine predeployment prefix; never an AFTER value.

    This prepares a reproducible evaluation start, not arbitrary state restore.
    Intent bounds are not rewritten. Existing constraints can be reviewed at
    this explicit current state, while their original anchors remain archived.
    """
    if type(observe_slots)is not int or not 1<=observe_slots<=20000:
        return dict(status='unavailable',reason='invalid_observation_slots')
    baseline=((observe_slots+1+39)//40)*40
    members=[m for m in batch.get('members',[])if m.get('phase')=='complete'and m.get('comparison')]
    if not members:return dict(status='unavailable',reason='evaluation_baseline_pending')
    archive=source_archive or batch.get('worker_source_archive_path')
    if not archive:return dict(status='unavailable',reason='evaluation_origin_source_archive_required')
    try:dynamics=archived_dynamics_fingerprint(archive,batch.get('worker_source_archive_sha256'))
    except (OSError,ValueError,tarfile.TarError):return dict(status='unavailable',reason='evaluation_origin_source_archive_unverified')
    from smo.contracts import KPIS
    from smo.runtime import FrameBuilder,trace_users
    from smo.window_metrics import WindowIndex
    for member in members:
        comparison=member['comparison']
        if comparison.get('baseline_origin')!='fresh_run_all_xapps_off'or comparison.get('baseline',{}).get('slot')!=baseline:continue
        run=Path(output)/member['job_id']/'data/run'
        try:
            config=json.loads((run/'config_snapshot.json').read_text())
            if config.get('manifest',{}).get('rng_seed')!=batch.get('seed'):continue
            before_preview=None
            for item in records(run,'xapp_decision_previews'):
                if item.get('slot',0)>baseline:break
                if item.get('slot')==baseline:before_preview=item
            if not before_preview or not before_preview.get('apps')or any(a.get('mode')!='cold_activation'for a in before_preview['apps']):continue
            index=WindowIndex(config,population=True);builder=FrameBuilder(config,population=True)
            prefix=hashlib.sha256();frame=None
            for row in records(run,'rt_sys_trace'):
                slot=row['slot_id']
                if slot>baseline:break
                if row.get('xapp_bindings'):raise ValueError('evaluation_origin_prefix_not_native')
                users=trace_users(row);services=row.get('ue_services',{});ap=row.get('ap_services',{}).get('cells',{})
                index.observe(slot,users,services,ap,row.get('ue_service_events',[]))
                frame=builder.frame(slot,users,services,ap,{})
                prefix.update(json.dumps(_exogenous(row),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            if frame is None or frame['slot']!=baseline:continue
            window=index.window(KPIS,baseline-observe_slots,baseline,through=baseline)
            window['slot_duration_ms']=config['manifest'].get('slot_duration_ms',.5)
            frame=origin_frame(frame,window)
            proof=dict(schema=ORIGIN_SCHEMA,seed=batch['seed'],baseline_slot=baseline,observe_slots=observe_slots,
                configuration=origin_configuration(config),before_frame=deepcopy(frame),
                prefix_exogenous_sha256=prefix.hexdigest(),model_sha256=_model_pins(run),
                dynamics_sha256=dynamics,source_archive_sha256=batch['worker_source_archive_sha256'],
                state_basis='reproduced_native_prefix_with_observation_and_control_checks',
                source_job_id=member['job_id'],source_sha256=config.get('source_sha256'),
                after_kpis_used=False,intent_bounds_rebased=False,new_simulations=0)
            proof['origin_id']=_digest(proof)
            frame['evaluation_origin']={k:proof[k]for k in('schema','origin_id','baseline_slot','observe_slots')}
            return dict(status='ready',frame=frame,preview=before_preview,proof=proof,run_path=str(run),source_job_id=member['job_id'])
        except (OSError,ValueError,KeyError,TypeError):continue
    return dict(status='unavailable',reason='evaluation_origin_exact_predeployment_prefix_unavailable')


def signature(ranges):
    return sorted((r['kpi'], r.get('lower'), r.get('upper'),
        r.get('lower_inclusive', True), r.get('upper_inclusive', True),
        bool(r.get('empty'))) for r in ranges)


def comparable(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(comparable(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(comparable(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def configuration(config):
    result = deepcopy(config)
    # Log destinations vary by UUID. All model, radio, traffic, seed and timing
    # settings remain in the fingerprint, including the source/model hashes.
    for app in result.get('configs', {}).get('ric', {}).get('xapps', []):
        app.get('parameters', {}).pop('log_path', None)
    result.get('ap_services_profile', {}).pop('path', None)
    return result


def load(batch, output, *, cache=None):
    members = [m for m in batch.get('members', []) if m.get('phase') == 'complete' and m.get('comparison')]
    if not members:
        return dict(status='unavailable', reason='evaluation_baseline_pending')
    paths = [(Path(output)/m['job_id']/'data/run', m) for m in members]
    if any(not (p/'config_snapshot.json').is_file() for p, _ in paths):
        return dict(status='unavailable', reason='evaluation_baseline_recording_missing')
    key = (batch['id'], tuple((m['job_id'], (p/'config_snapshot.json').stat().st_mtime_ns) for p, m in paths))
    if cache is not None and key in cache:
        return deepcopy(cache[key])
    reference = None; input_hashes = []
    for path, member in paths:
        comparison = member['comparison']
        baseline = comparison['baseline']['slot']
        application = comparison.get('application_slot')
        if type(application) is not int or application <= baseline or comparison.get('baseline_origin') != 'fresh_run_all_xapps_off':
            return dict(status='unavailable', reason='evaluation_baseline_not_predeployment')
        config = configuration(json.loads((path/'config_snapshot.json').read_text()))
        # Limit playback to genuine captured decisions strictly before action.
        source = replay(path, 180, population=True)
        visible = {f['slot']: f for f in source['frames']}
        slots = [p['slot'] for p in records(path, 'xapp_decision_previews')
            if type(p.get('slot')) is int and p['slot'] <= baseline and p['slot'] < application and p['slot'] in visible]
        if not slots:
            return dict(status='unavailable', reason='evaluation_baseline_preview_missing')
        slot = max(slots); frame = visible[slot]; preview = PreviewStore(path).read(slot)
        if any(a.get('mode') != 'cold_activation' for a in preview.get('apps', [])) or not preview.get('apps'):
            return dict(status='unavailable', reason='evaluation_baseline_not_cold')
        input_hashes.append([a.get('input_sha256') for a in preview['apps']])
        # Public-input hashes can differ with floating-point/device telemetry.
        # Compare retained state and actual proposed values with the declared
        # numeric tolerance; do not claim bit-identical public observations.
        check_preview = deepcopy(preview)
        for app in check_preview['apps']:
            app.pop('input_sha256', None)
        before = {r['kpi']: r['before'].get('value') for r in comparison['rows']}
        item = dict(config=config, baseline_slot=baseline, before=before, frame=frame, preview=check_preview)
        if reference is None:
            reference = item; original_preview = preview
        elif not comparable(reference, item):
            return dict(status='unavailable', reason='evaluation_baseline_mismatch')
    result = dict(status='ready', frame=reference['frame'], preview=original_preview, config=reference['config'],
        baseline_reference=reference['before'], reference=dict(
            method='common_recorded_predeployment_v1', batch_id=batch['id'],
            source_job_id=members[0]['job_id'], decision_slot=reference['frame']['slot'],
            baseline_slot=reference['baseline_slot'],
            decision_to_baseline_ms=(reference['baseline_slot']-reference['frame']['slot'])*source['slot_duration_ms'],
            matched_completed_runs=len(members), total_runs=len(batch['members']),
            config_sha256=hashlib.sha256(json.dumps(reference['config'], sort_keys=True).encode()).hexdigest(),
            seed=batch['seed'], observe_slots=batch['observe_slots'],
            baseline_equal=True, controls_equal=True, proposals_equal=True,
            numeric_relative_tolerance=1e-7, numeric_absolute_tolerance=1e-8,
            public_input_hashes_equal=all(h == input_hashes[0] for h in input_hashes),
            after_kpis_used=False, intent_bounds_rebased=False,
            temporal_scope='recorded_predeployment_decision',
            paired_no_xapp_continuation=False))
    if cache is not None:
        cache[key] = deepcopy(result)
    return result
