"""Finite RT network runner with explicit full/fast models and offline HTML replay."""
import argparse
from pathlib import Path
import sys
import json
import time
import resource
import platform
import hashlib
import os
from dataclasses import asdict

# Small per-link matrices suffer severe oversubscription in concurrent cases.
# Respect explicitly configured thread budgets; set defaults before NumPy/Torch.
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phy.rt_sys.scenario import build_case
from phy.rt_sys.contracts import source_fingerprint


def close_run_resources(resources):
    """Attempt every close; report cleanup failure without leaking later resources."""
    errors=[]
    for name,resource in resources:
        close=getattr(resource,'close',None)
        if close is None:continue
        try:close()
        except BaseException as exc:
            errors.append(dict(resource=name,error=f'{type(exc).__name__}: {exc}'))
    return errors


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['simple_street_canyon','munich'],default='simple_street_canyon')
    p.add_argument('--cells',type=int,default=2); p.add_argument('--ues',type=int,default=4)
    p.add_argument('--seed',type=int,default=1)
    duration=p.add_mutually_exclusive_group()
    duration.add_argument('--slots',type=int)
    duration.add_argument('--duration-s',type=float,help='Simulation duration, aligned to the configured slot duration')
    p.add_argument('--path-update-ms',type=float)
    p.add_argument('--max-geometry-displacement-m',type=float,help='Explicit motion-distance cap for RT geometry blocks')
    p.add_argument('--platform',choices=['dgx','agx'],
        help='dgx: CUDA/OptiX; agx: LLVM CPU RT/NR + CuPy (CPU fallback). Efficient fast overrides small-matrix algebra to CPU.')
    p.add_argument('--device',choices=['cuda','cpu'],
        help='NR/full SYS device; efficient fast SYS uses CPU. Default CUDA without --platform.')
    p.add_argument('--rt-device',choices=['cuda','cpu'],
        help='Ray-tracing device; defaults to --device. --device selects PHY/SYS.')
    p.add_argument('--algebra-backend',choices=['torch','cupy'],
        help='Algebra/channel reconstruction backend. Efficient fast always uses CPU RZF/LMMSE.')
    p.add_argument('--config',type=Path)
    p.add_argument('--control-timeline',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--live-state',type=Path,help='Atomically publish actual slot state for the read-only live viewer')
    p.add_argument('--live-xapp-test',action='store_true',help='Register reversible live test xApps, initially disabled')
    p.add_argument('--xapp-pool',choices=['yisong','chongyu'],help='Select the xApp owner pool; overrides config xapp_pool')
    p.add_argument('--xapp-test-app',help='Implementation in the selected pool to activate or compare')
    p.add_argument('--xapp-test-start',choices=['off','on'],default='off')
    p.add_argument('--xapp-test-cell',default='*',help='Target cell for the registered test; * means all cells')
    p.add_argument('--xapp-test-compare-slots',type=int,default=200)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--deployment-id',type=int,choices=[0,1,2])
    p.add_argument('--traffic-load',type=float,default=1)
    p.add_argument('--phase',choices=['integration','train','validation','test'],default='integration')
    p.add_argument('--selection-mode',choices=['reflective','fixed_plan'],default='reflective')
    p.add_argument('--control-mode',choices=['baseline','orchestrated'],default='baseline',
        help='Baseline disables rApps/xApps and uses fixed round-robin MAC')
    p.add_argument('--model-state',type=Path)
    p.add_argument('--initial-queue-bytes',type=int)
    p.add_argument('--ue-speed-mps',type=float)
    model=p.add_mutually_exclusive_group()
    model.add_argument('--model',choices=['full','fast'],help='full: actual NR; fast: 10 ms geometry and on-demand SYS channels with explicit NR fallback (default full)')
    p.add_argument('--fast-profile',choices=['efficient','reference','system'],help='fast only: system (default, 2 ms gains), per-slot efficient, or full-grid reference')
    p.add_argument('--fast-gain-update-ms',type=float,help='System profile only: override both spatial/reference gain update intervals')
    p.add_argument('--fast-channel-mode',choices=['dense','sampled'],help='System profile only: reconstruct full CFR or requested physical coordinates')
    p.add_argument('--rt-prefetch-workers',type=int,choices=range(1,9),help='CPU path prefetch workers; blocks have exclusive generation ownership')
    model.add_argument('--feedback-mode',choices=['nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'])
    p.add_argument('--warmup-ms',type=float,default=0)
    p.add_argument('--csi-error-nmse-db',type=float)
    p.add_argument('--harq-combining',choices=['none','chase'],default='none')
    p.add_argument('--recording-mode',choices=['audit','experiment'])
    p.add_argument('--record-compression',choices=['none','gzip'],default='none',help='Losslessly compress CSV/JSONL records; HTML reads either format')
    p.add_argument('--policy-timeline',type=Path)
    p.add_argument('--channel-cache-root',type=Path)
    p.add_argument('--nr-execution',choices=['scalar','batched','parallel'])
    p.add_argument('--diagnostic-window-slots',type=int,default=200)
    p.add_argument('--html',action=argparse.BooleanOptionalAction,default=True,help='Automatically export network-replay.html after a successful simulation')
    p.add_argument('--replay-window-ms',type=float,help='Optional KPI aggregation window; default adapts to run duration')
    p.add_argument('--replay-max-frames',type=int,default=500,help='Target replay time windows; raw data is retained')
    args=p.parse_args()
    if args.slots is None:args.slots=40
    return args


def run_case(args):
    from phy.rt_sys.platforms import resolve_platform
    platform_profile=resolve_platform(args)
    rt_device=getattr(args,'rt_device',None) or args.device
    duration_s=getattr(args,'duration_s',None)
    if duration_s is not None:
        import math
        base=json.loads((args.config or ROOT/'configs/rt_sys/default.json').read_text())
        slots=duration_s*1000/base['radio']['slot_duration_ms']
        if not math.isfinite(slots) or slots<=0 or not math.isclose(slots,round(slots),abs_tol=1e-8,rel_tol=1e-12):
            raise ValueError('--duration-s must be positive and align with whole time slots')
        args.slots=round(slots)
    manifest,configs,intent=build_case(args.scene,args.cells,args.ues,args.seed,args.slots,args.path_update_ms,args.device,args.config,
        deployment_id=getattr(args,'deployment_id',None),traffic_load=getattr(args,'traffic_load',1),
        initial_queue_bytes=getattr(args,'initial_queue_bytes',None),ue_speed_mps=getattr(args,'ue_speed_mps',None),
        xapp_pool=getattr(args,'xapp_pool',None))
    displacement=getattr(args,'max_geometry_displacement_m',None)
    if displacement is not None:
        import math
        if not math.isfinite(displacement) or displacement<=0:raise ValueError('Geometry displacement cap must be positive and finite')
        manifest['max_geometry_displacement_m']=displacement
    args.xapp_pool=configs['ric']['xapp_pool']
    phase=getattr(args,'phase','integration');mode=getattr(args,'selection_mode','reflective')
    warmup=getattr(args,'warmup_ms',0)
    if not 0<=warmup<manifest['total_slots']*manifest['slot_duration_ms']:raise ValueError('Warmup must leave an evaluation interval')
    model_path=getattr(args,'model_state',None);model_hash=None
    if model_path:
        model_path=model_path.resolve()
        if not model_path.is_relative_to(ROOT/'outputs'):raise ValueError('Model states must stay under copy outputs/')
        if model_path.exists():model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if phase in ('validation','test') and mode=='reflective':
        if not model_path or not model_path.is_file():raise ValueError('Frozen evaluation requires a training artifact')
        payload=json.loads(model_path.read_text())
        if not payload.get('success_cases') and not payload.get('failure_cases'):
            raise ValueError('Training memory is empty; cannot declare trained evaluation')
    configs['ric'].update(selection_mode=mode,freeze_learning=phase in ('validation','test'))
    control_mode=getattr(args,'control_mode','baseline')
    configs['phy']['rt_sys']['initial_association']='strongest_rt_reference'
    configs['phy']['rt_sys']['rt_device']=rt_device
    if getattr(args,'algebra_backend',None):
        configs['phy']['rt_sys']['algebra_backend']=args.algebra_backend
    algebra_backend=configs['phy']['rt_sys'].get('algebra_backend','torch')
    configs['phy']['rt_sys']['channel_array_backend']='cupy' if algebra_backend=='cupy' else 'numpy'
    if getattr(args,'feedback_mode',None):configs['phy']['rt_sys']['feedback_mode']=args.feedback_mode
    selected_model=getattr(args,'model',None) or (None if getattr(args,'feedback_mode',None) else configs['phy']['rt_sys'].get('model'))
    if selected_model is not None:
        if selected_model not in ('full','fast'):raise ValueError('model must be full or fast')
        configs['phy']['rt_sys']['feedback_mode']='fast' if selected_model=='fast' else 'nr_direct'
    selected_model='fast' if configs['phy']['rt_sys']['feedback_mode']=='fast' else 'full'
    configs['phy']['rt_sys']['model']=selected_model
    from phy.rt_sys.fast_profile import configure_fast_profile
    fast_profile=configure_fast_profile(configs['phy']['rt_sys'],getattr(args,'fast_profile',None))
    gain_update=getattr(args,'fast_gain_update_ms',None)
    if gain_update is not None:
        import math
        slots=gain_update/manifest['slot_duration_ms']
        if fast_profile!='system':raise ValueError('--fast-gain-update-ms requires --model fast --fast-profile system')
        if not math.isfinite(slots) or slots<1 or not math.isclose(slots,round(slots),rel_tol=1e-12,abs_tol=1e-8):
            raise ValueError('Fast gain interval must align with positive whole slots')
        configs['phy']['rt_sys'].update(system_gain_update_ms=gain_update,reference_gain_update_ms=gain_update)
    elif fast_profile=='system':
        phy_config=configs['phy']['rt_sys']
        if phy_config.get('system_gain_update_ms')==phy_config.get('reference_gain_update_ms'):
            # Freeze resolved defaults in subsequent OFF/ON comparison specs.
            args.fast_gain_update_ms=phy_config['system_gain_update_ms']
    args.fast_profile=fast_profile
    channel_mode=getattr(args,'fast_channel_mode',None)
    if channel_mode is not None:
        if fast_profile!='system':raise ValueError('--fast-channel-mode requires --model fast --fast-profile system')
        configs['phy']['rt_sys']['fast_channel_mode']=channel_mode
    channel_mode=configs['phy']['rt_sys'].get('fast_channel_mode','dense')
    if fast_profile=='system':args.fast_channel_mode=channel_mode
    if getattr(args,'rt_prefetch_workers',None) is not None:
        configs['phy']['rt_sys']['rt_prefetch_workers']=args.rt_prefetch_workers
    from phy.rt_sys.fast_profile import configure_fast_channel
    configure_fast_channel(configs['phy']['rt_sys'],manifest,
        path_update_ms=args.path_update_ms,displacement_m=displacement,rt_device=rt_device)
    if fast_profile=='system':
        # Freeze defaults for live restart and OFF/ON comparison commands.
        args.path_update_ms=manifest['path_update_ms']
        args.max_geometry_displacement_m=manifest['max_geometry_displacement_m']
        args.rt_prefetch_workers=configs['phy']['rt_sys']['rt_prefetch_workers']
    execution_devices={'rt':rt_device,'phy_sys':'cpu' if fast_profile in ('efficient','system') else args.device,
        'nr_fallback':args.device if selected_model=='fast' else None,
        'linear_algebra':'cpu' if fast_profile in ('efficient','system') else ('cuda' if algebra_backend=='cupy' else args.device),
        'channel_reconstruction':'cuda' if algebra_backend=='cupy' and channel_mode=='dense' else 'cpu'}
    args.model=selected_model if not getattr(args,'feedback_mode',None) else None
    if selected_model=='fast' and getattr(args,'harq_combining','none')!='none':raise ValueError('fast does not support waveform soft combining; select full')
    if control_mode == 'baseline':
        configs['ric'].update(rapps=[], xapps=[], freeze_learning=True, run_rapps_on_start=False)
        configs['phy']['rt_sys']['initial_association']='strongest_rt_reference'
    if control_mode == 'baseline' and phase != 'integration':
        raise ValueError('Fixed environment baseline is an integration run, not algorithm training/evaluation')
    from experiments.live_xapp_test import configure as configure_live_test
    live_test_enabled=configure_live_test(configs,manifest,args)
    if live_test_enabled:control_mode='xapp_test'
    if model_path:
        configs['ric'].update(persistent_memory_enabled=True,memory_store_path=str(model_path))
        for rapp in configs['ric'].get('rapps',[]):
            if rapp['type']=='xapp_selection_rapp':rapp.update(persistent_memory_enabled=True,memory_store_path=str(model_path))
    configs['phy']['rt_sys'].update(csi_error_nmse_db=getattr(args,'csi_error_nmse_db',None),harq_combining=getattr(args,'harq_combining','none'))
    if channel_mode=='sampled' and configs['phy']['rt_sys'].get('csi_error_nmse_db') is not None:
        raise ValueError('Sampled system channel currently requires ideal CSI')
    recording=getattr(args,'recording_mode',None) or configs['phy']['rt_sys'].get('recording_mode','audit')
    execution=getattr(args,'nr_execution',None) or configs['phy']['rt_sys'].get('nr_execution','scalar')
    if selected_model=='fast' and execution=='parallel':
        if getattr(args,'nr_execution',None)=='parallel':raise ValueError('fast supports scalar/batched feedback; process-parallel NR is a full-model option')
        execution='batched'
    configs['phy']['rt_sys'].update(recording_mode=recording,nr_execution=execution)
    if recording not in ('audit','experiment'):raise ValueError('Unknown recording mode')
    window_slots=getattr(args,'diagnostic_window_slots',200)
    if window_slots<1:raise ValueError('Positive diagnostic window required')
    if getattr(args,'replay_max_frames',500)<1:raise ValueError('Replay max frames must be positive')
    if getattr(args,'replay_window_ms',None) is not None:
        import math
        if not math.isfinite(args.replay_window_ms) or args.replay_window_ms<=0:raise ValueError('Replay window must be positive')
    name=f'{args.scene}_c{args.cells}_u{args.ues}_s{args.seed}_dt{manifest["path_update_ms"]}_{control_mode}_pool_{args.xapp_pool}_{selected_model}'
    if fast_profile:name+=f'_{fast_profile}'
    if rt_device!=args.device:name+=f'_rt{rt_device}_phy{args.device}'
    if algebra_backend=='cupy':name+='_cupy'
    if args.output is None:
        from datetime import datetime,timezone
        import uuid
        name+='_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output=(args.output or ROOT/'outputs/rt_sys'/name).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Outputs must stay in the independent copy')
    live_path=getattr(args,'live_state',None)
    if live_path is not None and not live_path.resolve().is_relative_to(ROOT/'outputs'):
        raise ValueError('Live telemetry must stay under project outputs/')
    if args.dry_run:
        print(json.dumps({'manifest':manifest,'output':str(output),'phy_config':configs['phy'],'model':selected_model,'automatic_html':getattr(args,'html',True),
            'execution_devices':execution_devices,'platform_profile':platform_profile,
            'xapp_pool':args.xapp_pool,'registered_xapps':configs['ric'].get('xapps',[])},indent=2)); return
    output.mkdir(parents=True,exist_ok=True)
    if configs['phy']['rt_sys']['feedback_mode'] in ('nr_cached_reference','nr_bler_bank'):
        ref=configs['phy']['rt_sys'].setdefault('reference',{})
        bank=ref.setdefault('probability_bank',{})
        bank.setdefault('bank_file',str(output/'link_bler_bank.json'))
    source_hash=source_fingerprint(ROOT)
    compression=getattr(args,'record_compression','none')
    snapshot={'source_sha256':source_hash,'manifest':manifest,'configs':configs,'intent':intent,'phase':phase,'control_mode':control_mode,'warmup_ms':warmup,'input_model_sha256':model_hash,'platform_profile':platform_profile,'record_compression':compression,'validation':'integration or protocol execution; not an automatic generalization conclusion'}
    (output/'config_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    status={'completed':False,'spark_verified':False,'stage':'initializing','model':selected_model,'fast_profile':fast_profile,'source_sha256':source_hash,'xapp_pool':args.xapp_pool,
        'platform_profile':platform_profile,
        'execution_devices':execution_devices,
        'thread_budgets':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    def write_status():
        temporary=output/f'.run_status.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(status,indent=2))
        os.replace(temporary,output/'run_status.json')
    write_status(); started=time.perf_counter()
    from phy.rt_sys.efficiency import StageTimings
    timings=StageTimings();simulation_started=None;steady_started=None;steady_decodes=0;live=None
    try:
        from phy.rt_sys.efficiency import configure_torch_threads
        status['torch_execution_threads']=configure_torch_threads(configs['phy']['rt_sys'])
        from phy.rt_sys.channel import RtChannelProvider
        from phy.rt_sys.adapter import RtSysPhyAdapter
        rt_cpu_threads=configs['phy']['rt_sys'].get('rt_cpu_threads')
        if rt_device=='cpu' and rt_cpu_threads is not None:
            if type(rt_cpu_threads) is not int or rt_cpu_threads<1:
                raise ValueError('rt_cpu_threads must be a positive integer')
            import drjit
            drjit.set_thread_count(rt_cpu_threads)
            status['rt_cpu_threads']=rt_cpu_threads
        cache_root=(getattr(args,'channel_cache_root',None) or ROOT/'channel_cache').resolve()
        if not cache_root.is_relative_to(ROOT):raise ValueError('Channel caches must stay in the independent copy')
        provider=RtChannelProvider(manifest,cache_root,device=rt_device,
            reconstruction_backend=configs['phy']['rt_sys']['channel_array_backend'],
            retain_device_cfr=configs['phy']['rt_sys'].get('retain_channel_on_gpu',False),
            prefetch=configs['phy']['rt_sys'].get('rt_prefetch',False),channel_mode=channel_mode,
            prefetch_workers=configs['phy']['rt_sys'].get('rt_prefetch_workers',1))
        if live_path:
            from experiments.rt_live import LivePublisher
            live=LivePublisher(live_path.resolve(),manifest,provider,execution_devices,configs['phy']['rt_sys']['feedback_mode'],
                configs=configs,control_mode=control_mode,platform_profile=platform_profile)
        timings.wrap(provider,'snapshot','channel_snapshot_including_io_and_cfr')
        for method in ('generate_paths','generate'):
            if hasattr(provider.backend,method):timings.wrap(provider.backend,method,'rt_generation')
        if args.generate_only:
            provider.pregenerate(); status.update(completed=True,stage='channels_generated',cache_key=provider.key)
            return
        from core.factory import build_controller_registry,build_scheduler
        from core.simulator import Simulator
        from core.debug import DebugConfig
        from experiments.scenario_scan_utils import build_runtime_scenario_config
        from ric.non_rt.active_intent_manager import ActiveIntentManager
        from scripts.spark.preflight import environment_info
        environment=environment_info()
        environment['platform_profile']=platform_profile
        environment['execution_devices']=execution_devices
        environment['algebra_backend']=algebra_backend
        if algebra_backend=='cupy':
            import cupy as cp
            environment['cupy']=cp.__version__
            environment['algebra_compute_capability']=cp.cuda.Device().compute_capability
            environment['algebra_gpu_name']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        adapter=RtSysPhyAdapter(configs['phy']['rt_sys'],channel_provider=provider)
        registry=build_controller_registry(configs['ric'])
        if getattr(args,'policy_timeline',None):
            from experiments.fixed_policy_timeline import FixedPolicyTimeline
            registry.register_rapp(FixedPolicyTimeline(args.policy_timeline))
            (output/'policy_timeline.json').write_text(args.policy_timeline.read_text())
        timeline_path=getattr(args,'control_timeline',None)
        if timeline_path:
            if not configs['phy']['rt_sys'].get('paper_interfaces',{}).get('enabled'):raise ValueError('Control timeline requires paper interfaces')
            from experiments.fixed_control_timeline import FixedControlTimeline
            registry.register_xapp(FixedControlTimeline(timeline_path))
            (output/'control_timeline.json').write_text(Path(timeline_path).read_text())
        registry.active_intent_manager=ActiveIntentManager(initial_intent_text=intent)
        runtime=build_runtime_scenario_config(scenario_name=name,scenario_type='rt_scene',sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
        simulator=Simulator(runtime,registry,build_scheduler(configs['sim']),adapter,configs['cell'],configs['ue'],configs['beamforming'],str(output))
        if recording=='experiment' and configs['phy']['rt_sys'].get('compact_snapshots',False):
            simulator.state.snapshot_interval_slots=window_slots
            simulator.state.snapshot_final_slot=args.slots
        if live_test_enabled:
            from experiments.live_xapp_test import LiveXAppSession
            args._live_slot_ms=manifest['slot_duration_ms']
            simulator.live_xapp_test=LiveXAppSession(simulator,args,output,live_path)
        for obj,method,label in ((simulator,'step','slot_total'),
            (simulator,'_run_managed_cell_schedulers','scheduling'),
            (adapter,'run_cluster_slot','precoding_interference_and_tb_inclusive'),
            (simulator.measurement_updater,'update','measurement_update'),
            (simulator.state_updater,'apply_phy_result','state_update'),
            (simulator.controller_manager,'run_near_rt','near_rt_control')):
            if hasattr(obj,method):timings.wrap(obj,method,label)
        for method,label in (('waveform','codec_lookup_or_build'),('_transmitted_feedback','actual_tb_feedback'),('evaluate_many','batch_feedback_inclusive')):
            if hasattr(adapter.abstraction,method):timings.wrap(adapter.abstraction,method,label)
        initial={'xapp_bindings':simulator.controller_manager.instance_selection.bindings or {},'seed':args.seed,'cache_key':provider.key,'deployment':manifest,'business':configs['ue'],
            'ues':[{"ue_id":u.ue_id,"serving_cell_id":simulator.state.connections[u.ue_id].serving_cell_id,"beam_id":u.serving_beam_id,"queue_bytes":u.queue_bytes} for u in simulator.state.ues]}
        from phy.rt_sys.contracts import digest
        initial['environment_sha256']=digest(initial)
        (output/'initial_conditions.json').write_text(json.dumps(initial,indent=2))
        DebugConfig.SLOT_METRICS=False
        status.update(stage='simulating',cache_key=provider.key); write_status()
        from experiments.steady_state import SteadyStateMonitor
        steady=SteadyStateMonitor(simulator.state.ues,warmup,manifest['slot_duration_ms'])
        from experiments.rt_diagnostics import RuntimeDiagnostics,compact_trace
        diagnostics=RuntimeDiagnostics(simulator.state.ues,manifest['slot_duration_ms'],warmup,window_slots*manifest['slot_duration_ms'])
        from contextlib import ExitStack
        from core.record_io import open_record
        simulator.result_writer.record_compression=compression
        status['record_compression']=compression
        stack=ExitStack()
        auxiliary={name:stack.enter_context(open_record(output/f'{name}.jsonl','w',compression=compression))
            for name in ('olla_feedback','olla_decisions','control_receipts','state_snapshots','diagnostic_trace','rule_decisions','xapp_selection')}
        simulation_started=time.perf_counter();io_s=0.;diagnostic_until=-1
        evaluated_bytes=0;evaluated_slots=0;completed_handovers=0
        with open_record(output/'rt_sys_trace.jsonl','w',compression=compression) as trace, open_record(output/'packet_events.jsonl','w',compression=compression) as packet_events, open_record(output/'harq_events.jsonl','w',compression=compression) as harq_events:
            import csv
            with open_record(output/'ue_metrics.csv','w',compression=compression,newline='') as ufile, open_record(output/'cell_metrics.csv','w',compression=compression,newline='') as cfile:
                uw=cw=None
                for _ in range(args.slots):
                    if steady_started is None and simulator.state.sim_time_ms>=warmup:
                        steady_started=time.perf_counter();steady_decodes=adapter.abstraction.decode_trials
                    if live_test_enabled:simulator.live_xapp_test.before_step()
                    simulator.step()
                    selection=simulator.controller_manager.instance_selection
                    for event in selection.receipts:auxiliary['xapp_selection'].write(json.dumps(event)+'\n')
                    selection.receipts.clear()
                    for event in simulator.controller_manager.policy_store.binding_receipts:auxiliary['xapp_selection'].write(json.dumps(event)+'\n')
                    simulator.controller_manager.policy_store.binding_receipts.clear()
                    feedback,decisions=diagnostics.observe(simulator.state,adapter.trace)
                    if live_test_enabled:simulator.live_xapp_test.after_step(adapter)
                    if live:live.publish(simulator,adapter)
                    io_start=time.perf_counter()
                    for event in getattr(simulator.controller_manager,'rule_decisions',[]):
                        auxiliary['rule_decisions'].write(json.dumps({'slot_id':simulator.state.slot_id,**event},allow_nan=False)+'\n')
                    for name,events in (('olla_feedback',feedback),('olla_decisions',decisions),
                        ('control_receipts',simulator.state.control.actuation_outcomes)):
                        for event in events:auxiliary[name].write(json.dumps({'slot_id':simulator.state.slot_id,**event},allow_nan=False)+'\n')
                    if recording=='audit':auxiliary['state_snapshots'].write(json.dumps(simulator.state.snapshots[-1],allow_nan=False)+'\n')
                    simulator.state.snapshots[:]=simulator.state.snapshots[-1:]
                    slot_packet_events=[]
                    ue_by_id={u.ue_id:u for u in simulator.state.ues}
                    for row in adapter.trace['ues']:
                        user=ue_by_id[row['ue_id']]
                        row.update({'packet_'+k:v for k,v in user.packet_ledger.summary().items()})
                        sample=simulator.state.measurement.ue_measurements[user.ue_id]
                        row.update(reference_rsrp_dbm=sample.rsrp_dbm,reference_sinr_db=sample.reference_sinr_db,
                            data_sinr_db=sample.data_sinr_db,measurement_sinr_kind=sample.sinr_kind,
                            measured_hol_delay_ms=sample.hol_delay_ms)
                        for event in user.packet_ledger.drain_events():
                            packet_events.write(json.dumps(event)+'\n');slot_packet_events.append(event)
                        for event in user.harq.events:
                            harq_events.write(json.dumps({'ue_id':user.ue_id,**event})+'\n')
                        user.harq.events.clear()
                    steady.observe(simulator.state.sim_time_ms,simulator.state.ues,adapter.trace,slot_packet_events)
                    adapter.trace['mobility_events']=simulator.mobility_manager.events
                    adapter.trace['handover_counts']={uid:c.handover_count for uid,c in simulator.state.connections.items()}
                    adapter.trace['connection_states']={uid:c.connection_state for uid,c in simulator.state.connections.items()}
                    if simulator.state.sim_time_ms>warmup:
                        completed_handovers+=sum(e['handover_completed'] for e in simulator.mobility_manager.events)
                        evaluated_bytes+=sum(u['delivered_bytes'] for u in adapter.trace['ues']);evaluated_slots+=1
                    full=simulator.slot_records[-1]['rt_sys']
                    full['xapp_bindings']=simulator.controller_manager.instance_selection.bindings or {}
                    if any(r.get('ack_source') in ('harq_incompatible_no_tb','unsupported_nr_code_no_tb') for r in adapter.trace['ues']) or any(e.get('status')=='rejected' for e in simulator.state.control.actuation_outcomes) or any(e.get('kind') in ('connection_drop','handover_failure') for e in getattr(getattr(simulator.state,'paper',None),'events',[])):
                        diagnostic_until=simulator.state.slot_id+window_slots
                    # Full audit output is a single stream, not duplicated in
                    # every historical scalar-metrics record and CSV column.
                    trace.write(json.dumps(full if recording=='audit' else compact_trace(full),allow_nan=False)+'\n')
                    if recording=='experiment' and simulator.state.slot_id<=diagnostic_until:
                        auxiliary['diagnostic_trace'].write(json.dumps(full,allow_nan=False)+'\n')
                    simulator.slot_records[-1].pop('rt_sys')
                    for row in adapter.trace['ues']:
                        if recording=='experiment' and simulator.state.slot_id%window_slots and simulator.state.slot_id!=args.slots:continue
                        row={'slot_id':simulator.state.slot_id,**row}
                        if uw is None: uw=csv.DictWriter(ufile,fieldnames=list(row)); uw.writeheader()
                        uw.writerow(row)
                    for cid,row in adapter.trace['cells'].items():
                        if recording=='experiment' and simulator.state.slot_id%window_slots and simulator.state.slot_id!=args.slots:continue
                        row={'slot_id':simulator.state.slot_id,'cell_id':cid,**row}
                        if cw is None: cw=csv.DictWriter(cfile,fieldnames=list(row)); cw.writeheader()
                        cw.writerow(row)
                    io_s+=time.perf_counter()-io_start
                    if simulator.state.slot_id%1000==0 or simulator.state.slot_id==args.slots:
                        status.update(last_completed_slot=simulator.state.slot_id,
                            simulated_seconds=simulator.state.sim_time_ms/1000,
                            simulation_wall_s_so_far=time.perf_counter()-simulation_started)
                        write_status()
        simulation_elapsed=time.perf_counter()-simulation_started
        steady_elapsed=time.perf_counter()-steady_started if steady_started is not None else 0.
        stack.close()
        summary=simulator._build_final_summary()
        if getattr(simulator.state,'paper',None):
            paper=simulator.state.paper
            summary['paper_interfaces']=dict(model=paper.model,connection_events_enabled=paper.events_enabled,final_reports=paper.published,events=paper.all_events,rf_energy_j=paper.rf_energy)
            (output/'connection_events.json').write_text(json.dumps(paper.all_events,indent=2))
            for app in registry.xapps:
                if app.name=='fixed_control_timeline':(output/'public_kpi_inputs.json').write_text(json.dumps(app.inputs,indent=2))
        if any(getattr(app,'rule_based_v1',False) for app in registry.xapps):
            summary['rule_based_v1']={app.name:dict(statistics=dict(app.statistics),decision_seconds=getattr(app,'decision_seconds',0.),
                last_decisions=app.last_decisions,kind=getattr(app,'kind','monitor')) for app in registry.xapps if getattr(app,'rule_based_v1',False)}
        if any(getattr(app,'library_instance',False) for app in registry.xapps):
            summary['xapp_library']=dict(instances=[a.descriptor() for a in registry.xapps if getattr(a,'library_instance',False)],
                bindings=simulator.controller_manager.instance_selection.bindings)
        if hasattr(adapter.abstraction,'bank'):adapter.abstraction.bank.save()
        if hasattr(adapter.abstraction,'harq'):adapter.abstraction.statistics.update(adapter.abstraction.harq.stats)
        summary['link_reference_statistics']=getattr(adapter.abstraction,'statistics',{})
        summary['steady_state']=steady.summary()
        summary['runtime_diagnostics']=diagnostics.summary(summary['steady_state'])
        summary['link_statistics']=summary['runtime_diagnostics']['link_statistics']
        summary['completion_latency_scope']=summary['runtime_diagnostics']['completion_latency_scope']
        (output/'runtime_diagnostics.json').write_text(json.dumps(summary['runtime_diagnostics'],indent=2,allow_nan=False))
        (output/'steady_state.json').write_text(json.dumps(summary['steady_state'],indent=2))
        summary['packet_metrics']={u.ue_id:u.packet_ledger.summary() for u in simulator.state.ues}
        summary['du_olla']={u.ue_id:u.du_olla.summary() for u in simulator.state.ues if hasattr(u,'du_olla')}
        summary['tb_outcomes']={u.ue_id:dict(successful_tbs=len(u.packet_ledger.completed_tbs),payload_flushes=u.harq.harq_flush_count,retx_deferred_resource=getattr(u.harq,'retx_deferred_resource',0),pending_tbs=len(u.packet_ledger.reservations)) for u in simulator.state.ues}
        summary['harq_lifecycle']={u.ue_id:{'payload_flushes':u.harq.harq_flush_count,
            'empty_process_cleanup_count':u.harq.empty_process_cleanup_count} for u in simulator.state.ues}
        if live_test_enabled:summary['xapp_test']=simulator.live_xapp_test.snapshot()
        summary['rt_sys']={'model':selected_model,'fast_profile':fast_profile,'environment':environment,'cache_key':provider.key,'ideal_csi':configs['phy']['rt_sys'].get('csi_error_nmse_db') is None,
            'harq_soft_combining':configs['phy']['rt_sys'].get('harq_combining','none')=='chase','full_phy_calibrated':False,
            'feedback_mode':configs['phy']['rt_sys'].get('feedback_mode','sys_abstraction'),
            'nr_equivalent_link_reference':getattr(adapter.abstraction,'feedback_mode',None) in ('nr_direct','nr_tb_reference','nr_cached_reference'),
            'default_bler_eesm_used':getattr(adapter.abstraction,'feedback_mode',None) not in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'),
            'cache_hits':provider.hits,'cache_misses':provider.misses}
        summary['rt_sys']['legacy_diagnostic_fields']={
            'pre_eq_sinr_db':'Alias of post-LMMSE arithmetic-mean SINR; not an independent pre-equalizer measurement',
            'ber':'Not measured; legacy default must not be interpreted as zero measured BER',
            'beam_gain_db':'Legacy compatibility value, not a measured beam gain; use supported reference reports',
            'precoder_match_score':'Not measured; legacy compatibility default is not evidence of precoder quality',
            'cqi':'MCS-derived legacy hint; use independently generated paper wideband/subband CQI reports',
            'rank_and_ri':'Allocated layer count, not an independently estimated propagation-channel rank'}
        summary['rt_sys']['energy_scope']='Emitted RF energy only; excludes device/baseband/circuit power'
        summary['rt_sys']['feedback_scope']='Fast uses SYS-sampled and actual-NR ACK/NACK; actual_decode distinguishes them'
        if hasattr(adapter.abstraction,'model_statistics'):summary['model_statistics']=adapter.abstraction.model_statistics()
        if hasattr(adapter,'system_gains'):summary['model_statistics']['system_gain_statistics']=dict(adapter.system_gains.stats)
        gain_cache=getattr(getattr(simulator.state,'paper',None),'gain_cache',None)
        if gain_cache is not None:
            summary['model_statistics']['reference_gain_statistics']=dict(builds=gain_cache.builds,hits=gain_cache.hits,update_slots=gain_cache.slots)
        simulator.result_writer.write_slot_metrics(simulator.slot_records)
        simulator.result_writer.write_final_summary(summary)
        simulator.result_writer.write_orchestration_epoch_history(simulator.controller_manager.get_orchestration_epoch_history())
        if phase=='train' and model_path:
            for app in registry.rapps:
                if hasattr(app,'_persist_memory_state'):app._persist_memory_state()
        if phase in ('validation','test') and model_hash:
            if hashlib.sha256(model_path.read_bytes()).hexdigest()!=model_hash:raise RuntimeError('Evaluation mutated training artifact')
        evaluation={'phase':phase,'selection_mode':mode,'control_mode':control_mode,'warmup_ms':warmup,'evaluated_slots':evaluated_slots,
            'delivered_bytes':evaluated_bytes,'throughput_mbps':evaluated_bytes*8/max(evaluated_slots*manifest['slot_duration_ms']*1000,1),
            'completed_handovers':completed_handovers,'training_artifact_sha256':model_hash,
            'learning_frozen':configs['ric']['freeze_learning']}
        (output/'evaluation_summary.json').write_text(json.dumps(evaluation,indent=2))
        status.update(simulation_completed=True,stage='finalizing',runtime_environment=environment,
            spark_verified=args.device=='cuda' and rt_device=='cuda' and environment['system']=='Linux' and environment['machine'] in ('aarch64','arm64') and environment['cuda_available'] and
            environment['rt_variant'].startswith('cuda') and 'GB10' in environment['gpu_name'])
        performance=dict(**timings.report(),channel_mode=channel_mode,channel_sampling_statistics=provider.sampling_statistics,
            recording_mode=recording,nr_execution=execution,
            torch_execution_threads=status.get('torch_execution_threads'),
            reference_cache_statistics=dict(getattr(getattr(simulator.state,'paper',None),'reference_cache_stats',{})),
            rt_prefetch_statistics=provider.prefetch_statistics(),
            initialization_s=simulation_started-started,simulation_s=simulation_elapsed,
            output_and_telemetry_s=io_s,steady_wall_s=steady_elapsed,
            steady_slots=evaluated_slots,steady_slots_per_s=evaluated_slots/max(steady_elapsed,1e-12),
            steady_actual_tbs_per_s=(adapter.abstraction.decode_trials-steady_decodes)/max(steady_elapsed,1e-12),
            actual_decodes=adapter.abstraction.decode_trials,codec_statistics=adapter.abstraction.statistics,
            source_sha256=source_hash,cache_key=provider.key,
            model=selected_model,fast_profile=fast_profile,
            numerical_fidelity='sys_default_eesm_bernoulli_with_explicit_nr_fallback' if selected_model=='fast' else 'actual_nr_20_decoder_iterations_no_online_probes',
            csv_semantics='audit: each slot; experiment: instantaneous samples at diagnostic-window boundaries')
        (output/'performance.json').write_text(json.dumps(performance,indent=2))
        if getattr(args,'html',True):
            status.update(stage='exporting_replay',elapsed_s=time.perf_counter()-started);write_status()
            export_started=time.perf_counter()
            from experiments.replay import export_replay
            status['replay']=export_replay(output,provider=provider,
                max_frames=getattr(args,'replay_max_frames',500),window_ms=getattr(args,'replay_window_ms',None),allow_finalizing=True)
            performance['replay_export_s']=time.perf_counter()-export_started
            (output/'performance.json').write_text(json.dumps(performance,indent=2))
            print('Replay: '+status['replay']['html'],flush=True)
        else:
            from experiments.replay import write_scene_mesh
            write_scene_mesh(provider,output/'replay.mesh.npz')
        status['stage']='finalizing'
    except BaseException as exc:
        stage=('cancelled' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else
               'replay_export_failed' if status.get('stage')=='exporting_replay' else 'failed')
        status.update(completed=False,stage=stage,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        active_exception=sys.exc_info()[1]
        cleanup_errors=close_run_resources([
            ('phy',getattr(locals().get('adapter'),'abstraction',None)),
            ('rt',locals().get('provider')),('writers',locals().get('stack'))])
        status.update(elapsed_s=time.perf_counter()-started,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        status['peak_rss_bytes']=status['peak_rss_native'] if platform.system()=='Darwin' else status['peak_rss_native']*1024
        try:
            import torch
            status['peak_cuda_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        except ImportError: pass
        if algebra_backend=='cupy':
            import cupy
            cupy.cuda.Stream.null.synchronize()
            status['cupy_pool_reserved_bytes']=cupy.get_default_memory_pool().total_bytes()
        status['output_bytes']=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if cleanup_errors:
            status.update(completed=False,stage='cleanup_failed',cleanup_errors=cleanup_errors)
            status.setdefault('error','; '.join(e['error'] for e in cleanup_errors))
        elif status.get('stage')=='finalizing':status.update(completed=True,stage='completed')
        if live:
            try:live.finish('completed' if status.get('completed') else status['stage'],status.get('error'))
            except BaseException as exc:
                cleanup_errors.append(dict(resource='live',error=f'{type(exc).__name__}: {exc}'))
                status.update(completed=False,stage='cleanup_failed',cleanup_errors=cleanup_errors)
                status.setdefault('error',cleanup_errors[-1]['error'])
        write_status()
        if cleanup_errors and active_exception is None:
            raise RuntimeError('Run cleanup failed: '+'; '.join(e['error'] for e in cleanup_errors))


if __name__=='__main__': run_case(parse_args())
