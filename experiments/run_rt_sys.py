"""Dedicated, finite RT/SYS runner. Never falls back to an approximate PHY."""
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


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['simple_street_canyon','munich'],default='simple_street_canyon')
    p.add_argument('--cells',type=int,default=2); p.add_argument('--ues',type=int,default=4)
    p.add_argument('--seed',type=int,default=1); p.add_argument('--slots',type=int,default=40)
    p.add_argument('--path-update-ms',type=float)
    p.add_argument('--platform',choices=['dgx','agx'],
        help='dgx: CUDA/OptiX + PyTorch CUDA; agx: LLVM CPU RT/NR + CuPy CUDA algebra (CPU fallback if unavailable)')
    p.add_argument('--device',choices=['cuda','cpu'],
        help='PHY/SYS device; default CUDA without --platform, otherwise selected by platform')
    p.add_argument('--rt-device',choices=['cuda','cpu'],
        help='Ray-tracing device; defaults to --device. --device selects PHY/SYS.')
    p.add_argument('--algebra-backend',choices=['torch','cupy'],
        help='CuPy executes RZF/LMMSE on CUDA independently of the PHY/SYS device.')
    p.add_argument('--config',type=Path)
    p.add_argument('--control-timeline',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--live-state',type=Path,help='Atomically publish actual slot state for the read-only live viewer')
    p.add_argument('--live-xapp-test',action='store_true',help='Register reversible live test xApps, initially disabled')
    p.add_argument('--xapp-test-app',default='live_scheduler_test',help='Registered implementation to activate or compare')
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
    p.add_argument('--feedback-mode',choices=['nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'])
    p.add_argument('--warmup-ms',type=float,default=0)
    p.add_argument('--csi-error-nmse-db',type=float)
    p.add_argument('--harq-combining',choices=['none','chase'],default='none')
    p.add_argument('--recording-mode',choices=['audit','experiment'])
    p.add_argument('--policy-timeline',type=Path)
    p.add_argument('--channel-cache-root',type=Path)
    p.add_argument('--nr-execution',choices=['scalar','batched'])
    p.add_argument('--diagnostic-window-slots',type=int,default=200)
    return p.parse_args()


def run_case(args):
    from phy.rt_sys.platforms import resolve_platform
    platform_profile=resolve_platform(args)
    rt_device=getattr(args,'rt_device',None) or args.device
    manifest,configs,intent=build_case(args.scene,args.cells,args.ues,args.seed,args.slots,args.path_update_ms,args.device,args.config,
        deployment_id=getattr(args,'deployment_id',None),traffic_load=getattr(args,'traffic_load',1),
        initial_queue_bytes=getattr(args,'initial_queue_bytes',None),ue_speed_mps=getattr(args,'ue_speed_mps',None))
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
    execution_devices={'rt':rt_device,'phy_sys':args.device,
        'linear_algebra':'cuda' if algebra_backend=='cupy' else args.device,
        'channel_reconstruction':'cuda' if algebra_backend=='cupy' else 'cpu'}
    configs['phy']['rt_sys']['channel_array_backend']='cupy' if algebra_backend=='cupy' else 'numpy'
    if getattr(args,'feedback_mode',None):configs['phy']['rt_sys']['feedback_mode']=args.feedback_mode
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
    recording=getattr(args,'recording_mode',None) or configs['phy']['rt_sys'].get('recording_mode','audit')
    execution=getattr(args,'nr_execution',None) or configs['phy']['rt_sys'].get('nr_execution','scalar')
    configs['phy']['rt_sys'].update(recording_mode=recording,nr_execution=execution)
    if recording not in ('audit','experiment'):raise ValueError('Unknown recording mode')
    window_slots=getattr(args,'diagnostic_window_slots',200)
    if window_slots<1:raise ValueError('Positive diagnostic window required')
    name=f'{args.scene}_c{args.cells}_u{args.ues}_s{args.seed}_dt{manifest["path_update_ms"]}_{control_mode}'
    if rt_device!=args.device:name+=f'_rt{rt_device}_phy{args.device}'
    if algebra_backend=='cupy':name+='_cupy'
    output=(args.output or ROOT/'outputs/rt_sys'/name).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Outputs must stay in the independent copy')
    live_path=getattr(args,'live_state',None)
    if live_path is not None and not live_path.resolve().is_relative_to(ROOT/'outputs'):
        raise ValueError('Live telemetry must stay under project outputs/')
    if args.dry_run:
        print(json.dumps({'manifest':manifest,'output':str(output),'phy_config':configs['phy'],
            'execution_devices':execution_devices,'platform_profile':platform_profile},indent=2)); return
    output.mkdir(parents=True,exist_ok=True)
    if configs['phy']['rt_sys']['feedback_mode'] in ('nr_cached_reference','nr_bler_bank'):
        ref=configs['phy']['rt_sys'].setdefault('reference',{})
        bank=ref.setdefault('probability_bank',{})
        bank.setdefault('bank_file',str(output/'link_bler_bank.json'))
    source_hash=source_fingerprint(ROOT)
    snapshot={'source_sha256':source_hash,'manifest':manifest,'configs':configs,'intent':intent,'phase':phase,'control_mode':control_mode,'warmup_ms':warmup,'input_model_sha256':model_hash,'platform_profile':platform_profile,'validation':'integration or protocol execution; not an automatic generalization conclusion'}
    (output/'config_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    status={'completed':False,'spark_verified':False,'stage':'initializing','source_sha256':source_hash,
        'platform_profile':platform_profile,
        'execution_devices':execution_devices,
        'thread_budgets':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    def write_status(): (output/'run_status.json').write_text(json.dumps(status,indent=2))
    write_status(); started=time.perf_counter()
    from phy.rt_sys.efficiency import StageTimings
    timings=StageTimings();simulation_started=None;steady_started=None;steady_decodes=0;live=None
    try:
        from phy.rt_sys.channel import RtChannelProvider
        from phy.rt_sys.adapter import RtSysPhyAdapter
        cache_root=(getattr(args,'channel_cache_root',None) or ROOT/'channel_cache').resolve()
        if not cache_root.is_relative_to(ROOT):raise ValueError('Channel caches must stay in the independent copy')
        provider=RtChannelProvider(manifest,cache_root,device=rt_device,
            reconstruction_backend=configs['phy']['rt_sys']['channel_array_backend'])
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
        initial={'seed':args.seed,'cache_key':provider.key,'deployment':manifest,'business':configs['ue'],
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
        stack=ExitStack()
        auxiliary={name:stack.enter_context((output/f'{name}.jsonl').open('w'))
            for name in ('olla_feedback','olla_decisions','control_receipts','state_snapshots','diagnostic_trace','rule_decisions','xapp_selection')}
        simulation_started=time.perf_counter();io_s=0.;diagnostic_until=-1
        evaluated_bytes=0;evaluated_slots=0;completed_handovers=0
        with (output/'rt_sys_trace.jsonl').open('w') as trace, (output/'packet_events.jsonl').open('w') as packet_events, (output/'harq_events.jsonl').open('w') as harq_events:
            import csv
            with (output/'ue_metrics.csv').open('w',newline='') as ufile, (output/'cell_metrics.csv').open('w',newline='') as cfile:
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
        (output/'runtime_diagnostics.json').write_text(json.dumps(summary['runtime_diagnostics'],indent=2,allow_nan=False))
        (output/'steady_state.json').write_text(json.dumps(summary['steady_state'],indent=2))
        summary['packet_metrics']={u.ue_id:u.packet_ledger.summary() for u in simulator.state.ues}
        summary['du_olla']={u.ue_id:u.du_olla.summary() for u in simulator.state.ues if hasattr(u,'du_olla')}
        summary['tb_outcomes']={u.ue_id:dict(successful_tbs=len(u.packet_ledger.completed_tbs),payload_flushes=u.harq.harq_flush_count,retx_deferred_resource=getattr(u.harq,'retx_deferred_resource',0),pending_tbs=len(u.packet_ledger.reservations)) for u in simulator.state.ues}
        summary['harq_lifecycle']={u.ue_id:{'payload_flushes':u.harq.harq_flush_count,
            'empty_process_cleanup_count':u.harq.empty_process_cleanup_count} for u in simulator.state.ues}
        if live_test_enabled:summary['xapp_test']=simulator.live_xapp_test.snapshot()
        summary['rt_sys']={'environment':environment,'cache_key':provider.key,'ideal_csi':configs['phy']['rt_sys'].get('csi_error_nmse_db') is None,
            'harq_soft_combining':configs['phy']['rt_sys'].get('harq_combining','none')=='chase','full_phy_calibrated':False,
            'feedback_mode':configs['phy']['rt_sys'].get('feedback_mode','sys_abstraction'),
            'nr_equivalent_link_reference':getattr(adapter.abstraction,'feedback_mode',None) in ('nr_direct','nr_tb_reference','nr_cached_reference'),
            'default_bler_eesm_used':getattr(adapter.abstraction,'feedback_mode',None) not in ('nr_direct','nr_tb_reference','nr_cached_reference','nr_bler_bank'),
            'cache_hits':provider.hits,'cache_misses':provider.misses}
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
        status.update(completed=True,stage='completed',runtime_environment=environment,
            spark_verified=args.device=='cuda' and rt_device=='cuda' and environment['system']=='Linux' and environment['machine'] in ('aarch64','arm64') and environment['cuda_available'] and
            environment['rt_variant'].startswith('cuda') and 'GB10' in environment['gpu_name'])
        performance=dict(**timings.report(),recording_mode=recording,nr_execution=execution,
            initialization_s=simulation_started-started,simulation_s=simulation_elapsed,
            output_and_telemetry_s=io_s,steady_wall_s=steady_elapsed,
            steady_slots=evaluated_slots,steady_slots_per_s=evaluated_slots/max(steady_elapsed,1e-12),
            steady_actual_tbs_per_s=(adapter.abstraction.decode_trials-steady_decodes)/max(steady_elapsed,1e-12),
            actual_decodes=adapter.abstraction.decode_trials,codec_statistics=adapter.abstraction.statistics,
            source_sha256=source_hash,cache_key=provider.key,
            numerical_fidelity='actual_nr_20_decoder_iterations_no_online_probes',
            csv_semantics='audit: each slot; experiment: instantaneous samples at diagnostic-window boundaries')
        (output/'performance.json').write_text(json.dumps(performance,indent=2))
        if live:live.finish('completed')
    except Exception as exc:
        if live:live.finish('failed',f'{type(exc).__name__}: {exc}')
        status.update(stage='failed',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        if 'stack' in locals():stack.close()
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
        write_status()


if __name__=='__main__': run_case(parse_args())
