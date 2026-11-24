"""One local SMO with separate, explicitly selected traffic-context graphs.

The manifest is an operator-owned file, never a browser-supplied model or path.
Only the native view can reuse the existing native deployment pool. Pressure
views remain read-only until a matching deployment configuration is provided.
"""
import argparse
from copy import deepcopy
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

from smo.__main__ import GraphService, handler, export_html
from smo.intent_language import IntentLanguage,IntentError,LlmUnavailable,resolve_ranges
from smo.presets import PresetStore


def selected_controller_validation(active_release,recipe,environment,root):
    """Expose the selected actors' actual final window, separately from BEFORE."""
    if active_release.get('models',{}).get('x12_mec',{}).get('implementation_variant')=='adaptive_v2':
        from smo.x12_release_validation import validate
        return validate(active_release,recipe,environment,root)
    pin=active_release.get('final_report_sha256')
    path=Path(root)/'artifacts/xapps/chongyu'/recipe['model_release']['id']/'evidence/final-holdout-report.json'
    if (not isinstance(pin,str)or not re.fullmatch('[0-9a-f]{64}',pin)
            or any(not re.fullmatch('[0-9a-f]{64}',active_release.get(key)or'')for key in ('selection_sha256','source_archive_sha256'))
            or hashlib.sha256(path.read_bytes()).hexdigest()!=pin):
        raise ValueError('The selected release lacks its exact final paired report')
    report=json.loads(path.read_text());evidence=environment.get('model_evaluation',{})
    actors={'power':'x01','throughput':'x08'};seeds=[341721,341722]
    rows=report.get('comparisons',[]);cases=report.get('case_evidence',[])
    expected={(actor,seed)for actor in actors for seed in seeds}
    if (evidence.get('source')!='actual_final_paired_report'or evidence.get('report_sha256')!=pin
            or evidence.get('final_evaluated_actors')!=['x01','x08']or evidence.get('all_six_newly_retested')is not False
            or report.get('schema')!='oran.actual-xapp-comparison-report.v1'or report.get('role')!='holdout'
            or report.get('synthetic_fixture')is not False or report.get('registered_seeds')!=seeds
            or report.get('selection_sha256')!=active_release.get('selection_sha256')
            or report.get('window')!={'total_slots':22440,'start_slot_exclusive':2040,'end_slot_inclusive':22040,'duration_s':10.}
            or len(rows)!=4 or {(row.get('actor'),row.get('seed'))for row in rows}!=expected
            or any(row.get('role')!='holdout'or row.get('label')!='selected'
                or row.get('model_sha256')!=active_release['models'][actors[row['actor']]]['sha256']
                or row.get('prefix_certificate',{}).get('passed')is not True for row in rows)
            or len(cases)!=6 or {(case.get('actor'),case.get('seed'))for case in cases}!=
                {(actor,seed)for actor in (None,*actors)for seed in seeds}
            or any(case.get('source_archive_sha256')!=active_release.get('source_archive_sha256')
                or case.get('raw_endpoint_verification',{}).get('window_seconds')!=10.
                or case.get('raw_endpoint_verification',{}).get('contiguous_slots_verified')!=22040 for case in cases)):
        raise ValueError('The selected release must bind both actors and both real final ten-second paired windows')
    return dict(window_verified=True,paired_controllers_verified=True,window_basis='actual_final_paired_report',
        observe_slots=20000,before_observe_slots=2000,baseline_slot=2040,
        paired_evaluated_actors=['power','throughput'],paired_evaluated_models=['x01','x08'],
        paired_report_sha256=pin,seeds=seeds,six_apps_newly_retested=False,report_url='/environment-report/')


def attach_environment(service,entry):
    """Attach an operator-pinned runtime and keep old calibration separate."""
    if not entry.get('environment'):
        return
    from smo.runtime_recipe import load_recipe
    from smo.xapp_preview import PreviewStore
    release_path=Path(entry['environment'])
    release=json.loads(release_path.read_text())
    if release.get('status')!='validated_common_environment':
        raise ValueError('The selected environment must have a validated release')
    recipe,proof=load_recipe(entry['runtime_recipe'])
    from smo.model_release import verify_release,validate_preview,descriptor,MODEL_IDS
    active_release=verify_release(recipe,Path(__file__).resolve().parents[1])
    if recipe['id']!=entry['id'] or release['source_sha256']!=recipe['validated_source_sha256'] or release['source_archive_sha256']!=recipe['validated_source_archive_sha256'] or release['frozen_models']!=recipe['frozen_models']:
        raise ValueError('Runtime recipe differs from the validated environment')
    for field,name in [('traffic_profile','traffic.json'),('ue_services_config','ue-services.json'),('ap_services_profile','ap-services.json'),('base_config','base.json')]:
        path=Path(recipe['profiles'][field]['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=release['files_sha256']['configs/'+name]:
            raise ValueError('Runtime environment profile differs from validation')
    base=json.loads(Path(recipe['profiles']['base_config']['path']).read_text())
    reference=json.loads(Path(entry['statistical_reference']).read_text())
    if reference.get('current_scenario_id')!=entry['id'] or reference.get('use_in_current_gain_model')is not False or reference.get('use_in_current_risk_quantification')is not False:
        raise ValueError('Historical statistics must remain separate from current calibration')
    if hashlib.sha256(release_path.read_bytes()).hexdigest()!=reference['current_environment']['sha256']:
        raise ValueError('Historical-reference metadata belongs to a different environment')
    for ref in reference['references']:
        for key in ('influences','all_effects'):
            item=ref[key]
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Historical statistical reference changed')
    service.statistical_reference=reference
    service.intent_context_id=entry['id']
    service.evaluation_source_name=entry.get('label','统一 xApp 测试环境')+' · 共同起点'
    radio=base['radio']
    service.runtime_environment=dict(id=entry['id'],label=entry['label'],
        configuration=dict(cells=3,ues=6,carrier_frequency_ghz=radio['carrier_frequency_hz']/1e9,
            bandwidth_mhz=radio['bandwidth_hz']/1e6,ue_speed_mps=1.5,
            rt_max_depth=base['ray_solver']['max_depth'],traffic_load=recipe['traffic_load'],
            urllc_profile='20 ms ×6 / 1 s',mec_offload_fractions=dict(ue_0=.5,ue_4=.8)),
        origin_observe_slots=recipe['origin_observe_slots'],baseline_slot=recipe['baseline_slot'],
        validation=dict(six_apps_verified=True,observe_slots=release['observed_window']['observe_slots'],
            seeds=release['seeds'],report_url='/environment-report/'),
        statistics=dict(kind='uncalibrated',current_calibration=False,label='新环境增益待标定',
            reference_label='历史统计参考',reference_url='/contexts/'+reference['default_reference']+'/'))
    if active_release is not None:
        # The graph's old contract hashes remain the immutable ledger context.
        # Current runtime policy/model identities are a separate, explicit layer.
        from smo.model_release import energy_catalogue
        changed=[]
        for node in service.graph['nodes']:
            if node.get('kind')!='xapp':continue
            key=node['key'];model=active_release['models'][key]
            node['historical_contract_reference']={k:deepcopy(node.get(k))for k in
                ('model_sha256','evaluation_status','policy_evaluation','drx_profiles','runtime_parameters','algorithm')}
            if node.get('model_sha256')!=model['sha256']:changed.append(key)
            node.update(model_sha256=model['sha256'],runtime_parameters=deepcopy(model['runtime_parameters']),
                evaluation_status=model.get('evaluation_status','frozen_candidate_evaluation_retained'),
                runtime_model_release=active_release['release'],algorithm=model['algorithm'],
                runtime_implementation_variant=model.get('implementation_variant','legacy9'))
            node.pop('policy_evaluation',None)
            if key=='x01':
                version=recipe['model_release']['action_set_version']
                node['runtime_action_set_version']=version
                node['runtime_action_catalogue_sha256']=recipe['model_release']['action_catalogue_sha256']
                node['drx_profiles']=[deepcopy(a['drx'])for a in energy_catalogue(version)[::3]]
        service.runtime_environment['model_release']=deepcopy(recipe['model_release'])
        service.runtime_environment['validation']=dict(
            **selected_controller_validation(active_release,recipe,release,Path(__file__).resolve().parents[1]),
            retrained_or_replaced_models=sorted(active_release.get('updated_model_ids',changed)),
            unchanged_inherited_models=sorted(set(MODEL_IDS)-set(active_release.get('updated_model_ids',changed))),
            )
        service.runtime_environment['statistics']['current_calibration']=False
    from smo.current_calibration import verify_current_calibration
    current_calibration=verify_current_calibration(entry,recipe)
    if current_calibration is not None:
        service.runtime_environment['statistics']=current_calibration
    if entry.get('action_gain_supplement'):
        from smo.action_gain_supplement import apply
        from smo.gain_audit import audit_gains
        service.graph, supplement = apply(service.graph, entry, recipe)
        service.gain_audit = audit_gains(service.graph, all_kpis=True)
        service.primary_gain_audit = audit_gains(service.graph)
        service.runtime_environment['statistics']['action_gain_supplement'] = supplement
    service.environment_report_path=release_path.parent.parent/'report-v1/xapp-target-comparison.html'
    service.intent_migration=dict(from_source_name=entry.get('intent_migration',{}).get('source_name','共同评估起点'),
        record_ids=list(service.intent_ledger.records),token=entry['id']+':preserved-ledger-v1')
    run=Path(recipe['validated_origin']['run'])
    preview=PreviewStore(run=run).read(proof['baseline_slot'])
    if not preview.get('apps')or any(a.get('mode')!='cold_activation'for a in preview['apps']):
        raise ValueError('Validated environment has no recorded predeployment preview')
    validate_preview(recipe,preview)
    service.prepare_evaluation_origin(entry.get('prepared_origin_slots',500),_prepared=dict(
        proof=proof,frame=deepcopy(proof['before_frame']),preview=preview,run_path=str(run),source_job_id=proof['source_job_id']))


def prepare(manifest, *, language=None, staged_read_only=False):
    manifest=deepcopy(manifest); entries=manifest['scenarios']; default=manifest['default']
    ids=[c['id'] for c in entries]
    if not entries or len(ids)!=len(set(ids)) or default not in ids or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,40}', c) for c in ids):
        raise ValueError('Distinct known scenario IDs and a default are required')
    language=language or IntentLanguage(); presets=PresetStore(manifest.get('preset_store'))
    services={}; choices=[dict(id=c['id'],label=c['label'],url='/contexts/'+c['id']+'/',available=True) for c in entries]
    for entry in entries:
        matrix=json.loads(Path(entry['influences']).read_text())
        if matrix.get('scenario',{}).get('id')!=entry['id']:
            raise ValueError('A scene view cannot borrow another scene influence matrix')
        if entry['id']!='native_1x' and entry.get('deployment_config')and not entry.get('runtime_recipe'):
            raise ValueError('The retained native deployment pool cannot execute pressure scenes')
        service=GraphService(run=entry['replay'],influences=entry['influences'],view='population',
            max_frames=manifest.get('max_frames',160),language=language,intent_store=entry.get('intent_store'),
            deployment_config=None if staged_read_only else entry.get('deployment_config'),preset_store=manifest.get('preset_store'),
            empirical_response=entry.get('empirical_response'),
            candidate_selection=entry.get('candidate_selection',manifest.get('candidate_selection')),
            influence_selection=entry.get('influence_selection',manifest.get('influence_selection','all_evidence')),
            influence_confidence_level=entry.get('influence_confidence_level',manifest.get('influence_confidence_level',.95)),
            origin_source_archives=entry.get('origin_source_archives',manifest.get('origin_source_archives',[])))
        # Share one revision-checked memo store and one in-memory LLM session.
        service.presets=presets
        if entry.get('environment'):
            if entry.get('empirical_response'):
                raise ValueError('A new environment cannot silently reuse the old empirical model')
            if service.deployments.runtime_recipe and service.deployments.runtime_recipe['id']!=entry['id']:
                raise ValueError('Deployment pool belongs to a different environment')
            attach_environment(service,entry)
        if entry.get('retained_job_id')and not staged_read_only: service.deployments.view(entry['retained_job_id'])
        if entry.get('prepared_origin_slots')and not staged_read_only and not entry.get('environment'):
            service.prepare_evaluation_origin(entry['prepared_origin_slots'])
        service.scenario_view=dict(current=entry['id'],choices=choices,
            calibration=matrix['scenario'],matrix_sha256=service.graph['influence_input']['sha256'],
            traffic_contexts_pooled=False,manual_evaluation_enabled=bool(entry.get('deployment_config'))and not staged_read_only)
        services[entry['id']]=service
    return services,default


def router(services,default):
    handlers={key:handler(service) for key,service in services.items()}
    class Router(handlers[default]):
        def dispatch(self,method):
            parsed=urlparse(self.path); path=parsed.path; selected=default
            if path.startswith('/contexts/'):
                parts=path.split('/',3)
                if len(parts)<3 or parts[2] not in handlers:
                    self.send_error(404);return
                selected=parts[2]
                if len(parts)==3:
                    self.send_response(308);self.send_header('Location',path+'/');self.end_headers();return
                self.path='/'+parts[3]+(('?'+parsed.query) if parsed.query else '')
            try: getattr(handlers[selected],method)(self)
            finally:self.path=parsed.path+(('?'+parsed.query) if parsed.query else '')
        def do_GET(self):self.dispatch('do_GET')
        def do_POST(self):self.dispatch('do_POST')
    return Router


def retire_service(pid,port):
    """Retire only an explicitly named, verified local SMO listener."""
    if type(pid)is not int or pid<=1:raise ValueError('Expected an owned SMO process ID')
    args=shlex.split(subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True))
    modules=[args[i+1]for i,a in enumerate(args[:-1])if a=='-m']
    ports=[args[i+1]for i,a in enumerate(args[:-1])if a=='--port']
    if not any(m in('smo','smo.scenario_service')for m in modules)or not ports or ports[-1]!=str(port):
        raise ValueError('The selected process is not the SMO on the requested port')
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        with socket.socket()as connection:
            connection.settimeout(.1)
            if connection.connect_ex(('127.0.0.1',port))!=0:return
        time.sleep(.1)
    raise RuntimeError('The previous SMO port did not close')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--port',type=int,default=53505)
    p.add_argument('--session-key-stdin',action='store_true',help='Read a session-only key from a pipe, never an argument or file')
    p.add_argument('--verify-language',action='store_true',help='Verify one pure interval parse without writing any intent record')
    p.add_argument('--replace-pid',type=int,help='Retire this verified SMO only after read-only source and LLM preflight succeeds')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());language=IntentLanguage()
    if a.session_key_stdin:language.configure(sys.stdin.readline().strip(),manifest.get('llm_model'))
    if a.replace_pid is not None and not a.verify_language:p.error('Replacing a service requires --verify-language')
    services,default=prepare(manifest,language=language,staged_read_only=a.replace_pid is not None)
    if a.verify_language:
        service=services[default];_,frame=service.intent_context()
        try:language.parse('降低所有 UE 的总平均功率。',service.graph,validator=lambda raw:resolve_ranges(raw,service.graph,frame))
        except (IntentError,LlmUnavailable)as error:
            print(json.dumps(dict(ready=False,language_verification_error=str(error))),flush=True)
            raise SystemExit(1)
    if a.replace_pid is not None:
        retire_service(a.replace_pid,a.port)
        services,default=prepare(manifest,language=language)
    for ident,service in services.items():
        service.service_url=f'http://127.0.0.1:{a.port}/contexts/{ident}/'
        if manifest.get('export_dir'):export_html(service.bundle(),Path(manifest['export_dir'])/(ident+'-graph.html'))
    if manifest.get('canonical_export'):export_html(services[default].bundle(),Path(manifest['canonical_export']))
    server=ThreadingHTTPServer(('127.0.0.1',a.port),router(services,default))
    print(json.dumps(dict(ready=True,port=a.port,scenarios=list(services),llm=language.status())),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
