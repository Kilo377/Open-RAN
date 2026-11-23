"""Run the graph service with only Python's standard library.

python -m smo --replay outputs/run --port 8891
python -m smo --state outputs/run/live.json --port 8891
python -m smo --export outputs/smo/graph.html --export-only
"""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
from threading import RLock
from urllib.parse import urlparse, parse_qs

from smo.contracts import build_graph, validate_influences
from smo.runtime import LiveSource, replay
from smo.population import graph_view, SCOPE
from smo.intent_language import IntentLanguage, IntentError, LlmUnavailable, resolve_ranges
from smo.presets import PresetConflict
from smo.candidate_selection import select_combinations, options as selection_options, METHODS as SELECTION_METHODS
from smo.intent_ledger import IntentLedger, intersect_intents

WEB = Path(__file__).parent/'web'


def graph_intent_review(graph, frame, interpreted, candidate_selection=None, decision_context=None):
    """Find related controllers and structural graph risks without actions."""
    result = dict(schema='oran.intent-review.v2', ranges=deepcopy(interpreted['ranges']),
        questions=deepcopy(interpreted['questions']),
        snapshot={k:frame.get(k) for k in ('slot','time_ms')},
        analysis_basis='knowledge_graph', runtime_proposals_used=False,
        model_inference_invoked=False, hypothetical_actions_used=False,
        automatic_selection=False, deployment_enabled=False,
        note='KPI ranges and structural graph risks only. Each trained xApp owns its policy decisions; SMO does not select actions or alter model parameters.')
    result['invalidated_kpis'] = [r['kpi'] for r in result['ranges'] if r['kpi']=='ue_energy_j']
    # A saved constraint can block joint planning without invalidating the new
    # language parse. Keep its provenance so the UI marks the owning record.
    result['constraint_issues'] = []
    for row in result['ranges']:
        issue = None
        if row['kpi'] == 'ue_energy_j':
            issue = dict(kind='legacy_energy_requires_reparse',
                message='此旧意图使用累计 UE 能量（J）。请停用后重新解析为 UE 总功率（W）；原始记录和 J 基准保留。')
        elif row.get('empty'):
            issue = dict(kind='empty_intersection',
                message=f"{row.get('label', row['kpi'])} 的共同允许区间为空，请调整或停用对应意图。")
        if issue:
            result['constraint_issues'].append(dict(**issue, kpi=row['kpi'], blocking=True,
                intent_ids=list(dict.fromkeys(a['intent_id'] for a in row.get('anchors', [])))))
    result['planning_blocked'] = bool(result['questions'] or result['constraint_issues'])
    result['combination_catalog'] = select_combinations(graph, frame, result, show_dashed=False, settings=candidate_selection, decision_context=decision_context)
    result['combination_catalog_with_uncertain'] = select_combinations(graph, frame, result, show_dashed=True, settings=candidate_selection, decision_context=decision_context)
    from smo.graph_combinations import enumerate_combinations
    from smo.quantification_scope import quantification_catalog
    for show_dashed, suffix in ((False, ''), (True, '_with_uncertain')):
        structural = enumerate_combinations(graph, frame, result, show_dashed=show_dashed)
        result['quantification_catalog'+suffix] = quantification_catalog(
            result['combination_catalog'+suffix], 'all_covered', structural_catalog=structural)
    if result['combination_catalog'].get('direct_action_filter',{}).get('action_values_used'):
        result.update(runtime_proposals_used=True,analysis_basis='knowledge_graph_with_recorded_action_filter',
            note='Graph candidates with confirmed opposite recorded writes to one actuator removed; no hypothetical actions, KPI forecasts or deployment.')
    result['candidates'] = result['combination_catalog']['candidates']
    return result


class GraphService:
    def __init__(self, *, run=None, state=None, influences=None, max_frames=180, view='population', language=None, intent_store=None, deployment_config=None, preset_store=None, empirical_response=None, candidate_selection=None, influence_selection='all_evidence', influence_confidence_level=.95, origin_source_archives=None):
        if view not in ('population', 'entity'):
            raise ValueError('Expected population or entity view')
        self.candidate_selection = selection_options(candidate_selection)
        population = view == 'population'
        self.graph = build_graph()
        if population:
            self.graph = graph_view(self.graph)
        if influences:
            data = json.loads(Path(influences).read_text())
            scope = data.get('metric_scope')
            if (population and scope != SCOPE) or (not population and scope == SCOPE):
                raise ValueError('Influence KPI scope differs from the selected view; regenerate estimates or select --view entity for legacy target metrics.')
            if population and any(r.get('population_control_eligible') is not True for r in data.get('estimates', [])):
                raise ValueError('A single-member intervention cannot identify a common population control response')
            self.graph = validate_influences(self.graph, influences)
            matrix = Path(influences).with_name('all-effects.json')
            if population and matrix.exists() and data.get('measurement_dataset_sha256'):
                full = json.loads(matrix.read_text())
                if full.get('metric_scope') != SCOPE or any(r.get('data_sha256') != data['measurement_dataset_sha256'] or r.get('population_control_eligible') is not True for r in full.get('estimates', [])):
                    raise ValueError('The intent response matrix differs from the displayed calibration dataset.')
                indexed = {(r['parameter'],r['kpi'],r.get('contrast_id')):r for r in full['estimates']}
                if any(indexed.get((r['parameter'],r['kpi'],r.get('contrast_id'))) != r for r in data['estimates']):
                    raise ValueError('Published estimates are missing or changed in the intent response matrix.')
                published = {e['id'] for e in self.graph['edges']}
                merged = validate_influences(graph_view(build_graph()), matrix)
                for edge in merged['edges']:
                    if edge['id'] not in published:
                        edge['intent_only'] = True
                    if edge.get('target')=='kpi:urllc_failure_fraction':
                        edge.pop('intent_only',None)
                merged['influence_input'] = self.graph['influence_input']
                merged['intent_matrix_input'] = dict(name=matrix.name, sha256=hashlib.sha256(matrix.read_bytes()).hexdigest())
                self.graph = merged
        if influence_selection not in ('all_evidence','significant_only','significant_relations'):
            raise ValueError('Unknown influence selection policy')
        if influence_selection in ('significant_only','significant_relations'):
            from smo.statistical_filter import filter_graph
            self.graph=filter_graph(self.graph,confidence_level=influence_confidence_level,
                retain_measured_contexts=influence_selection=='significant_relations')
        # The unit projection must carry the very same effective interval as
        # the active graph, including an explicitly relaxed screening level.
        from smo.influence_units import align_graph
        self.graph = align_graph(self.graph)
        from smo.gain_audit import audit_gains
        self.gain_audit=audit_gains(self.graph,all_kpis=True)
        self.primary_gain_audit=audit_gains(self.graph)
        self.live = LiveSource(state, population=population) if state else None
        from smo.xapp_preview import PreviewStore
        self.preview_store = PreviewStore(run, state)
        self.empirical_response_path=Path(empirical_response)if empirical_response else None
        from smo.window_metrics import WindowIndex
        self.window_source = WindowIndex(json.loads((Path(run)/'config_snapshot.json').read_text()),population=population) if run else None
        self.window_run_id = None
        self.source = replay(run, max_frames, population=population,window_index=self.window_source) if run else dict(mode='contracts', name='Chongyu contracts',
            stage='not_attached', frames=[], read_only=True, note='Attach an actual run to populate dynamic state.')
        self.evaluation_source=None
        self.evaluation_origin=None
        self.evaluation_preview=None
        self.evaluation_run=None
        self.origin_source_archives=list(origin_source_archives or [])
        if deployment_config:self.origin_source_archives.append(str(Path(deployment_config).parent/'source.tgz'))
        self.origin_archive_cache={}
        self.lock = RLock()
        self.language = language if language is not None else IntentLanguage()
        from smo.presets import PresetStore
        self.presets=PresetStore(preset_store or (Path(intent_store).parent/'intent-presets.json'if intent_store else None))
        self.service_url = None
        from smo.deployment import DeploymentManager
        self.deployments = DeploymentManager(deployment_config)
        self.intent_ledger = IntentLedger(dict(metric_scope=self.graph.get('metric_scope'),
            contracts=self.graph.get('contract_sha256'), run=str(Path(run or state).resolve()) if run or state else None), intent_store)
        from smo.intent_comparison import IntentComparison
        self.comparisons = IntentComparison(self.intent_ledger.context,
            Path(intent_store).with_suffix('.comparisons.json') if intent_store else None)

    def bundle(self):
        with self.lock:
            source = self.evaluation_source or self.deployments.source or (self.live.read() if self.live else self.source)
            result=dict(graph=self.graph, graph_sha256=hashlib.sha256(json.dumps(self.graph,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                source=source, preset_data=self.presets.read(), deployment=self.deployments.status(), intent_service=self.language_status())
            from smo.graph_audit import audit_graph
            result['graph_completeness']=audit_graph(self.graph)
            result['gain_audit']=deepcopy(self.gain_audit)
            result['primary_gain_audit']=deepcopy(self.primary_gain_audit)
            result['candidate_selection']=deepcopy(self.candidate_selection)
            result['evaluation_origin']=dict(status='ready',origin_id=self.evaluation_origin['origin_id'],
                baseline_slot=self.evaluation_origin['baseline_slot'],observe_slots=self.evaluation_origin['observe_slots'])if self.evaluation_origin else dict(status='not_prepared')
            if hasattr(self,'scenario_view'):result['scenario_view']=deepcopy(self.scenario_view)
            for field in ('runtime_environment','statistical_reference','intent_migration','intent_context_id'):
                if hasattr(self,field):result[field]=deepcopy(getattr(self,field))
            return result

    def language_status(self):
        return {**self.language.status(),'url':self.service_url,'conflict_input_schema':'oran.trusted-control-chain-inputs.v1'}

    def intent_context(self, slot=None):
        bundle = self.bundle()
        if self.graph.get('metric_scope') != SCOPE:
            raise IntentError('Intent matching requires the population graph view.')
        frames = bundle['source']['frames']
        if slot is not None and type(slot) is not int:
            raise IntentError('Select a valid simulation snapshot.')
        if frames:
            frame = frames[-1] if bundle['source']['mode'] in ('live','snapshot') else next((f for f in frames if f['slot'] == slot), None) if slot is not None else frames[-1]
            if frame is None:
                raise IntentError('This simulation snapshot is no longer available. Refresh and try again.')
        else:
            frame = dict(slot=None, time_ms=None, states={})
        return bundle, frame

    def review_intents(self, active_intent_ids, slot=None):
        bundle, frame = self.intent_context(slot)
        records = self.intent_ledger.select(active_intent_ids)
        result = self.planning_review(bundle,frame,intersect_intents(records,self.graph,frame))
        result.update(active_intents=records, source_name=bundle['source']['name'])
        return result

    def preview_xapps(self, slot=None):
        with self.lock:
            bundle, frame = self.intent_context(slot)
            source=bundle['source']
            source_id=source.get('run_id')or source.get('source_sha256')or source['name']
            if self.evaluation_source and frame['slot']==self.evaluation_origin['baseline_slot']:
                result=deepcopy(self.evaluation_preview)
            elif self.deployments.run:
                from smo.xapp_preview import PreviewStore
                result=PreviewStore(run=self.deployments.run).read(frame['slot'])
            else:result=self.preview_store.read(frame['slot'])
            graph_sha=bundle.get('graph_sha256')or hashlib.sha256(json.dumps(self.graph,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            return {**result,'source_id':source_id,'graph_sha256':graph_sha}

    def prepare_evaluation_origin(self,observe_slots=500,*,_prepared=None):
        """Reuse a certified BEFORE prefix; never borrow an AFTER measurement."""
        if not self.deployments.config and _prepared is None:raise IntentError('此场景尚未配置部署评估设备。')
        if type(observe_slots)is not int or not 1<=observe_slots<=20000:
            raise IntentError('评估窗口须为 1–20,000 个整数 slots。')
        from smo.evaluation_reference import prepare_origin
        prepared=deepcopy(_prepared)
        if prepared is not None:
            from smo.evaluation_origin_guard import validate_proof
            validate_proof(prepared['proof'])
        fixed_origin=getattr(self.deployments,'runtime_origin',None)
        if prepared is None and fixed_origin is not None:
            from smo.evaluation_origin_guard import validate_proof
            from smo.xapp_preview import PreviewStore
            proof=deepcopy(fixed_origin);validate_proof(proof)
            run=self.deployments.runtime_origin_run
            preview=PreviewStore(run=run).read(proof['baseline_slot'])
            if not preview.get('apps')or any(a.get('mode')!='cold_activation'for a in preview['apps']):
                raise IntentError('新环境缺少共同起点的真实预部署决策记录。')
            prepared=dict(proof=proof,frame=deepcopy(proof['before_frame']),preview=preview,
                run_path=str(run),source_job_id=proof['source_job_id'])
        for batch in ([]if prepared else sorted(self.deployments.batch.items.values(),key=lambda b:b['created_unix_s'],reverse=True)):
            if batch['seed']!=self.deployments.config['seed']:continue
            pin=batch.get('worker_source_archive_sha256')
            for filename in self.origin_source_archives:
                path=Path(filename)
                if path.exists() and filename not in self.origin_archive_cache:
                    self.origin_archive_cache[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
            archive=next((p for p,d in self.origin_archive_cache.items()if d==pin),None)
            candidate=prepare_origin(batch,self.deployments.output,observe_slots=observe_slots,source_archive=archive)
            if candidate.get('status')=='ready':prepared=candidate;break
        if prepared is None:raise IntentError('没有与此观察窗口匹配的已保存部署前基线；需先采集一次共同前缀，不能用部署后状态替代。')
        proof=prepared['proof'];frame=prepared['frame']
        # The certified prefix has no active controllers. Its delayed public
        # control observations therefore remain valid at the baseline. Fill
        # missing display fields from recorded CURRENT values, never proposals;
        # keep the worker's independently verified proof frame untouched.
        from smo.evaluation_reference import comparable
        for parameter,entity_key in (('scheduler_mode','cell_id'),('ue_scheduler_weights','ue_id')):
            observed={};ambiguous=set()
            for app in prepared['preview'].get('apps',[]):
                for action in app.get('actions',[]):
                    entity=action.get(entity_key)
                    if action.get('parameter')!=parameter or not entity or action.get('current')is None:continue
                    if entity in observed and not comparable(observed[entity],action['current']):ambiguous.add(entity)
                    observed[entity]=deepcopy(action['current'])
            observed={k:v for k,v in observed.items()if k not in ambiguous}
            state=frame['states'].get('rcp:'+parameter,{})
            if observed and not state.get('by_entity'):
                values=list(observed.values())
                frame['states']['rcp:'+parameter]=dict(state,value=values[0]if all(comparable(v,values[0])for v in values)else None,
                    by_entity=observed,status='observed',source='recorded_predeployment_public_control_observation',
                    decision_slot=prepared['preview'].get('slot'),no_after_used=True)
        frame['evaluation_origin']=dict(origin_id=proof['origin_id'],baseline_slot=proof['baseline_slot'],observe_slots=proof['observe_slots'])
        with self.lock:
            self.evaluation_origin=proof;self.evaluation_preview=prepared['preview'];self.evaluation_run=Path(prepared['run_path'])
            self.evaluation_source=dict(mode='snapshot',name=getattr(self,'evaluation_source_name','共同评估起点'),stage='prepared',frames=[frame],
                run_id='evaluation-origin:'+proof['origin_id'],total_slots=proof['baseline_slot'],slot_duration_ms=.5,
                read_only=True,evaluation_origin_id=proof['origin_id'],source_job_id=prepared['source_job_id'],
                note='已保存的真实部署前窗口；重用共同前缀，无新增仿真。',after_kpis_used=False)
        return self.bundle()

    def evaluation_context(self,frame,observe_slots):
        origin=self.evaluation_origin
        fixed_origin=getattr(self.deployments,'runtime_origin',None)
        valid_horizon=type(observe_slots)is int and 1<=observe_slots<=20000
        if not origin or frame.get('evaluation_origin',{}).get('origin_id')!=origin['origin_id'] or not valid_horizon or (fixed_origin is None and origin['observe_slots']!=observe_slots):
            raise IntentError('请先点击“对齐评估起点”，在同一观察窗口的共同基线上解析或更新意图；原意图区间不会被改写。')
        context=dict(deepcopy(self.intent_ledger.context),evaluation_origin=deepcopy(origin))
        if fixed_origin is not None:
            context['runtime_environment']=dict(id=self.deployments.runtime_recipe['id'],
                baseline_slot=origin['baseline_slot'],origin_observe_slots=origin['observe_slots'])
        return context

    def selection_decision_context(self,bundle,frame):
        """Earlier recorded decisions use their own frame; never future actions."""
        at=frame.get('slot');frames=bundle.get('source',{}).get('frames',[])
        preview=self.preview_xapps(at)if type(at)is int else dict(status='unavailable')
        decision=next((f for f in frames if f.get('slot')==preview.get('slot')),None)
        if preview.get('slot')==at:decision=frame
        if preview.get('status')=='unavailable' and type(at)is int:
            run=self.evaluation_run or self.deployments.run or self.preview_store.run
            if run:
                from smo.runtime import records as recorded_rows
                from smo.xapp_preview import PreviewStore
                visible={f['slot']:f for f in frames if f.get('slot',at+1)<=at}
                slots=[p['slot']for p in recorded_rows(run,'xapp_decision_previews')if p.get('slot')in visible]
                if slots:
                    selected=max(slots);preview=PreviewStore(run).read(selected);decision=visible[selected]
        if type(preview.get('slot'))is not int or type(at)is not int or preview['slot']>at:
            preview=dict(status='unavailable');decision=None
        return dict(preview=preview,frame=decision or {},requested_slot=at,
            lag_ms=frame.get('time_ms',0)-decision.get('time_ms',0)if decision else None)

    def planning_review(self,bundle,frame,interpreted):
        return graph_intent_review(self.graph,frame,interpreted,self.candidate_selection,
            self.selection_decision_context(bundle,frame))

    def conflict_inputs(self,active_intent_ids,slot=None,observe_slots=500,show_dashed=False,group_scope='deployment_candidates'):
        # Keep internal origin/source transitions out of a trusted packet.
        # A live publisher does not share this lock, so also verify that its
        # atomic telemetry file stayed unchanged during frame/preview capture.
        with self.lock,self.deployments.lock:
            stamp=self.live.path.stat().st_mtime_ns if self.live else None
            return self._captured_conflict_inputs(active_intent_ids,slot,observe_slots,show_dashed,group_scope,live_stamp=stamp)

    def _captured_conflict_inputs(self,active_intent_ids,slot=None,observe_slots=500,show_dashed=False,group_scope='deployment_candidates',*,live_stamp=None):
        """Trusted control-chain inputs, independent of arbiters/KPI forecasting."""
        if type(show_dashed)is not bool or type(observe_slots)is not int or not 1<=observe_slots<=20000:
            raise IntentError('量化观察窗口须为 1–20,000 slots。')
        from smo.quantification_scope import quantification_catalog, SCOPES
        if group_scope not in SCOPES:raise IntentError('请选择已定义的量化组合范围。')
        bundle,frame=self.intent_context(slot)
        preview=self.preview_xapps(frame['slot'])
        source_id=bundle['source'].get('run_id')or bundle['source'].get('source_sha256')or bundle['source']['name']
        changed_live=self.live and self.live.path.stat().st_mtime_ns!=live_stamp
        changed_source=preview.get('source_id',source_id)!=source_id
        changed_graph=bundle.get('graph_sha256')and preview.get('graph_sha256',bundle['graph_sha256'])!=bundle['graph_sha256']
        if changed_live or changed_source or changed_graph:
            return dict(method='trusted_control_chain_inputs_v1',schema='oran.trusted-control-chain-inputs.v1',
                status='unavailable',reason='network_snapshot_changed_during_capture',groups=[],
                snapshot={k:frame.get(k)for k in('slot','time_ms')},preview_slot=preview.get('slot'),
                source_id=source_id,graph_sha256=bundle.get('graph_sha256'))
        selected=self.intent_ledger.select(active_intent_ids)
        review=graph_intent_review(self.graph,frame,intersect_intents(selected,self.graph,frame),self.candidate_selection,
            dict(preview=preview,frame=frame,requested_slot=frame['slot'],lag_ms=0))
        catalog_key='combination_catalog_with_uncertain'if show_dashed else'combination_catalog'
        catalog=(review['quantification_catalog_with_uncertain'if show_dashed else'quantification_catalog']
            if group_scope=='all_covered' else quantification_catalog(review[catalog_key],group_scope))
        result=dict(method='trusted_control_chain_inputs_v1',schema='oran.trusted-control-chain-inputs.v1',
            status='ready',snapshot={k:frame.get(k)for k in('slot','time_ms')},preview_slot=preview.get('slot'),
            active_intent_ids=[r['id']for r in selected],
            source_id=source_id,
            graph_sha256=bundle.get('graph_sha256'),
            groups=[],group_scope=group_scope,quantification_scope=deepcopy(catalog['quantification_scope']),
            report_slots=observe_slots,kpi_forecast_performed=False,arbitration_performed=False,
            observed_after_kpis_used=False,selection_performed=False)
        if preview.get('slot')!=frame['slot'] or preview.get('status')=='unavailable':
            from smo.runtime import records as recorded_rows
            run=self.evaluation_run or self.deployments.run or self.preview_store.run
            visible={f['slot']for f in bundle['source']['frames']}
            slots=[p['slot']for p in recorded_rows(run,'xapp_decision_previews')if p.get('slot')in visible and p['slot']<=frame['slot']]if run else[]
            result.update(status='unavailable',reason='matching_decision_preview_unavailable',suggested_slot=max(slots)if slots else None)
            return result
        ranges=catalog.get('ranges',review.get('ranges',[]))
        run=self.evaluation_run or self.deployments.run or self.preview_store.run
        config=json.loads((Path(run)/'config_snapshot.json').read_text())if run else{}
        ric=config.get('configs',{}).get('ric',{})
        result['arbitration_reference']=dict(mode=ric.get('direct_conflict_mitigation','uniform_random'),
            window_ms=ric.get('mitigation_window_ms',100),held_state_reconstructed=False,
            source='recorded_configuration'if run else'project_defaults')
        result['groups']=[dict(group_id=g.get('id',g.get('group_id')),app_ids=g['app_ids'],
            deployment_disposition=g['deployment_disposition'],deployment_eligible=g['deployment_eligible'],
            conflict_summaries=deepcopy(g.get('conflict_summaries',[])),
            kpis=[dict(range=deepcopy(r))for r in ranges])for g in catalog['groups']]
        from smo.influence_units import normalization
        # One frozen intent reference across all candidate groups. No deployment
        # batch, measured BEFORE/AFTER result, or outcome-dependent scale is read.
        result['influence_normalization']=normalization(ranges,frame)
        result.update(deployment_results_used=False,deployment_batch_used=False,
            quantification_basis='current_intent_graph_and_recorded_proposals')
        payload=dict(graph=self.graph,frame=frame,catalog=catalog,preview=preview,
            active_intent_ids=result['active_intent_ids'],show_dashed=show_dashed,
            arbitration_reference=result['arbitration_reference'],influence_normalization=result['influence_normalization'])
        result['input_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        # The browser consumes this exact packet, rather than fetching another
        # preview and accidentally combining it with a different same-slot KPI.
        result['decision_frame']=deepcopy(frame)
        result['decision_preview']={**deepcopy(preview),'source_id':source_id,'graph_sha256':bundle.get('graph_sha256')}
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        result['decision_frame_sha256']=digest(result['decision_frame'])
        result['decision_preview_sha256']=digest(result['decision_preview'])
        result['snapshot_packet_schema']='oran.frozen-control-snapshot.v1'
        result['snapshot_packet_sha256']=digest({key:result[key]for key in('source_id','graph_sha256','decision_frame_sha256','decision_preview_sha256')})
        return result

    def quantify_conflicts(self,active_intent_ids,slot=None,observe_slots=500,show_dashed=False,group_scope='deployment_candidates'):
        """Current, read-only method; no client-supplied actions or gains."""
        if type(show_dashed)is not bool or type(observe_slots)is not int or not 1<=observe_slots<=20000:
            raise IntentError('量化观察窗口须为 1–20,000 slots。')
        from smo.quantification_scope import quantification_catalog, SCOPES
        if group_scope not in SCOPES:raise IntentError('请选择已定义的量化组合范围。')
        bundle,frame=self.intent_context(slot)
        records=self.intent_ledger.select(active_intent_ids)
        review=self.planning_review(bundle,frame,intersect_intents(records,self.graph,frame))
        catalog_key='combination_catalog_with_uncertain'if show_dashed else'combination_catalog'
        catalog=(review['quantification_catalog_with_uncertain'if show_dashed else'quantification_catalog']
            if group_scope=='all_covered' else quantification_catalog(review[catalog_key],group_scope))
        from smo.conflict_quantification import quantify_catalog
        run=self.evaluation_run or self.deployments.run or self.preview_store.run
        config=json.loads((Path(run)/'config_snapshot.json').read_text())if run else{}
        empirical=None
        if self.empirical_response_path:
            from smo.mec_empirical import load_model
            empirical=load_model(self.empirical_response_path)
        result=quantify_catalog(self.graph,frame,catalog,self.preview_xapps(frame['slot']),config=config,
            horizon_ms=observe_slots*bundle['source'].get('slot_duration_ms',.5),show_dashed=show_dashed,empirical_model=empirical)
        if result.get('reason')=='matching_decision_preview_unavailable'and run:
            from smo.runtime import records as recorded_rows
            visible={f['slot']for f in bundle['source']['frames']}
            slots=[p['slot']for p in recorded_rows(run,'xapp_decision_previews')if p.get('slot')in visible and p['slot']<=frame['slot']]
            result['suggested_slot']=max(slots)if slots else None
        result['active_intent_ids']=[r['id']for r in records]
        result['group_scope']=group_scope
        result['quantification_scope']=deepcopy(catalog['quantification_scope'])
        result['source_id']=bundle['source'].get('run_id')or bundle['source'].get('source_sha256')or bundle['source']['name']
        return result

    def start_comparison(self, active_intent_ids, slot=None, observe_slots=500):
        bundle, frame = self.intent_context(slot)
        records = self.intent_ledger.select(active_intent_ids)
        ident = self.comparisons.start(records, self.graph, frame, observe_slots,
            bundle['source'].get('run_id') or bundle['source'].get('source_sha256'))
        with self.lock:
            return self.comparisons.review(ident, self.graph, bundle['source'], frame,windows=self.comparison_windows(bundle))

    def review_comparison(self, comparison_id, slot=None):
        bundle, frame = self.intent_context(slot)
        with self.lock:
            return self.comparisons.review(comparison_id, self.graph, bundle['source'], frame,windows=self.comparison_windows(bundle))

    def comparison_windows(self, bundle):
        if self.deployments.source and bundle['source'].get('deployment_job_id'):
            return self.deployments.windows
        if self.live:
            from smo.window_metrics import RecordedWindows
            token=bundle['source'].get('run_id')
            if self.window_run_id != token:
                self.window_source=RecordedWindows(self.live.path.parent,self.live.config)
                self.window_run_id=token
        if self.window_source is not None:self.window_source.sync()
        return self.window_source

    def deploy_combination(self, active_intent_ids, group_id, slot=None, observe_slots=500, show_dashed=False):
        if type(show_dashed)is not bool or not isinstance(group_id,str):raise IntentError('请选择一个候选组合。')
        bundle,frame=self.intent_context(slot)
        records=self.intent_ledger.select(active_intent_ids)
        if not records:raise IntentError('请先解析意图。')
        review=self.planning_review(bundle,frame,intersect_intents(records,self.graph,frame))
        catalog=review['combination_catalog_with_uncertain'if show_dashed else 'combination_catalog']
        group=next((g for g in catalog['groups']if g['id']==group_id),None)
        if group is None:raise IntentError('此组合已被过滤、未覆盖全部目标 KPI，或观测已变化；请更新观测并重新选择。')
        names={n['id']:n['label']for n in self.graph['nodes']if n['kind']=='xapp'}
        return self.deployments.start(records=records,group={**group,'selected_names':[names[a]for a in group['app_ids']]},
            observe_slots=observe_slots,context=self.evaluation_context(frame,observe_slots))

    def evaluate_all_combinations(self, active_intent_ids, slot=None, observe_slots=500, show_dashed=False):
        if type(show_dashed)is not bool:raise IntentError('视图选项无效。')
        bundle,frame=self.intent_context(slot);records=self.intent_ledger.select(active_intent_ids)
        if not records:raise IntentError('请先解析意图。')
        review=self.planning_review(bundle,frame,intersect_intents(records,self.graph,frame))
        catalog=review['combination_catalog_with_uncertain'if show_dashed else 'combination_catalog']
        names={n['id']:n['label']for n in self.graph['nodes']if n['kind']=='xapp'}
        groups=[dict(g,selected_names=[names[a]for a in g['app_ids']])for g in catalog['groups']]
        return self.deployments.batch.start(records=records,groups=groups,observe_slots=observe_slots,
            context=self.evaluation_context(frame,observe_slots),snapshot=dict(slot=frame['slot'],source_id=bundle['source'].get('run_id')or bundle['source'].get('source_sha256'),influence_sha256=self.graph.get('influence_input',{}).get('sha256')),
            ranges=review['ranges'],show_dashed=show_dashed)

    def analyze_intent(self, text, slot=None, active_intent_ids=None):
        bundle, frame = self.intent_context(slot)
        decision_context=self.selection_decision_context(bundle,frame)
        records = self.intent_ledger.select([] if active_intent_ids is None else active_intent_ids)
        if len(records) >= 32:
            raise IntentError('Select up to 32 distinct saved intents.')
        # Freeze the snapshot before the external language request. No actions run.
        def validate(raw):
            parsed=resolve_ranges(raw,self.graph,frame)
            if any(r['kpi']=='ue_energy_j' for r in parsed['ranges']):
                raise IntentError('UE 节能意图请使用 ue_power_w（W）；累计 ue_energy_j（J）仅供历史记录，不能作为需要下降的当前值。')
            return parsed
        if hasattr(self.language,'parse_at_snapshot'):
            interpreted,provenance=self.language.parse_at_snapshot(text,self.graph,validator=validate,snapshot=frame)
        else:
            interpreted,provenance=self.language.parse(text,self.graph,validator=validate)
        if not interpreted['questions']:
            # Validate the union before appending. Empty intersections are valid
            # review results and remain removable constraints, not failed parses.
            pending = dict(id='pending', snapshot={k:frame.get(k) for k in ('slot','time_ms')}, ranges=interpreted['ranges'])
            intersect_intents(records+[pending], self.graph, frame)
            records.append(self.intent_ledger.add(text, frame, interpreted, provenance))
        combined = intersect_intents(records, self.graph, frame)
        combined['questions'] = interpreted['questions']
        result = graph_intent_review(self.graph,frame,combined,self.candidate_selection,decision_context)
        result['active_intents'] = records
        result['language'] = provenance
        result['source_name'] = bundle['source']['name']
        return result


def export_html(bundle, path):
    """A real offline artifact, with CSS, JS and recorded frames embedded."""
    html = (WEB/'index.html').read_text()
    css, js = (WEB/'style.css').read_text(), (WEB/'app.js').read_text()
    encoded = json.dumps(bundle, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    html = html.replace('<link rel="stylesheet" href="./style.css">', '<style>'+css+'</style>')
    html = html.replace('<script src="./goal_obstruction.js" defer></script>',
        '<script>'+(WEB/'goal_obstruction.js').read_text()+'</script>')
    html = html.replace('<script src="./intent_obstruction.js" defer></script>',
        '<script>'+(WEB/'intent_obstruction.js').read_text()+'</script>')
    html = html.replace('<script src="./app.js" defer></script>',
        '<script>window.SMO_BOOTSTRAP='+encoded+';</script><script>'+js+'</script>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path.resolve()


def handler(service):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ('/api/conflicts/inputs', '/api/intent', '/api/intent/review', '/api/intent/session-key', '/api/intent/clear-key', '/api/xapps/preview', '/api/conflicts/quantify', '/api/intent/comparison/start', '/api/intent/comparison/review', '/api/deployment/start', '/api/deployment/stop', '/api/deployment/reconnect', '/api/deployment/batch/start', '/api/deployment/batch/pause', '/api/deployment/batch/resume', '/api/deployment/view', '/api/evaluation/origin', '/api/intent/presets'):
                self.send_error(501)
                return
            allowed = {'http://127.0.0.1:'+str(self.server.server_port),
                       'http://localhost:'+str(self.server.server_port)}
            if self.headers.get('Origin') and self.headers['Origin'] not in allowed:
                self.send_bytes(b'{"error":"Use the local SMO page for intent requests."}', 'application/json', 403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= (131072 if path=='/api/intent/presets'else 16384) or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise IntentError('Expected a small JSON request.')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise IntentError('Expected a JSON object.')
                if path in ('/api/conflicts/quantify','/api/conflicts/inputs'):
                    if set(data)-{'active_intent_ids','slot','observe_slots','show_dashed','group_scope'}:
                        raise IntentError('冲突量化仅接受已保存意图、快照、观察窗口和虚线选项。')
                    method=service.conflict_inputs if path.endswith('/inputs')else service.quantify_conflicts
                    result=method(data.get('active_intent_ids'),data.get('slot'),data.get('observe_slots',500),data.get('show_dashed',False),data.get('group_scope','deployment_candidates'))
                elif path == '/api/evaluation/origin':
                    if set(data)-{'observe_slots'}:raise IntentError('起点准备仅接受观察窗口长度。')
                    result=service.prepare_evaluation_origin(data.get('observe_slots',500))
                elif path == '/api/intent/presets':
                    if set(data)!={'revision','edits','deleted_ids'}:raise IntentError('预设保存仅接受版本、编辑内容和删除记录。')
                    result=service.presets.save(data['revision'],data['edits'],data['deleted_ids'])
                elif path == '/api/deployment/batch/start':
                    if set(data)-{'active_intent_ids','slot','observe_slots','show_dashed'}:
                        raise IntentError('全部组合评估不接受单组 ID、组合子集、动作、命令或参数值。')
                    result=service.evaluate_all_combinations(data.get('active_intent_ids'),data.get('slot'),data.get('observe_slots',500),data.get('show_dashed',False))
                elif path in ('/api/deployment/batch/pause','/api/deployment/batch/resume'):
                    if set(data)!={'batch_id'}:raise IntentError('此操作仅接受批次 ID。')
                    method=service.deployments.batch.pause if path.endswith('/pause')else service.deployments.batch.resume
                    result=method(data['batch_id'])
                elif path == '/api/deployment/view':
                    if set(data)!={'job_id'}:raise IntentError('查看实测仅接受任务 ID。')
                    with service.lock:
                        result=service.deployments.view(data['job_id'])
                        service.evaluation_source=None;service.evaluation_origin=None;service.evaluation_preview=None;service.evaluation_run=None
                elif path == '/api/deployment/start':
                    if set(data)-{'active_intent_ids','group_id','slot','observe_slots','show_dashed'}:
                        raise IntentError('部署仅接受意图 ID、候选组合和窗口长度；不能提交参数动作或命令。')
                    result=service.deploy_combination(data.get('active_intent_ids'),data.get('group_id'),data.get('slot'),data.get('observe_slots',500),data.get('show_dashed',False))
                elif path == '/api/deployment/stop':
                    if set(data)!={'job_id'}:raise IntentError('停止操作仅接受本次任务 ID。')
                    result=service.deployments.cancel(data['job_id'])
                elif path == '/api/deployment/reconnect':
                    if data:raise IntentError('重新连接不接受附加参数。')
                    result=service.deployments.reconnect()
                elif path == '/api/intent/comparison/start':
                    if set(data) - {'active_intent_ids','slot','observe_slots'}:
                        raise IntentError('对比仅接受意图 ID、快照和观察 slots；不能传入动作或伪造 KPI。')
                    result = service.start_comparison(data.get('active_intent_ids'), data.get('slot'), data.get('observe_slots',500))
                elif path == '/api/intent/comparison/review':
                    if set(data) - {'comparison_id','slot'}:
                        raise IntentError('对比更新仅接受已保存的对比 ID 和快照。')
                    result = service.review_comparison(data.get('comparison_id'),data.get('slot'))
                elif path == '/api/xapps/preview':
                    if set(data) - {'slot'}:
                        raise IntentError('Decision preview accepts a snapshot slot only; no actions or parameters.')
                    result = service.preview_xapps(data.get('slot'))
                elif path == '/api/intent':
                    if set(data) - {'text', 'slot', 'active_intent_ids'}:
                        raise IntentError('Intent requests accept text, snapshot slot and saved intent IDs only.')
                    result = service.analyze_intent(data.get('text'), data.get('slot'), data.get('active_intent_ids'))
                elif path == '/api/intent/review':
                    if set(data) - {'slot', 'active_intent_ids'}:
                        raise IntentError('Review accepts saved intent IDs and snapshot slot only.')
                    result = service.review_intents(data.get('active_intent_ids'), data.get('slot'))
                elif path == '/api/intent/session-key':
                    if set(data) - {'api_key', 'model'}:
                        raise IntentError('Unexpected language configuration field.')
                    service.language.configure(data.get('api_key'), data.get('model'))
                    result = service.language_status()
                else:
                    if data:
                        raise IntentError('The clear-key request must be empty.')
                    service.language.clear_key()
                    result = service.language_status()
                self.send_bytes(json.dumps(result, allow_nan=False).encode(), 'application/json')
            except PresetConflict as error:
                self.send_bytes(json.dumps(dict(error=str(error))).encode(),'application/json',409)
            except IntentError as error:
                self.send_bytes(json.dumps(dict(error=str(error))).encode(), 'application/json', 400)
            except (ValueError, TypeError, UnicodeError):
                self.send_bytes(json.dumps(dict(error='Invalid intent or interval. Check the KPI, bounds and current snapshot.')).encode(), 'application/json', 400)
            except LlmUnavailable as error:
                self.send_bytes(json.dumps(dict(error=str(error))).encode(), 'application/json', 503)
            except Exception:
                self.send_bytes(b'{"error":"Intent review failed; no network control was changed."}', 'application/json', 500)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/api/graph':
                    self.send_bytes(json.dumps(service.bundle(), allow_nan=False).encode(), 'application/json')
                elif url.path == '/api/intent/presets':
                    self.send_bytes(json.dumps(service.presets.read(),ensure_ascii=False).encode(),'application/json')
                elif url.path == '/api/deployment':
                    self.send_bytes(json.dumps(service.deployments.status(),allow_nan=False).encode(),'application/json')
                elif url.path == '/api/deployment/batch/results':
                    query=parse_qs(url.query);ident=query.get('batch_id',[''])[0];format=query.get('format',['json'])[0]
                    raw,mime=service.deployments.batch.export(ident,format)
                    self.send_bytes(raw,mime)
                elif url.path == '/api/deployment/result':
                    ident=parse_qs(url.query).get('job_id',[''])[0]
                    self.send_bytes(service.deployments.result_file(ident).read_bytes(),'text/html')
                elif url.path == '/health':
                    self.send_bytes(json.dumps(dict(service='oran-intent-graph',read_only=not bool(service.deployments.config),manual_deployment_enabled=bool(service.deployments.config))).encode(),'application/json')
                elif url.path == '/api/intent/status':
                    self.send_bytes(json.dumps(service.language_status()).encode(), 'application/json')
                elif url.path in ('/environment-report/','/environment-report') and hasattr(service,'environment_report_path'):
                    self.send_bytes(Path(service.environment_report_path).read_bytes(),'text/html')
                elif url.path in ('/', '/index.html', '/style.css', '/app.js', '/goal_obstruction.js', '/intent_obstruction.js'):
                    name = 'index.html' if url.path == '/' else url.path[1:]
                    mime = {'index.html': 'text/html', 'style.css': 'text/css', 'app.js': 'text/javascript', 'goal_obstruction.js': 'text/javascript', 'intent_obstruction.js': 'text/javascript'}[name]
                    self.send_bytes((WEB/name).read_bytes(), mime)
                elif url.path == '/favicon.ico':
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)
            except (OSError, ValueError, KeyError) as error:
                self.send_bytes(json.dumps(dict(error=str(error))).encode(), 'application/json', 503)

        def send_bytes(self, data, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime+'; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--replay', type=Path, help='Completed run directory, read-only')
    source.add_argument('--state', type=Path, help='Atomic live.json from the existing simulator')
    parser.add_argument('--influences', type=Path, help='Audited controlled-perturbation estimates JSON')
    parser.add_argument('--view', choices=('population','entity'), default='population', help='Population monitoring (default) or legacy target/entity metrics')
    parser.add_argument('--max-frames', type=int, default=180)
    parser.add_argument('--intent-store', help='Optional JSON ledger retaining immutable intent anchors')
    parser.add_argument('--preset-store',type=Path,help='Durable user preset text and memo tags, shared across browser sessions')
    parser.add_argument('--empirical-response',type=Path,help='Optional operator-owned, validated observational MEC response model')
    parser.add_argument('--deployment-config',type=Path,help='Operator-controlled SSH simulation worker JSON; UI can start only validated combinations')
    parser.add_argument('--candidate-selection',choices=SELECTION_METHODS,default='reverse_graph')
    parser.add_argument('--coverage-threshold',type=float,default=.05)
    parser.add_argument('--marginal-tolerance',type=float,default=.05)
    parser.add_argument('--capability-catalog',type=Path,help='Optional audited frozen-policy secondary capability evidence')
    parser.add_argument('--port', type=int, default=8891)
    parser.add_argument('--language-broker',help='Reuse a loopback LLM interval service; resolve original symbolic ranges locally without extracting its session key')
    parser.add_argument('--export', type=Path, help='Write a self-contained HTML with this graph/data')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.max_frames <= 1000:
        parser.error('--max-frames must be between 2 and 1000')
    if args.export_only and not args.export:
        parser.error('--export-only requires --export')
    from smo.intent_relay import LocalIntentRelay
    language=LocalIntentRelay(args.language_broker)if args.language_broker else None
    service = GraphService(run=args.replay, state=args.state, influences=args.influences, max_frames=args.max_frames, view=args.view,language=language,intent_store=args.intent_store,deployment_config=args.deployment_config,preset_store=args.preset_store,empirical_response=args.empirical_response,candidate_selection=dict(method=args.candidate_selection,coverage_threshold=args.coverage_threshold,marginal_tolerance=args.marginal_tolerance,capability_catalog=str(args.capability_catalog) if args.capability_catalog else None))
    if language:language.verify_context(service.bundle())
    service.service_url = 'http://127.0.0.1:'+str(args.port)+'/'
    if args.export:
        print('Exported:', export_html(service.bundle(), args.export), flush=True)
    if args.export_only:
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(service))
    print('Intent graph: http://127.0.0.1:'+str(server.server_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
