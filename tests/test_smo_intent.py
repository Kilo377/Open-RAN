"""Analytic fixtures test the planner. They are never presented as LLM run data."""
from copy import deepcopy
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from smo.__main__ import GraphService, handler, export_html
from smo.contracts import build_graph, validate_influences
from smo.population import graph_view
from smo.grid_identify import fit_grid
from smo.intent_language import IntentLanguage, IntentError, LlmUnavailable, resolve_ranges
from smo.intent_match import match_intent, profiles


def interval(kpi, lower='minimum', upper='current', li=True, ui=False):
    return dict(kpi=kpi, lower=lower, upper=upper, lower_inclusive=li, upper_inclusive=ui)


def snapshot():
    states={}
    for k,v in dict(ue_energy_j=3.,ue_power_w=6.,network_goodput_mbps=30.,mec_expected_gflops=1.,urllc_failure_fraction=0.,task_success_ratio=1.).items():
        states['kpi:'+k]=dict(value=v)
    states['rcp:ue_drx_configs']=dict(value=None,by_entity={u:dict(enabled=True,cycle_ms=40.,on_duration_ms=4.,inactivity_ms=8.,wakeup_ms=.5,offset_ms=0.)for u in('ue_0','ue_4')})
    for k,values in dict(ue_offload_fraction={'ue_0':.5,'ue_4':.5},ue_scheduler_weights={u:1. for u in ('ue_0','ue_1','ue_2','ue_4','ue_6','ue_8')},
                         scheduler_mode={'cell_0':'round_robin','cell_1':'round_robin','cell_2':'round_robin'},
                         cell_sleep_mode={'cell_0':'active','cell_1':'active','cell_2':'active'}).items():
        states['rcp:'+k]=dict(value=next(iter(values.values())),by_entity=values)
    return dict(slot=1600,time_ms=800,states=states)


def graph_with_gains(tmp_path):
    levels=[0.,.2,.4,.5,.6,.8,1.];seeds=[1,2,3]
    g=dict(id='offload_population_range',parameter='ue_offload_fraction',reference=.5,levels=levels,
           arms=[str(x)for x in levels],mode='round_robin',component='mec_workload',x_unit='fraction',scope='Analytic fixture only')
    lookup={(s,str(x)):dict(metrics=dict(ue_energy_j=3-x,mec_expected_gflops=2*x))for s in seeds for x in levels}
    rows=[fit_grid(g,seeds,lookup,k,'a'*64)for k in ('ue_energy_j','mec_expected_gflops')]
    p=tmp_path/'gains.json';p.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=rows)))
    return validate_influences(graph_view(build_graph()),p)


def test_ranges_distinguish_increase_from_do_not_decrease_and_apply_domains():
    g=graph_view(build_graph());f=snapshot()
    raw=dict(ranges=[interval('network_goodput_mbps','current','maximum',False,True)],questions=[])
    r=resolve_ranges(raw,g,f)['ranges'][0]
    assert r['lower']==30 and r['upper']is None and r['needed_direction']=='increase'
    raw['ranges'][0]['lower_inclusive']=True
    assert resolve_ranges(raw,g,f)['ranges'][0]['status']=='within_range'
    raw['ranges']=[interval('task_success_ratio','current','maximum',False,True)]
    assert resolve_ranges(raw,g,f)['ranges'][0]['empty']is True


def test_explicit_bands_relative_targets_and_intersections():
    g=graph_view(build_graph());f=snapshot()
    raw=dict(ranges=[interval('network_goodput_mbps',{'current_multiplier':1.2},'maximum',True,True),
                     interval('network_goodput_mbps',20,40,True,True)],questions=[])
    r=resolve_ranges(raw,g,f)['ranges'][0]
    assert (r['lower'],r['upper'])==(36,40)
    assert r['needed_direction']=='increase'
    raw['ranges'].append(interval('network_goodput_mbps',0,25,True,True))
    assert resolve_ranges(raw,g,f)['ranges'][0]['empty']is True


@pytest.mark.parametrize('row',[
    interval('invented_kpi'),interval('ue_energy_j',float('nan'),3),
    {**interval('ue_energy_j'),'xapp':'invented'},
    {**interval('ue_energy_j'),'lower_inclusive':'false'},
    interval(['ue_energy_j']),
])
def test_unknown_or_malformed_llm_ranges_are_rejected(row):
    with pytest.raises(IntentError):resolve_ranges(dict(ranges=[row],questions=[]),graph_view(build_graph()),snapshot())


def test_graph_matching_checks_action_direction_bounds_locked_paths_and_guard_effects(tmp_path):
    g=graph_with_gains(tmp_path);f=snapshot();before=deepcopy((g,f))
    ranges=resolve_ranges(dict(ranges=[interval('ue_energy_j'),interval('mec_expected_gflops','minimum','current',True,True)],questions=[]),g,f)
    plan=match_intent(g,f,ranges)
    x01=next(c for c in plan['candidates']if c['app_id']=='xapp:ue_energy_rf')
    assert x01['variants'][0]['actions'][0]['after']==1
    assert any(e['kpi']=='mec_expected_gflops'and e['range_risk']for e in x01['variants'][0]['side_effects'])
    assert any(c['app_id']=='xapp:mec_offload_budget_qgbdt'for c in plan['candidates'])
    assert any(a['parameter']=='ue_drx_configs'for c in plan['candidates']for v in c['variants']for a in v['actions'])
    assert not any(e['parameter']=='rcp:ue_drx_configs'for e in plan['excluded_paths'])
    assert all(e['parameter']!='ue_drx_configs'for c in plan['candidates']for v in c['variants']for m in v['matches']for e in m['paths'])
    assert plan['deployment_enabled']is False and (g,f)==before
    assert 'influence:ue_offload_fraction:ue_energy_j'in x01['variants'][0]['path_edge_ids']


def test_subset_budget_action_is_not_assigned_the_population_gain_magnitude(tmp_path):
    g=graph_with_gains(tmp_path);f=snapshot()
    ranges=resolve_ranges(dict(ranges=[interval('mec_expected_gflops')],questions=[]),g,f)
    p=match_intent(g,f,ranges);x13=next(c for c in p['candidates']if c['app_id']=='xapp:mec_offload_budget_qgbdt')
    assert x13['variants'][0]['matches'][0]['paths'][0]['scope_approximation']is True
    assert 'predicted_kpi'not in x13['variants'][0]
    assert all(v['actions'][0]['after']in (0,.5)for v in x13['variants'])
    ranges=resolve_ranges(dict(ranges=[interval('mec_expected_gflops','current','maximum',False,True)],questions=[]),g,f)
    # Relation to a KPI is independent of the policy's minimize label; its
    # ability to satisfy this increasing range remains unverified.
    result=match_intent(g,f,ranges)
    assert any(c['app_id']=='xapp:mec_offload_budget_qgbdt'for c in result['candidates'])
    assert result['deployment_enabled'] is False


def test_satisfied_ranges_and_questions_do_not_force_actions(tmp_path):
    g=graph_with_gains(tmp_path);f=snapshot()
    ranges=resolve_ranges(dict(ranges=[interval('ue_power_w',0,10,True,True)],questions=[]),g,f)
    result=match_intent(g,f,ranges)
    assert result['candidates'] and result['deployment_enabled'] is False
    ranges['questions']=['Which energy metric?']
    assert match_intent(g,f,ranges)['candidates']==[]


def test_efficiency_policy_does_not_inherit_a_service_class_gain(tmp_path):
    g=graph_with_gains(tmp_path);levels=[.5,.75,1.,1.25,1.5,2.];seeds=[1,2,3]
    grid=dict(id='weight_mec_population_range',parameter='ue_scheduler_weights',reference=1.,levels=levels,
              arms=[str(x)for x in levels],mode='latency_aware',component='mec_priority',x_unit='weight',scope='Analytic fixture only')
    lookup={(s,str(x)):dict(metrics=dict(network_goodput_mbps=30+2*x))for s in seeds for x in levels}
    row=fit_grid(grid,seeds,lookup,'network_goodput_mbps','a'*64)
    path=tmp_path/'priority.json';path.write_text(json.dumps(dict(schema='oran.intervention-edges.v1',estimates=[row])))
    g=validate_influences(g,path);f=snapshot()
    ranges=resolve_ranges(dict(ranges=[interval('network_goodput_mbps','current','maximum',False,True)],questions=[]),g,f)
    plan=match_intent(g,f,ranges);x08=next(c for c in plan['candidates']if c['app_id']=='xapp:throughput_scheduler_mlp')
    assert all(not m['paths']and m['status']=='contract_only'for v in x08['variants']for m in v['matches'])


def test_complete_matrix_keeps_supplementary_edges_intent_only_and_rejects_drift(tmp_path):
    graph_with_gains(tmp_path)
    data=json.loads((tmp_path/'gains.json').read_text());data.update(metric_scope='ue_population.v1',measurement_dataset_sha256='a'*64)
    published=tmp_path/'influences.json';published.write_text(json.dumps(data))
    extra=deepcopy(data['estimates'][0]);extra.update(kpi='ap_power_w',unit='W / fraction')
    full={**data,'estimates':data['estimates']+[extra]};matrix=tmp_path/'all-effects.json';matrix.write_text(json.dumps(full))
    service=GraphService(influences=published)
    edge=next(e for e in service.graph['edges']if e['id']=='influence:ue_offload_fraction:ap_power_w')
    assert edge['intent_only']is True
    assert not next(e for e in service.graph['edges']if e['id']=='influence:ue_offload_fraction:ue_energy_j').get('intent_only')
    full['estimates'][0]['estimate']=999;matrix.write_text(json.dumps(full))
    with pytest.raises(ValueError):GraphService(influences=published)


def test_catalog_contains_the_released_action_families_and_qos_v2():
    apps={a['key']:a for a in build_graph()['nodes']if a['kind']=='xapp'}
    assert len(apps)==6 and all(profiles(a)for a in apps.values())
    assert len(profiles(apps['x01']))==9
    assert all(any(p['parameter']=='ue_drx_configs'for p in v['parts'])for v in profiles(apps['x01']))
    qos=profiles(apps['x12_urllc'])
    assert [v['parts'][1]['value']for v in qos]==[1,1,2,4]
    assert all(v['parts'][0]['value']=='qos_deadline'for v in qos[1:])
    assert {v['parts'][0]['value']for v in profiles(apps['x13'])}=={0,.5}


def test_language_layer_calls_real_client_interface_and_never_keyword_falls_back(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    with pytest.raises(LlmUnavailable):IntentLanguage().parse('降低能耗',graph_view(build_graph()))
    seen={}
    class Client:
        def generate_json_or_raise(self,**prompts):
            seen.update(prompts);return dict(ranges=[interval('ue_energy_j')],questions=[])
    def factory(**kwargs):seen['config']=kwargs;return Client()
    language=IntentLanguage(factory);raw,provenance=language.parse('降低UE能耗',graph_view(build_graph()))
    assert raw['ranges'][0]['kpi']=='ue_energy_j'and provenance['interpreted_by']=='llm'
    assert seen['config']['provider']=='deepseek'
    assert 'current'in seen['system_prompt']and 'xApp'in seen['system_prompt']
    assert '降低UE能耗'in seen['user_prompt']


@pytest.mark.parametrize('key', ['mec_offloaded_latency_ms','task_latency_ms'])
def test_language_catalog_explains_both_task_populations_and_keeps_model_kpi(key):
    seen={}
    class Client:
        def generate_json_or_raise(self,**prompts):
            seen.update(prompts);return dict(ranges=[interval(key)],questions=[])
    graph=graph_view(build_graph());language=IntentLanguage(lambda **kw:Client())
    raw,provenance=language.parse('Reduce the requested task E2E latency.',graph)
    assert raw['ranges'][0]['kpi']==key
    catalog={n['kpi']:n for n in json.loads(seen['user_prompt'])['kpi_catalog']}
    assert 'offloaded only' in catalog['mec_offloaded_latency_ms']['label']
    assert 'completed offloaded tasks' in catalog['mec_offloaded_latency_ms']['aggregation']
    assert 'local + offloaded' in catalog['task_latency_ms']['label']
    assert 'completed tasks' in catalog['task_latency_ms']['aggregation']
    assert 'Both are E2E metrics' in seen['system_prompt']
    assert 'MEC task delay' in catalog['mec_offloaded_latency_ms']['aliases']
    assert 'all-task E2E latency' in catalog['task_latency_ms']['aliases']
    assert 'explicit all-task scope takes precedence' in seen['system_prompt']
    assert provenance['kpi_semantics_version']=='mec-task-populations.v2'


def test_extra_top_level_commentary_is_not_an_action_or_range():
    raw=dict(ranges=[interval('ue_energy_j')],questions=[],commentary='Untrusted explanation',xapps=['invented'])
    parsed=resolve_ranges(raw,graph_view(build_graph()),snapshot())
    assert set(parsed)=={'ranges','questions'}and parsed['ranges'][0]['kpi']=='ue_energy_j'


def test_malformed_llm_shape_gets_one_validated_format_retry():
    calls=[]
    class Client:
        def generate_json_or_raise(self,**kw):
            calls.append(json.loads(kw['user_prompt']))
            return dict(targets=[])if len(calls)==1 else dict(ranges=[interval('ue_energy_j')],questions=[])
    settings={}
    def factory(**kw):settings.update(kw);return Client()
    g=graph_view(build_graph());language=IntentLanguage(factory)
    result,provenance=language.parse('降低UE能耗',g,validator=lambda raw:resolve_ranges(raw,g,snapshot()))
    assert len(calls)==2 and 'format_correction'in calls[1]
    assert calls[0]['user_intent']==calls[1]['user_intent']=='降低UE能耗'
    assert result['ranges'][0]['needed_direction']=='decrease'and provenance['format_retries']==1
    assert settings['extra_body']['response_format']=={'type':'json_object'}
    assert settings['extra_body']['thinking']=={'type':'disabled'}


def test_second_invalid_llm_output_stops_without_a_fallback():
    calls=[]
    class Client:
        def generate_json_or_raise(self,**kw):calls.append(kw);return dict(targets=[])
    g=graph_view(build_graph())
    with pytest.raises(IntentError):IntentLanguage(lambda **kw:Client()).parse('test',g,validator=lambda raw:resolve_ranges(raw,g,snapshot()))
    assert len(calls)==2


def test_existing_deepseek_client_sends_json_mode_and_disables_thinking_on_wire():
    from ric.non_rt.ollama_gptoss_client import StructuredJsonLlmClient
    seen=[]
    class ProviderFixture(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            answer=dict(ranges=[interval('ue_energy_j')],questions=[])
            body=json.dumps(dict(choices=[dict(message=dict(content=json.dumps(answer)))])).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    provider=ThreadingHTTPServer(('127.0.0.1',0),ProviderFixture)
    thread=Thread(target=provider.serve_forever,daemon=True);thread.start()
    try:
        def factory(**kwargs):
            kwargs['base_url']='http://127.0.0.1:'+str(provider.server_port)
            kwargs['api_key']='unit-provider-fixture-only'
            return StructuredJsonLlmClient(**kwargs)
        graph=graph_view(build_graph())
        parsed,provenance=IntentLanguage(factory).parse('降低UE能耗',graph,validator=lambda raw:resolve_ranges(raw,graph,snapshot()))
        assert parsed['ranges'][0]['needed_direction']=='decrease'and provenance['format_retries']==0
        assert len(seen)==1
        assert seen[0]['response_format']=={'type':'json_object'}
        assert seen[0]['thinking']=={'type':'disabled'}
    finally:provider.shutdown();provider.server_close();thread.join()


def test_read_only_intent_http_and_session_secret_never_enter_exports(tmp_path,monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    class Client:
        def generate_json_or_raise(self,**kw):return dict(ranges=[interval('ue_power_w')],questions=[])
    language=IntentLanguage(lambda **kw:Client())
    service=GraphService(language=language)
    service.graph=graph_with_gains(tmp_path)
    service.source=dict(mode='recorded',name='Analytic test fixture',frames=[snapshot()])
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(service));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def post(path,data,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        return urlopen(Request(base+path,data=json.dumps(data).encode(),headers=headers,method='POST'))
    try:
        with post('/api/intent/session-key',dict(api_key='unit-test-secret-only',model='model-test'))as r:assert json.load(r)['configured']is True
        page=export_html(service.bundle(),tmp_path/'graph.html').read_text()
        assert 'unit-test-secret-only'not in page and 'model-test'in page
        before=deepcopy(service.source)
        with post('/api/intent',dict(text='A fixture request',slot=1600))as r:
            plan=json.load(r)
        assert plan['language']['interpreted_by']=='llm'and plan['candidates']
        assert plan['deployment_enabled']is False and service.source==before
        with pytest.raises(HTTPError)as e:post('/api/intent',dict(text='test'),origin='https://foreign.invalid')
        assert e.value.code==403
        with pytest.raises(HTTPError)as e:post('/api/graph',{})
        assert e.value.code==501
        with post('/api/intent/clear-key',{})as r:assert json.load(r)['configured']is False
        assert all(n.get('state')is None for n in service.graph['nodes'])
    finally:server.shutdown();server.server_close();thread.join()


def test_unlocked_joint_drx_profile_does_not_borrow_native_drx_offload_calibration(tmp_path):
    graph=graph_with_gains(tmp_path);frame=snapshot()
    for v in frame['states']['rcp:ue_drx_configs']['by_entity'].values():
        v.update(cycle_ms=80.,inactivity_ms=1.)
    ranges=resolve_ranges(dict(ranges=[interval('ue_energy_j')],questions=[]),graph,frame)
    result=match_intent(graph,frame,ranges)
    effects=[e for c in result['candidates']for v in c['variants']for m in v['matches']for e in m['paths']]
    assert effects and all(e.get('context_approximation') for e in effects)
    assert all(e['scope_approximation'] for e in effects)
