"""Read-only structural scoring remains available after deployment rejection.

These tests use the real population contracts, selector, recorded-preview reader,
HTTP handler and browser risk engine. No inference, simulator or SSH is invoked.
"""
from copy import deepcopy
from hashlib import sha256
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from smo.__main__ import GraphService, graph_intent_review, handler
from smo.intent_language import IntentError, resolve_ranges
from smo.intent_ledger import intersect_intents
from smo.quantification_scope import quantification_catalog
from smo.xapp_preview import PreviewStore

ROOT = Path(__file__).resolve().parents[1]
PAIR = {'xapp:ue_energy_rf', 'xapp:mec_offload_budget_qgbdt'}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def before_service(tmp_path, monkeypatch):
    service = GraphService(candidate_selection={'method': 'semantic_influence_coverage_v2'})
    frame = dict(slot=2040, time_ms=1020, states={
        'kpi:ue_power_w': {'value': 5.25},
        'kpi:mec_expected_gflops': {'value': 14.},
        'rcp:ue_offload_fraction': {'by_entity': {'ue_0': .8, 'ue_4': .8}},
        'rcp:ue_drx_configs': {'by_entity': {
            uid: dict(enabled=True, cycle_ms=40, on_duration_ms=4, inactivity_ms=8)
            for uid in ('ue_0', 'ue_4')}}})
    preview = dict(schema='oran.xapp-decision-preview.v1', status='ready', slot=2040,
        time_ms=1020, deployment_enabled=False, automatic_selection=False, apps=[
            dict(app_id='xapp:' + app, status='proposed', inference_invoked=True,
                mode='cold_activation', observation_slot=2040, model_sha256=pin * 64,
                action_contract_failures=[], actions=[dict(parameter='ue_offload_fraction',
                    cell_id='cell_2', ue_id='ue_4', current=.8, proposed=value)])
            for app, value, pin in [('ue_energy_rf', 1., 'a'), ('mec_offload_budget_qgbdt', .5, 'b')]])
    preview_path = tmp_path / 'xapp_decision_previews.jsonl'
    preview_path.write_text(json.dumps(preview) + '\n')
    (tmp_path / 'config_snapshot.json').write_text(json.dumps(dict(configs=dict(ric=dict(
        direct_conflict_mitigation='uniform_random', mitigation_window_ms=100)))))
    service.preview_store = PreviewStore(run=tmp_path)
    source = dict(mode='recorded', name='captured BEFORE only', run_id='same-slot-before-source',
        source_sha256='c' * 64, slot_duration_ms=.5, frames=[frame], after_kpis_used=False)
    # Supply a captured source boundary, rather than making the bundle producer
    # fetch unrelated live job/status metadata while exercising scoring below.
    def context(slot=None):
        if slot not in (None, frame['slot']):
            raise IntentError('Unavailable fixture snapshot')
        return dict(source=source, graph=service.graph, graph_sha256=digest(service.graph)), frame
    monkeypatch.setattr(service, 'intent_context', context)
    parsed = resolve_ranges(dict(questions=[], ranges=[dict(kpi=key, lower=0, upper=upper,
        lower_inclusive=True, upper_inclusive=False)
        for key, upper in [('ue_power_w', 5.), ('mec_expected_gflops', 13.)]]), service.graph, frame)
    record = service.intent_ledger.add('Lower UE power and MEC demand', frame, parsed, {'provider': 'fixture'})
    return SimpleNamespace(service=service, frame=frame, preview=preview, preview_path=preview_path,
        record=record, ids=[record['id']], source=source, run=tmp_path)


def review(fixture):
    return fixture.service.review_intents(fixture.ids, fixture.frame['slot'])


@pytest.fixture
def endpoint(before_service):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(before_service.service))
    thread = Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    def post(path, payload):
        request = Request(f'http://127.0.0.1:{server.server_port}' + path,
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            return json.load(response)
    try:
        yield post
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_real_direct_rejection_preserves_independent_structural_object(before_service):
    fixture = before_service
    original = deepcopy((fixture.service.graph, fixture.frame, fixture.preview, fixture.service.intent_ledger.records))
    result = review(fixture)
    admitted = result['combination_catalog']
    assert admitted['group_count'] == 0 and admitted['groups'] == []
    assert len(admitted['excluded_groups']) == 1
    quantified = result['quantification_catalog']
    assert quantified['group_count'] == 1
    group = quantified['groups'][0]
    assert set(group['app_ids']) == PAIR
    assert group['deployment_disposition'] == 'excluded' and group['deployment_eligible'] is False
    assert group['action_exclusion'] == admitted['excluded_groups'][0]['action_exclusion']
    assert any(row['type'] == 'direct' and row['overlap_members'] == ['ue_4'] for row in group['conflicts'])
    assert quantified['quantification_scope']['deployment_admission_changed'] is False
    assert (fixture.service.graph, fixture.frame, fixture.preview, fixture.service.intent_ledger.records) == original


def test_initial_selector_cannot_erase_structural_scoring_objects(before_service, monkeypatch):
    # Simulate a future initial-admission heuristic which drops rows entirely.
    # A union of the admitted/reserve/excluded lists would incorrectly be empty.
    from smo import __main__ as main
    original_selector = main.select_combinations
    def discard_all(*args, **kwargs):
        catalog = original_selector(*args, **kwargs)
        catalog.update(groups=[], reserve_groups=[], excluded_groups=[], group_count=0)
        return catalog
    monkeypatch.setattr(main, 'select_combinations', discard_all)
    result = review(before_service)
    assert not any(result['combination_catalog'].get(key) for key in ('groups', 'reserve_groups', 'excluded_groups'))
    group, = result['quantification_catalog']['groups']
    assert set(group['app_ids']) == PAIR
    assert group['deployment_disposition'] == 'not_selected' and group['deployment_eligible'] is False


def test_default_scope_keeps_existing_deployment_candidate_semantics(before_service):
    fixture = before_service
    # Same-direction writes leave this real pair admissible, so compatibility
    # is checked on a nonempty list rather than on two trivially empty outputs.
    preview = deepcopy(fixture.preview)
    for app in preview['apps']:
        app['actions'][0]['proposed'] = .9
    fixture.preview_path.write_text(json.dumps(preview) + '\n')
    result = review(fixture); admitted = result['combination_catalog']; before = deepcopy(admitted)
    assert admitted['group_count'] == 1
    default = fixture.service.conflict_inputs(fixture.ids, 2040)
    explicit = fixture.service.conflict_inputs(fixture.ids, 2040, group_scope='deployment_candidates')
    assert default == explicit and default['group_scope'] == 'deployment_candidates'
    assert {g['group_id'] for g in default['groups']} == {g['id'] for g in admitted['groups']}
    assert default['groups'][0]['deployment_disposition'] == 'candidate'
    assert default['groups'][0]['deployment_eligible'] is True
    assert quantification_catalog(admitted) == quantification_catalog(admitted, 'deployment_candidates')
    assert admitted == before


def test_real_http_packet_keeps_excluded_group_same_slot_and_pins(before_service, endpoint):
    fixture = before_service
    payload = dict(active_intent_ids=fixture.ids, slot=2040, observe_slots=500)
    legacy = endpoint('/api/conflicts/inputs', payload)
    packet = endpoint('/api/conflicts/inputs', {**payload, 'group_scope': 'all_covered'})
    assert legacy['groups'] == [] and legacy['group_scope'] == 'deployment_candidates'
    group, = packet['groups']
    assert set(group['app_ids']) == PAIR
    assert group['deployment_disposition'] == 'excluded' and group['deployment_eligible'] is False
    assert packet['group_scope'] == 'all_covered' and packet['status'] == 'ready'
    assert packet['snapshot']['slot'] == packet['preview_slot'] == packet['decision_frame']['slot'] == packet['decision_preview']['slot'] == 2040
    assert packet['decision_frame'] == fixture.frame
    assert packet['source_id'] == packet['decision_preview']['source_id'] == fixture.source['run_id']
    assert packet['graph_sha256'] == packet['decision_preview']['graph_sha256'] == digest(fixture.service.graph)
    assert packet['decision_frame_sha256'] == digest(packet['decision_frame'])
    assert packet['decision_preview_sha256'] == digest(packet['decision_preview'])
    assert packet['snapshot_packet_sha256'] == digest({key: packet[key] for key in (
        'source_id', 'graph_sha256', 'decision_frame_sha256', 'decision_preview_sha256')})
    assert [app['model_sha256'] for app in packet['decision_preview']['apps']] == ['a' * 64, 'b' * 64]
    frozen = deepcopy(packet)
    fixture.frame['states']['kpi:ue_power_w']['value'] = 999
    fixture.preview['apps'][0]['actions'][0]['proposed'] = 0
    assert packet == frozen


def test_quantifying_excluded_rows_never_readmits_single_or_batch_deployment(before_service, monkeypatch):
    fixture = before_service; service = fixture.service
    group, = service.conflict_inputs(fixture.ids, 2040, group_scope='all_covered')['groups']
    monkeypatch.setattr(service.deployments, 'start', lambda **kwargs: pytest.fail('Excluded pair was deployed'))
    with pytest.raises(IntentError, match='过滤'):
        service.deploy_combination(fixture.ids, group['group_id'], 2040)
    fixture.frame['evaluation_origin'] = {'origin_id': 'before-only-origin'}
    service.evaluation_origin = dict(origin_id='before-only-origin', observe_slots=500)
    captured = []
    monkeypatch.setattr(service.deployments.batch, 'start', lambda **kwargs: captured.append(kwargs) or kwargs)
    service.evaluate_all_combinations(fixture.ids, 2040)
    assert len(captured) == 1 and captured[0]['groups'] == []
    assert service.review_intents(fixture.ids, 2040)['combination_catalog']['groups'] == []


def test_read_only_scope_never_reads_after_results_or_batch(before_service, monkeypatch):
    fixture = before_service; service = fixture.service
    expected = service.conflict_inputs(fixture.ids, 2040, group_scope='all_covered')
    class ForbiddenResults:
        def __getattribute__(self, name):
            raise AssertionError('Quantification inspected deployment/AFTER state: ' + name)
    service.deployments.batch = ForbiddenResults()
    service.comparisons = ForbiddenResults()
    service.deployments.jobs = ForbiddenResults()
    forbidden = fixture.run / 'comparison.json'
    forbidden.write_text('{"after":{"ue_power_w":-999},"baseline_reference":{"ue_power_w":1e-10}}')
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == forbidden:
            pytest.fail('Observed AFTER was read to score current structural groups')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    actual = service.conflict_inputs(fixture.ids, 2040, group_scope='all_covered')
    assert actual == expected
    assert actual['deployment_results_used'] is actual['deployment_batch_used'] is actual['observed_after_kpis_used'] is False
    assert actual['selection_performed'] is False


@pytest.mark.parametrize('route', ['/api/conflicts/inputs', '/api/conflicts/quantify'])
@pytest.mark.parametrize('extra', [{'group_scope': 'everything'}, {'group_scope': None},
    {'actions': [{'parameter': 'ue_offload_fraction', 'proposed': .1}]}, {'gains': [999]},
    {'group_ids': ['injected-group']}])
def test_http_scope_and_client_action_trust_boundary(before_service, endpoint, route, extra):
    request = dict(active_intent_ids=before_service.ids, slot=2040, observe_slots=500,
        group_scope='all_covered')
    with pytest.raises(HTTPError) as error:
        endpoint(route, {**request, **extra})
    assert error.value.code == 400


def test_all_covered_requires_an_independent_catalog():
    with pytest.raises(ValueError, match='structural'):
        quantification_catalog({'groups': []}, 'all_covered')
    with pytest.raises(ValueError, match='scope'):
        quantification_catalog({'groups': []}, 'unknown')


def test_browser_really_scores_uncalibrated_excluded_groups_as_unknown(before_service):
    node = shutil.which('node')
    if not node:
        pytest.skip('Production browser risk engine requires Node')
    fixture = before_service; service = fixture.service
    primary = service.conflict_inputs(fixture.ids, 2040, group_scope='all_covered')
    uncertain = service.conflict_inputs(fixture.ids, 2040, show_dashed=True, group_scope='all_covered')
    result = review(fixture)
    assert result['combination_catalog']['group_count'] == 0
    assert len(primary['groups']) == 1 and len(uncertain['groups']) == 24
    source = (ROOT / 'smo/web/app.js').read_text()
    quantification = source[source.index('  let conflictQuantification='):source.index('  const quantificationReasons=')]
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const data=JSON.parse(fs.readFileSync(0,'utf8')),pending=[],calls=[],c={structuredClone};
vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
vm.runInContext(fs.readFileSync('smo/web/goal_obstruction.js','utf8'),c);
c.state={bundle:{graph:data.graph,graph_sha256:data.primary.graph_sha256,
 source:{run_id:data.primary.source_id,frames:[data.primary.decision_frame]},intent_service:{conflict_input_schema:'v1'}},
 intent:data.review,intentBusy:false,activeIntents:[{id:data.id}],combinationCatalog:data.review.combination_catalog,showDashed:false};
c.frame=()=>data.primary.decision_frame;c.location={protocol:'http:'};c.render=()=>{};
c.$=id=>{if(id==='comparison-slots')throw Error('Deployment horizon leaked into scoring');return {value:'goal_obstruction_v2'};};
for(const name of ['activeBatch','matchingBatch','managedComparison','uncalibratedEnvironment'])
 c[name]=()=>{throw Error('Unexpected result/filter dependency: '+name);};
c.setTimeout=fn=>{pending.push(fn);return pending.length;};c.clearTimeout=()=>{};
c.postIntent=async(url,request)=>{calls.push({url,request});return structuredClone(request.show_dashed?data.uncertain:data.primary);};
const actualEngine=c.SmoGoalObstruction.review;let invocations=0;
c.SmoGoalObstruction.review=(...args)=>{invocations++;return actualEngine(...args);};
vm.runInContext(data.quantification,c);
(async()=>{
 for(const [showDashed,count]of [[false,1],[true,24]]){
  c.state.showDashed=showDashed;
  vm.runInContext('ensureConflictQuantification(true)',c);
  assert.equal(pending.length,1);await pending.shift()();
  const result=vm.runInContext('currentQuantification()',c);
  assert.equal(result.group_scope,'all_covered');assert.equal(result.groups.length,count);
  assert.ok(result.groups.every(group=>group.score===null));
  assert.ok(result.groups.every(group=>group.score_status==='insufficient_evidence'));
  assert.equal(result.deployment_results_used,false);assert.equal(result.deployment_batch_used,false);
  assert.equal(c.state.combinationCatalog.groups.length,0);
 }
 assert.equal(invocations,2);assert.equal(calls.length,2);
 assert.ok(calls.every(call=>call.url==='./api/conflicts/inputs'&&call.request.group_scope==='all_covered'&&call.request.observe_slots===500));
 process.stdout.write(JSON.stringify({invocations,groups:[1,24],unknown_scores_preserved:true}));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    completed = subprocess.run([node, '-e', script], cwd=ROOT, input=json.dumps(dict(
        graph=service.graph, review=result, primary=primary, uncertain=uncertain,
        id=fixture.ids[0], quantification=quantification)), text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)['unknown_scores_preserved'] is True
