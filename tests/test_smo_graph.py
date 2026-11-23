"""Contract provenance, time alignment and execution-vs-proposal semantics."""
from copy import deepcopy
import gzip
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from threading import Thread
from types import SimpleNamespace as NS
from urllib.error import HTTPError
from urllib.request import urlopen, Request
from http.server import ThreadingHTTPServer

import pytest

from smo.contracts import build_graph, validate_influences, APP_SPECS
from smo.runtime import FrameBuilder, LiveSource, replay,trace_users
from smo.telemetry import compact_telemetry
from smo.__main__ import GraphService, export_html, handler


def test_graph_uses_released_six_contracts_without_inventing_identified_effects():
    graph = build_graph()
    assert {n['key'] for n in graph['nodes'] if n['kind'] == 'xapp'} == set(APP_SPECS)
    assert len([n for n in graph['nodes'] if n['kind'] == 'rcp']) == 5
    ids = [n['id'] for n in graph['nodes']]
    assert len(ids) == len(set(ids))
    assert 'kpi:reference_sinr_db' not in ids  # read-only predictor, not a declared effect
    assert all(e['source'] in ids and e['target'] in ids for e in graph['edges'])
    candidates = [e for e in graph['edges'] if e['kind'] == 'influences']
    assert all(e['sign'] == 'unknown' and e['sensitivity'] is None and e['causal_status'] == 'unidentified' for e in candidates)
    assert next(e for e in graph['edges'] if e['source'] == 'xapp:ue_energy_rf' and e['target'] == 'rcp:ue_drx_configs')['deployment'] == 'controllable'
    urllc = next(e for e in graph['edges'] if e['source'] == 'xapp:latency_urllc_rf' and e['target'] == 'rcp:scheduler_mode')
    assert urllc['declared_values'] == ['round_robin', 'qos_deadline']
    assert len(graph['contract_sha256']) >= 14


def estimate_file(tmp_path, **changes):
    # Analytic test fixture only. Never included in the product graph.
    row = dict(parameter='ue_offload_fraction', kpi='ue_energy_j', estimate=-.2,
        unit='J / fraction', kind='local_derivative', sample_count=12, horizon_ms=100,
        scope='ue_4', context='test fixture, fixed load/seed/other controls',
        method='paired finite difference', data_sha256='a'*64, design='controlled_perturbation',
        confidence_interval=[-.3,-.1])
    row.update(changes)
    path = tmp_path/'influences.json'
    path.write_text(json.dumps(dict(schema='oran.intervention-edges.v1', estimates=[row])))
    return path


def test_controlled_estimate_import_preserves_units_context_and_original(tmp_path):
    original = build_graph(); before = deepcopy(original)
    updated = validate_influences(original, estimate_file(tmp_path))
    assert original == before
    edge = next(e for e in updated['edges'] if e.get('sensitivity'))
    assert edge['sign'] == 'negative' and edge['evidence'] == 'controlled_perturbation'
    assert edge['sensitivity']['scope'] == 'ue_4' and edge['sensitivity']['horizon_ms'] == 100


@pytest.mark.parametrize('changes',[
    dict(sample_count=1), dict(estimate=float('nan')), dict(design='observational'),
    dict(parameter='scheduler_mode'), dict(data_sha256='invalid'),
    dict(confidence_interval=[-.1,-.3]), dict(scope=''), dict(horizon_ms=-1),
])
def test_unidentified_or_invalid_estimates_cannot_masquerade_as_causal(tmp_path, changes):
    with pytest.raises((ValueError, KeyError)):
        validate_influences(build_graph(), estimate_file(tmp_path, **changes))


def test_discrete_control_uses_named_contrast(tmp_path):
    p = estimate_file(tmp_path, parameter='scheduler_mode', kind='finite_contrast',
        **{'from':'round_robin','to':'qos_deadline'})
    graph = validate_influences(build_graph(), p)
    assert any(e.get('sensitivity', {}).get('kind') == 'finite_contrast' for e in graph['edges'] if e.get('sensitivity'))


def users(delivered=0, completed=0, latency=0, failures=0):
    return [dict(ue_id=uid, cell_id='cell_0', queue_bytes=10,
        packet=dict(delivered_bytes=delivered, completed_packets=completed,
            completion_latency_ms_sum=latency, deadline_failed_packets=failures,
            overflow_packets=0, admitted_packets=20, overflow_bytes=0, hol_delay_ms=1))
        for uid in ('ue_0','ue_1','ue_2','ue_4','ue_6','ue_8')]


def test_counter_delta_units_weighted_completion_latency_and_missing_values():
    builder = FrameBuilder(dict(manifest=dict(slot_duration_ms=.5)))
    initial = builder.frame(1, users(), {}, {}, {})
    assert initial['states']['kpi:network_goodput_mbps']['value'] is None
    current = users(delivered=1000, completed=2, latency=10)
    current[2]['packet'].update(completed_packets=1, completion_latency_ms_sum=1)
    current[4]['packet'].update(completed_packets=3, completion_latency_ms_sum=9)
    frame = builder.frame(201, current, {}, {}, {})
    assert frame['states']['kpi:network_goodput_mbps']['value'] == pytest.approx(.48)
    assert frame['states']['kpi:urllc_packet_latency_ms']['value'] == pytest.approx(2.5)  # weighted, not mean of means
    assert frame['states']['kpi:urllc_failure_events_per_s']['value'] == 0
    assert frame['states']['kpi:ue_energy_j']['value'] is None
    assert frame['states']['kpi:mec_expected_gflops']['value'] is None
    assert frame['states']['kpi:urllc_packet_latency_ms']['window_start_slot'] == 1
    json.dumps(frame, allow_nan=False)


def test_recorded_queue_uses_post_delivery_ledger_like_live_telemetry():
    row=dict(ues=[dict(ue_id='ue_0',cell_id='cell_0',queue_bytes=100,
                      packet_pending_bytes=40,packet_delivered_bytes=60)])
    assert trace_users(row)[0]['queue_bytes']==40


def test_rejected_requests_do_not_change_parameters_and_same_tick_reenable_invalidates_suggestions():
    b = FrameBuilder(); ident = 'ue_energy_rf'
    selection = dict(status='applied', slot_id=0, cell_id='cell_0', instance_id=ident, selection_version=1, operation='added')
    b.observe_selection(selection)
    b.observe_proposal(ident, dict(kind='decision', slot_id=40, cell_id='cell_0', ue_id='ue_0', action=2, selection_version=1))
    b.observe_receipt(dict(status='applied', apply_slot=41, after={'cell_0':dict(scheduler_mode='round_robin')}, provenance=[]))
    b.observe_receipt(dict(status='rejected', apply_slot=41, after={'cell_0':dict(scheduler_mode='qos_deadline')}))
    bindings = {'cell_0':{'ue_energy_saving':[ident]}}
    f = b.frame(41, users(), {}, {}, bindings)
    assert f['states']['rcp:scheduler_mode']['value'] == 'round_robin'
    assert len(f['states']['xapp:'+ident]['current_proposals']) == 1
    b.observe_selection({**selection,'slot_id':41,'selection_version':2,'operation':'removed'})
    b.observe_selection({**selection,'slot_id':41,'selection_version':3})
    f = b.frame(42, users(), {}, {}, bindings)
    assert f['states']['xapp:'+ident]['enabled'] is True
    assert not f['states']['xapp:'+ident]['current_proposals']
    assert len(f['states']['xapp:'+ident]['last_proposals']) == 1


def test_compact_snapshot_is_read_only_and_excludes_large_cohort_lists():
    trace = dict(ue_services={'ue_0':dict(ue_energy_total_j=1., ue_task_metrics=None,
        ue_service_policy=dict(offload_fraction=.5), ue_task_cohorts=[dict(large='not exported')])},
        ap_services=dict(cells={'cell_0':dict(ap_policy='light_sleep',ap_energy_total_j=5.)}))
    original = deepcopy(trace)
    state = NS(slot_id=9, ues=[NS(ue_id='ue_0',scheduler_weight=2.)])
    result = compact_telemetry(state, trace)
    result['ue_services']['ue_0']['ue_service_policy']['offload_fraction'] = 1.
    assert trace == original
    assert 'ue_task_cohorts' not in result['ue_services']['ue_0']
    assert result['ue_scheduler_weights']['ue_0'] == 2.


def write_records(path, rows):
    with gzip.open(path,'wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')


def test_recorded_source_uses_trace_timestamps_and_leaves_all_files_unchanged(tmp_path):
    (tmp_path/'config_snapshot.json').write_text(json.dumps(dict(source_sha256='source-fixture',manifest=dict(slot_duration_ms=.5))))
    (tmp_path/'run_status.json').write_text(json.dumps(dict(completed=True,last_completed_slot=201)))
    rows = []
    for slot in range(1,202):
        u = users(delivered=slot*10, completed=slot, latency=slot*2)
        rows.append(dict(slot_id=slot,ues=[dict(ue_id=x['ue_id'],cell_id='cell_0',queue_bytes=10,
            **{'packet_'+k:v for k,v in x['packet'].items()})for x in u],xapp_bindings={}))
    write_records(tmp_path/'rt_sys_trace.jsonl.gz',rows)
    (tmp_path/'xapp_decision_previews.jsonl').write_text(json.dumps(dict(slot=200,apps=[]))+'\n')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    source=replay(tmp_path,max_frames=10)
    assert len(source['frames']) <= 12 and source['frames'][-1]['slot'] == 201
    assert 200 in [f['slot']for f in source['frames']] # exact decision survives stride 21
    assert source['frames'][-1]['states']['kpi:network_goodput_mbps']['value'] == pytest.approx(.96)
    assert source['mode'] == 'recorded' and source['read_only']
    assert before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


def test_live_source_updates_values_and_retains_unavailable_legacy_fields(tmp_path):
    p=tmp_path/'live.json'
    def update(slot, delivered):
        p.write_text(json.dumps(dict(slot=slot,total_slots=400,stage='simulating',ues=users(delivered=delivered),
            xapp_test=dict(run_id='one',active_sets={}),updated_unix_s=0)))
    update(1,0); source=LiveSource(p);a=source.read()
    update(201,1000);b=source.read()
    assert b['frames'][0]['states']['kpi:network_goodput_mbps']['value'] == pytest.approx(.48)
    assert b['frames'][0]['states']['kpi:ue_energy_j']['status'] == 'unavailable'
    assert b['frames'][0]['slot'] == 201 and a['frames'][0]['slot'] == 1


def test_service_is_read_only_and_offline_export_has_no_missing_assets(tmp_path):
    service=GraphService()
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(service))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    try:
        with urlopen(base+'/api/graph')as response:
            bundle=json.load(response)
        assert bundle['source']['mode'] == 'contracts'
        for asset in('/','/style.css','/app.js','/goal_obstruction.js'):
            with urlopen(base+asset)as response:assert response.status == 200
        with pytest.raises(HTTPError)as e:urlopen(base+'/../config_snapshot.json')
        assert e.value.code == 404
        with pytest.raises(HTTPError)as e:urlopen(Request(base+'/api/graph',data=b'{}',method='POST'))
        assert e.value.code == 501
        page=export_html(bundle,tmp_path/'graph.html').read_text()
        assert 'window.SMO_BOOTSTRAP=' in page
        assert 'href="./style.css"' not in page and 'src="./app.js"' not in page
        assert 'src="./goal_obstruction.js"' not in page
        assert page.index('var SmoGoalObstruction')<page.index('var SmoConflictRisk')
        class Assets(HTMLParser):
            def handle_starttag(self, tag, attrs):
                values=dict(attrs)
                assert not(tag=='script'and values.get('src'))
                assert not(tag=='link'and values.get('rel')=='stylesheet')
                assert not(tag=='img'and values.get('src','').startswith(('http:','https:')))
        Assets().feed(page)  # An explicit link to the intent service is not an offline asset dependency.
    finally:
        server.shutdown();server.server_close();thread.join()


def test_energy_proposals_show_structured_drx_without_claiming_locked_or_executed():
    from smo.runtime import proposal
    from ric.near_rt.pools.chongyu.ue_energy_rf import ACTIONS
    for index,action in enumerate(ACTIONS):
        row=proposal(dict(implementation_id='ue_energy_rf',action=index,slot_id=40))
        assert row['proposed_drx']==action['drx']
        assert row['proposed_offload_fraction']==action['offload_fraction']
        assert 'native_locked'not in row['drx_deployment']
        assert 'proposal_only'in row['drx_deployment']
