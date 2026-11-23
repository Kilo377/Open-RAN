"""A same-slot source switch cannot mix displayed proposals with risk inputs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from smo.__main__ import GraphService

ROOT=Path(__file__).resolve().parents[1]


def node(script):
    source=(ROOT/'smo/web/app.js').read_text()
    preview=source[source.index('  let decisionPreview='):source.index('  function renderDecisionPreview()')]
    program="""const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const pending=[],calls=[],c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
c.state={bundle:{source:{mode:'snapshot',name:'共同评估起点',run_id:'A'},graph:{edges:[]}},currentFrame:{slot:520}};
c.frame=()=>c.state.currentFrame;c.location={protocol:'http:'};c.render=()=>{};
c.setTimeout=fn=>{pending.push(fn);return pending.length;};c.clearTimeout=()=>{};
c.postIntent=async(url,request)=>{calls.push(request);return {slot:request.slot,source_id:c.state.bundle.source.run_id,apps:[]};};
vm.runInContext('let conflictQuantificationRequestKey="old-score";',c);
vm.runInContext(PREVIEW,c);
const run=x=>vm.runInContext(x,c);
(async()=>{SCRIPT})().catch(e=>{console.error(e);process.exitCode=1;});
""".replace('PREVIEW',json.dumps(preview)).replace('SCRIPT',script)
    subprocess.run(['node','-e',program],cwd=ROOT,check=True,capture_output=True,text=True)


def test_same_slot_and_name_new_source_hides_old_preview_and_fetches_again():
    node("""run('ensureDecisionPreview()');await pending.shift()();
assert.equal(run('currentPreview().source_id'),'A');
c.state.bundle={...c.state.bundle,source:{...c.state.bundle.source,run_id:'B'}};
assert.equal(run('currentPreview()'),null);
run('ensureDecisionPreview()');assert.equal(pending.length,1);await pending.shift()();
assert.equal(calls.length,2);assert.equal(run('currentPreview().source_id'),'B');
assert.equal(run('conflictQuantificationRequestKey'),null);
""")


def test_old_inflight_preview_cannot_publish_after_source_switch():
    node("""let release;
c.postIntent=(url,request)=>new Promise(resolve=>{release=resolve;});
run('ensureDecisionPreview()');const waiting=pending.shift()();
c.state.bundle={...c.state.bundle,source:{...c.state.bundle.source,run_id:'B'}};
release({slot:520,source_id:'A',apps:[]});await waiting;
assert.equal(run('currentPreview()'),null);
""")


def test_backend_source_switch_is_reported_instead_of_showing_foreign_proposals():
    node("""c.postIntent=async()=>({slot:520,source_id:'B',apps:[{app_id:'foreign'}]});
run('ensureDecisionPreview()');await pending.shift()();
assert.equal(run('currentPreview().apps.length'),0);
assert.ok(run('currentPreview().reason').includes('来源已变化'));
""")


def test_backend_graph_revision_switch_is_not_accepted_as_a_current_preview():
    node("""c.state.bundle={...c.state.bundle,graph_sha256:'graph-A'};
c.postIntent=async()=>({slot:520,source_id:'A',graph_sha256:'graph-B',apps:[{app_id:'foreign'}]});
run('ensureDecisionPreview()');await pending.shift()();
assert.equal(run('currentPreview().apps.length'),0);
assert.ok(run('currentPreview().reason').includes('图谱版本已变化'));
""")


def test_risk_cache_context_changes_for_gain_model_and_same_slot_reading():
    node("""const f={slot:520,states:{'kpi:k':{value:10}}},g={edges:[{estimate:1}]};
const b={source:c.state.bundle.source,graph:g};
const key=c.SmoGraph.quantificationContextKey(b,f);
assert.equal(key,c.SmoGraph.quantificationContextKey(b,f));
assert.notEqual(key,c.SmoGraph.quantificationContextKey({...b,graph:{edges:[{estimate:2}]}},f));
assert.notEqual(key,c.SmoGraph.quantificationContextKey(b,{slot:520,states:{'kpi:k':{value:20}}}));
assert.notEqual(key,c.SmoGraph.quantificationContextKey({...b,source:{...b.source,run_id:'B'}},f));
assert.notEqual(c.SmoGraph.decisionPreviewContextKey({...b,graph_sha256:'a'},f),
 c.SmoGraph.decisionPreviewContextKey({...b,graph_sha256:'b'},f));
""")


def test_prepared_preview_response_tags_source_without_editing_saved_record():
    service=GraphService();raw=dict(slot=520,apps=[dict(app_id='a')]);before=deepcopy(raw)
    frame=dict(slot=520,time_ms=260,states={})
    service.evaluation_source=dict(mode='snapshot',name='共同评估起点',run_id='A',frames=[frame])
    service.evaluation_origin=dict(origin_id='origin',baseline_slot=520,observe_slots=500);service.evaluation_preview=raw
    first=service.preview_xapps(520)
    assert first['source_id']=='A' and raw==before
    assert first['graph_sha256']==service.bundle()['graph_sha256']
    service.evaluation_source={**service.evaluation_source,'run_id':'B'}
    assert service.preview_xapps(520)['source_id']=='B'
    assert first['source_id']=='A' and raw==before


def test_python_and_browser_choose_the_same_order_independent_frozen_reference():
    from smo.influence_units import frozen_intent_reference
    row=dict(kpi='k',current=80,anchors=[
        dict(intent_id='b',snapshot=dict(time_ms=20,slot=40),current=30),
        dict(intent_id='a',snapshot=dict(time_ms=10,slot=20),current=20)])
    frame=dict(states={'kpi:k':dict(value=40)})
    for anchors in (row['anchors'],list(reversed(row['anchors']))):
        selected={**row,'anchors':anchors}
        expected=frozen_intent_reference(selected,frame)
        node('assert.equal(c.SmoGraph.frozenIntentReference('+json.dumps(selected)+','+json.dumps(frame)+'),'+json.dumps(expected)+');')


def test_quantification_rejects_new_backend_graph_before_invoking_risk_engine():
    source=(ROOT/'smo/web/app.js').read_text()
    quant=source[source.index('  let conflictQuantification='):source.index('  const quantificationReasons=')]
    program="""const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const pending=[],c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const frame={slot:520,states:{}},catalog={groups:[{id:'g',app_ids:['a']}]};
c.state={bundle:{graph_sha256:'graph-A',graph:{edges:[]},intent_service:{conflict_input_schema:'v1'},
 source:{run_id:'run',frames:[frame]}},intent:{snapshot:{slot:520},ranges:[{kpi:'k',lower:1}],combination_catalog:catalog},
 intentBusy:false,activeIntents:[{id:'saved'}],combinationCatalog:catalog,showDashed:false};
c.frame=()=>frame;c.location={protocol:'http:'};c.render=()=>{};
c.$=id=>({value:id==='conflict-method'?'goal_obstruction_v2':'500'});
c.setTimeout=fn=>{pending.push(fn);return pending.length;};c.clearTimeout=()=>{};
c.postIntent=async()=>({source_id:'run',graph_sha256:'graph-B',groups:[{group_id:'g'}]});
c.SmoGoalObstruction={review(){throw Error('Risk must not run with a foreign graph');}};
vm.runInContext(QUANT,c);vm.runInContext('ensureConflictQuantification()',c);
(async()=>{await pending.shift()();const result=vm.runInContext('conflictQuantification',c);
 assert.equal(result.status,'unavailable');assert.ok(result.error.includes('图谱增益版本已变化'));
})().catch(e=>{console.error(e);process.exitCode=1;});
""".replace('QUANT',json.dumps(quant))
    subprocess.run(['node','-e',program],cwd=ROOT,check=True,capture_output=True,text=True)


def test_packet_rejects_same_slot_kpi_change_instead_of_silently_rebasing():
    source=(ROOT/'smo/web/app.js').read_text()
    fn=source[source.index('  function decisionPacket('):source.index('  async function requestConflictQuantification')]
    script="""const assert=require('node:assert/strict');FN
const old={slot:100,states:{'kpi:k':{value:10}}};
const packet={snapshot:{slot:100},preview_slot:100,decision_frame:{states:{'kpi:k':{value:10}},slot:100},decision_preview:{slot:100}};
assert.equal(decisionPacket(packet,old).frame.states['kpi:k'].value,10);
packet.decision_frame.states['kpi:k'].value=20;
assert.throws(()=>decisionPacket(packet,old),/网络快照已更新/);
""".replace('FN',fn)
    subprocess.run(['node','-e',script],cwd=ROOT,check=True,capture_output=True,text=True)


def test_quantification_uses_one_trusted_packet_without_a_second_preview_request():
    source=(ROOT/'smo/web/app.js').read_text()
    quant=source[source.index('  let conflictQuantification='):source.index('  const quantificationReasons=')]
    script="""const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const pending=[],c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const frame={slot:520,states:{}},catalog={groups:[{id:'g',app_ids:['a']}]};
c.state={bundle:{graph_sha256:'graph-A',graph:{edges:[]},intent_service:{conflict_input_schema:'v1'},source:{run_id:'run',frames:[frame]}},
 intent:{snapshot:{slot:520},ranges:[{kpi:'k',lower:1}],combination_catalog:catalog},intentBusy:false,
 activeIntents:[{id:'saved'}],combinationCatalog:catalog,showDashed:false};
c.frame=()=>frame;c.location={protocol:'http:'};c.render=()=>{};
c.$=id=>({value:id==='conflict-method'?'goal_obstruction_v2':'500'});
c.setTimeout=fn=>{pending.push(fn);return pending.length;};c.clearTimeout=()=>{};
const packet={source_id:'run',graph_sha256:'graph-A',snapshot:{slot:520},preview_slot:520,
 decision_frame:structuredClone(frame),decision_preview:{source_id:'run',graph_sha256:'graph-A',slot:520,apps:[]},
 decision_frame_sha256:'frame-pin',decision_preview_sha256:'preview-pin',snapshot_packet_sha256:'packet-pin',
 groups:[{group_id:'g',app_ids:['a'],kpis:[{range:{kpi:'k',lower:1}}]}]};let posts=0;
c.postIntent=async(url)=>{posts++;assert.equal(url,'./api/conflicts/inputs');return packet;};
c.SmoGoalObstruction={review(graph,f,groups,ranges,p){assert.equal(f,packet.decision_frame);assert.equal(p,packet.decision_preview);
 return {method:'goal_obstruction_v2',groups:[{group_id:'g',score:2}]};}};
vm.runInContext(QUANT,c);vm.runInContext('ensureConflictQuantification()',c);
(async()=>{await pending.shift()();const result=vm.runInContext('conflictQuantification',c);
 assert.equal(posts,1);assert.equal(result.groups[0].score,2);assert.equal(result.snapshot_packet_sha256,'packet-pin');
})().catch(e=>{console.error(e);process.exitCode=1;});
""".replace('QUANT',json.dumps(quant))
    subprocess.run(['node','-e',script],cwd=ROOT,check=True,capture_output=True,text=True)
