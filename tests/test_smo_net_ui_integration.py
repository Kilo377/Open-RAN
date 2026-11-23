"""Staged UI wiring only; production assets and frozen equations stay untouched."""
import json
from pathlib import Path
import re
import subprocess
from threading import Thread
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from smo.__main__ import export_html,handler
from smo.influence_units import align_graph,normalization
from test_smo_goal_obstruction import fixture

ROOT=Path(__file__).resolve().parents[1]


def node(script):
    subprocess.run(['node','-e',script],cwd=ROOT,check=True,capture_output=True,text=True)


def test_net_is_optional_default_and_temporal_stay_unchanged():
    html=(ROOT/'smo/web/index.html').read_text();app=(ROOT/'smo/web/app.js').read_text()
    options=re.search(r'<select id="conflict-method">(.*?)</select>',html).group(1)
    assert re.findall(r'<option value="([^"]+)"',options)==['goal_obstruction_v2','net_intent_obstruction_v1','goal_obstruction_v1','intent_control_dynamics_v4']
    assert "?.value||'goal_obstruction_v2'"in app
    assert html.index('goal_obstruction.js')<html.index('intent_obstruction.js')<html.index('app.js')
    assert 'temporal_net_intent_obstruction_v1'not in options


@pytest.mark.parametrize('mutate_frame',[False,True])
def test_net_dispatch_uses_only_trusted_packet_and_rejects_identity_shift(mutate_frame):
    source=(ROOT/'smo/web/app.js').read_text();quant=source[source.index('  let conflictQuantification='):source.index('  const quantificationReasons=')]
    script=r'''const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const pending=[],c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const f={slot:520,states:{}},catalog={groups:[{id:'g',app_ids:['a']}]};
c.state={bundle:{graph_sha256:'G',graph:{edges:[]},intent_service:{conflict_input_schema:'v1'},source:{run_id:'R',frames:[f]}},intent:{snapshot:{slot:520},ranges:[{kpi:'k',lower:1}],combination_catalog:catalog},intentBusy:false,activeIntents:[{id:'saved'}],combinationCatalog:catalog,showDashed:false};
c.frame=()=>f;c.location={protocol:'http:'};c.render=()=>{};c.$=id=>({value:id==='conflict-method'?'net_intent_obstruction_v1':'500'});
c.setTimeout=fn=>{pending.push(fn);return pending.length;};c.clearTimeout=()=>{};
const packet={source_id:'R',graph_sha256:'G',snapshot:{slot:520},preview_slot:520,decision_frame:structuredClone(f),decision_preview:{source_id:'R',graph_sha256:'G',slot:520,apps:[]},snapshot_packet_sha256:'packet-pin',groups:[{group_id:'g',app_ids:['a'],kpis:[{range:{kpi:'k',lower:1}}]}],actual_after_kpis:{forbidden:999}};
if(MUTATE)packet.decision_frame.states.changed={value:99};
let calls=0,posts=0;c.postIntent=async(url,request)=>{posts++;assert.equal(url,'./api/conflicts/inputs');assert.deepEqual(Object.keys(request).sort(),['active_intent_ids','group_scope','observe_slots','show_dashed','slot']);assert.equal(request.group_scope,'all_covered');return packet;};
c.SmoIntentObstruction={review(graph,frame,groups,ranges,preview,options){calls++;assert.equal(frame,packet.decision_frame);assert.equal(preview,packet.decision_preview);assert.equal(options.method,'net_intent_obstruction_v1');assert.ok(!('actual_after_kpis'in options));assert.equal(groups.length,1);return{method:options.method,groups:[{group_id:'g',score:3}]};}};
vm.runInContext('SmoConflictRisk.review=()=>{throw Error("Wrong engine")}',c);c.SmoGoalObstruction={review(){throw Error('Wrong engine');}};
vm.runInContext(QUANT,c);vm.runInContext('ensureConflictQuantification()',c);
(async()=>{await pending.shift()();const result=vm.runInContext('conflictQuantification',c);assert.equal(posts,1);if(MUTATE){assert.equal(calls,0);assert.equal(result.status,'unavailable');assert.ok(result.error.includes('网络快照已更新'));}else{assert.equal(calls,1);assert.equal(result.method,'net_intent_obstruction_v1');assert.equal(result.snapshot_packet_sha256,'packet-pin');assert.equal(result.deployment_results_used,false);}})().catch(e=>{console.error(e);process.exitCode=1;});
'''.replace('MUTATE',str(mutate_frame).lower()).replace('QUANT',json.dumps(quant))
    node(script)


def test_net_renderer_accepts_native_schema_without_app_vectors():
    source=(ROOT/'smo/web/app.js').read_text();functions=source[source.index('  function renderNetObstruction('):source.index('  function renderCombinationCatalog() {')]
    script=r'''const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const q={group_id:'g',score:9.1,raw_obstruction_strength:.1,reference_effect:-.1,model_coverage_complete:true,arbitration_outcome_count:2,structural_counts:{direct:1,indirect:2},goal_components:[{kpi:'k',reference_progress:-.1,kpi_scale:10}]};
Object.assign(c,{chineseKpis:{k:'Target'},find:()=>({unit:'W'}),fmt:(n,d)=>String(Number(n.toFixed(d))),esc:String,quantifiedGroup:()=>q,currentQuantification:()=>({method:'net_intent_obstruction_v1'}),conflictQuantificationBusy:false});
vm.runInContext(FNS+';html=renderGroupQuantification({id:"g",conflict_summaries:[]})',c);assert.ok(c.html.includes('9.1'));assert.ok(c.html.includes('归一化净作用'));assert.ok(c.html.includes('实际 Metric 独立'));assert.ok(!c.html.includes('完整策略子集'));
q.score=null;q.reason='reference_projection_incomplete';vm.runInContext('html=renderGroupQuantification({id:"g"})',c);assert.ok(c.html.includes('缺少适用增益'));
'''.replace('FNS',json.dumps(functions))
    node(script)


def test_net_script_is_served_from_staged_static_allowlist():
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(SimpleNamespace()))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/intent_obstruction.js')as response:
            assert response.headers['Content-Type'].startswith('text/javascript')
            assert response.read()==(ROOT/'smo/web/intent_obstruction.js').read_bytes()
    finally:server.shutdown();server.server_close();thread.join()


def test_offline_export_embeds_net_and_evaluates_without_fetch_or_mutating_packet(tmp_path):
    g,f,c,p,_=fixture();payload=dict(graph=align_graph(g),frame=f,groups=c['groups'],ranges=c['ranges'],preview=p,options={'method':'net_intent_obstruction_v1','normalization':normalization(c['ranges'],f)})
    artifact=export_html({'graph':{},'source':{'frames':[]}},tmp_path/'offline.html').read_text()
    assert './intent_obstruction.js'not in artifact and 'var SmoIntentObstruction'in artifact
    scripts=[s for s in re.findall(r'<script>([\s\S]*?)</script>',artifact)if not s.startswith('window.SMO_BOOTSTRAP=')]
    program=r'''const assert=require('node:assert/strict'),vm=require('vm'),c={structuredClone};vm.createContext(c);for(const source of SCRIPTS)vm.runInContext(source,c);const d=DATA,before=JSON.stringify(d);const result=c.SmoIntentObstruction.review(d.graph,d.frame,d.groups,d.ranges,d.preview,d.options);assert.equal(result.method,'net_intent_obstruction_v1');assert.equal(result.status,'ready');assert.equal(result.observed_after_kpis_used,false);assert.equal(JSON.stringify(d),before);assert.equal(typeof c.fetch,'undefined');'''.replace('SCRIPTS',json.dumps(scripts)).replace('DATA',json.dumps(payload))
    node(program)
