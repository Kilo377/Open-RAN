/* Read-only conflict analysis. Node export is used by analytic tests. */
var SmoConflicts = (() => {
  'use strict';
  // Editable natural-language examples. No preset supplies an analysis verdict,
  // bypasses the LLM interval boundary or changes simulator/controller state.
  const intentCases = [
    {id:'c04',source:'C-04 / I-15',
      text:'提高全网下行吞吐量，同时降低所有基站的总功耗。',
      kpis:['network_goodput_mbps','ap_power_w'],
      note:''},
    {id:'c06',source:'C-06, scheduling subset',
      text:'提高全网下行吞吐量，同时降低 URLLC 数据包的超时与丢弃事件率。',
      kpis:['network_goodput_mbps','urllc_failure_events_per_s'],
      note:'请使用存在 URLLC 压力的场景。'},
    {id:'c02',source:'C-02, M1 mechanism adaptation',
      text:'降低所有 UE 的总能耗，同时降低 MEC 业务的总预期计算需求。',
      kpis:['ue_energy_j','mec_expected_gflops'],
      note:'X-13 目前仅控制部分 MEC 业务。'},
    {id:'i16',source:'I-16, request form from C-08',
      text:'降低所有 UE 的总能耗，同时降低 MEC 卸载任务的平均端到端时延。',
      kpis:['ue_energy_j','mec_offloaded_latency_ms'],
      note:'最新的 X-12-MEC 模型仍需独立仿真验证。'},
    {id:'lc02',source:'LC-02, M5 interval adaptation',
      text:'要求同一观测窗口内的全网下行吞吐量，同时严格高于和严格低于其当前值。',
      kpis:['network_goodput_mbps'],
      note:'同一 KPI 在同一观测窗口内的要求互相矛盾。'},
    {id:'c03',source:'C-03 / I-19, negative-control adaptation',needsThreshold:true,
      text:'在保持全网下行吞吐量的同时，将基站总功耗降至经过验证的可达最低值以下。',
      kpis:['ap_power_w','network_goodput_mbps'],
      note:'分析前，请填写经过本网络验证的基站功耗限值。'},
  ];
  /* ACTION_CATALOG_START */
  const catalog = {"schema":"oran.action-catalog.v1","contracts":{"artifacts/xapps/chongyu/20261117/release.json":"d12e25f7010ca03e1094269fbfa31f07742d746cc05e788ef9fbc56ff7ec74bc","ric/near_rt/pools/chongyu/ap_energy_rf.py":"9d9ef6e0d23b83384335c5c9b8b7d5150c82bcc2572d020500146474fcc70568","ric/near_rt/pools/chongyu/catalog.json":"a4da5bdb88811e6ee271bda0dcc2f2dc469a26f101fbb76b92f4303b85fdb565","ric/near_rt/pools/chongyu/latency_scheduler_rf.py":"b6e041b83cdc86e9c33aa0a44140c1560f5c23a936a67e442862fb1e57a7e920","ric/near_rt/pools/chongyu/latency_urllc_rf.py":"ed0deb9a97252467d5cb94bf72b700488e46fac2b46a413432dafba081e05475","ric/near_rt/pools/chongyu/mec_budget.json":"73f72e8964639cbb8948c53410287d9cbdc5d765de40f76e49ca9a28e0c3cdcb","ric/near_rt/pools/chongyu/mec_offload_budget_qgbdt.py":"b7eb96d813edb6b76edc7482552f33f6f0adc1976c672ef51680cbd2be3d4908","ric/near_rt/pools/chongyu/throughput_scheduler_mlp.py":"4adb3e5e086ec400d74a3f619f2839cbedee6f64b26bdf5931dc70018478522d","ric/near_rt/pools/chongyu/ue_energy_rf.py":"00c6e498544b794d17a0b046ec0382153c5e571df4af0b0b01c47411b1a46ec7","ric/near_rt/pools/chongyu/x01.json":"e1156af80c8269d16006fb4b0ce3472541d11cd31a399d78469724dd8a773ce2","ric/near_rt/pools/chongyu/x02.json":"dde70b89d62a5c7d36543992aa8329a8c4d88922957bca0e858d4888ab2479bc","ric/near_rt/pools/chongyu/x08.json":"5e4b879d257c3e747c361f1a236e824230ab66f684d22b4f19ae1b0a4b4d6206","ric/near_rt/pools/chongyu/x12.json":"f3001ea6648640b7df34dc35ee88184a7383770840c79d100d7e2752d10bb512","ric/near_rt/pools/chongyu/x12_urllc_qos.json":"6a28daf9a52d93b424f778bc23542e2c2f03d6317e347d373161a965f064dc6a"},"profiles":{"xapp:ue_energy_rf":[{"name":"DRX 40/4/8 ms \u00b7 Offload 0%","parts":[{"parameter":"ue_offload_fraction","value":0.0,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":40.0,"on_duration_ms":4.0,"inactivity_ms":8.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 40/4/8 ms \u00b7 Offload 50%","parts":[{"parameter":"ue_offload_fraction","value":0.5,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":40.0,"on_duration_ms":4.0,"inactivity_ms":8.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 40/4/8 ms \u00b7 Offload 100%","parts":[{"parameter":"ue_offload_fraction","value":1.0,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":40.0,"on_duration_ms":4.0,"inactivity_ms":8.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 20/4/1 ms \u00b7 Offload 0%","parts":[{"parameter":"ue_offload_fraction","value":0.0,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":20.0,"on_duration_ms":4.0,"inactivity_ms":1.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 20/4/1 ms \u00b7 Offload 50%","parts":[{"parameter":"ue_offload_fraction","value":0.5,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":20.0,"on_duration_ms":4.0,"inactivity_ms":1.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 20/4/1 ms \u00b7 Offload 100%","parts":[{"parameter":"ue_offload_fraction","value":1.0,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":20.0,"on_duration_ms":4.0,"inactivity_ms":1.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 80/4/1 ms \u00b7 Offload 0%","parts":[{"parameter":"ue_offload_fraction","value":0.0,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":80.0,"on_duration_ms":4.0,"inactivity_ms":1.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 80/4/1 ms \u00b7 Offload 50%","parts":[{"parameter":"ue_offload_fraction","value":0.5,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":80.0,"on_duration_ms":4.0,"inactivity_ms":1.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]},{"name":"DRX 80/4/1 ms \u00b7 Offload 100%","parts":[{"parameter":"ue_offload_fraction","value":1.0,"group":"mec_workload","members":["ue_0","ue_4"],"scope":"MEC workload"},{"parameter":"ue_drx_configs","value":{"enabled":true,"cycle_ms":80.0,"on_duration_ms":4.0,"inactivity_ms":1.0,"wakeup_ms":0.5,"offset_ms":0.0},"group":"mec_drx_profile","members":["ue_0","ue_4"],"scope":"MEC workload"}]}],"xapp:ap_energy_rf":[{"name":"All cells active","parts":[{"parameter":"cell_sleep_mode","value":"active","group":"all_cells","members":[],"scope":"All cells"}]},{"name":"Deep sleep for empty cells","parts":[{"parameter":"cell_sleep_mode","value":"deep_sleep","group":"empty_cells","members":[],"scope":"Empty cells only"}]},{"name":"Micro-sleep occupied cells","parts":[{"parameter":"cell_sleep_mode","value":"micro_sleep","group":"occupied_cells","members":[],"scope":"Occupied cells; empty cells use deep sleep"}]},{"name":"Light-sleep occupied cells","parts":[{"parameter":"cell_sleep_mode","value":"light_sleep","group":"occupied_cells","members":[],"scope":"Occupied cells; empty cells use deep sleep"}]}],"xapp:throughput_scheduler_mlp":[{"name":"Native round robin","parts":[{"parameter":"scheduler_mode","value":"round_robin","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"all_users_reset","members":["ue_0","ue_1","ue_2","ue_4","ue_6","ue_8"],"scope":"All traffic priorities reset to 1"}]},{"name":"Neutral latency-aware policy","parts":[{"parameter":"scheduler_mode","value":"latency_aware","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"all_users_reset","members":["ue_0","ue_1","ue_2","ue_4","ue_6","ue_8"],"scope":"All traffic priorities reset to 1"}]},{"name":"Efficiency-first priority 1.5","parts":[{"parameter":"scheduler_mode","value":"latency_aware","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"all_users_reset","members":["ue_0","ue_1","ue_2","ue_4","ue_6","ue_8"],"scope":"All traffic priorities reset to 1"},{"parameter":"ue_scheduler_weights","value":1.5,"group":"efficiency_priority","members":[],"scope":"Public efficiency-selected users"}]},{"name":"Efficiency-first priority 2.0","parts":[{"parameter":"scheduler_mode","value":"latency_aware","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"all_users_reset","members":["ue_0","ue_1","ue_2","ue_4","ue_6","ue_8"],"scope":"All traffic priorities reset to 1"},{"parameter":"ue_scheduler_weights","value":2.0,"group":"efficiency_priority","members":[],"scope":"Public efficiency-selected users"}]}],"xapp:latency_scheduler_rf":[{"name":"Native round robin","parts":[{"parameter":"scheduler_mode","value":"round_robin","group":"eligible_cells","members":[],"scope":"MEC-serving cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"mec_priority","members":["ue_0","ue_4"],"scope":"MEC workload","context_approximation":true}]},{"name":"MEC priority 0.5","parts":[{"parameter":"scheduler_mode","value":"latency_aware","group":"eligible_cells","members":[],"scope":"MEC-serving cells"},{"parameter":"ue_scheduler_weights","value":0.5,"group":"mec_priority","members":["ue_0","ue_4"],"scope":"MEC workload","context_approximation":true}]},{"name":"MEC priority 1.0","parts":[{"parameter":"scheduler_mode","value":"latency_aware","group":"eligible_cells","members":[],"scope":"MEC-serving cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"mec_priority","members":["ue_0","ue_4"],"scope":"MEC workload","context_approximation":true}]},{"name":"MEC priority 2.0","parts":[{"parameter":"scheduler_mode","value":"latency_aware","group":"eligible_cells","members":[],"scope":"MEC-serving cells"},{"parameter":"ue_scheduler_weights","value":2.0,"group":"mec_priority","members":["ue_0","ue_4"],"scope":"MEC workload","context_approximation":true}]}],"xapp:latency_urllc_rf":[{"name":"Native round robin","parts":[{"parameter":"scheduler_mode","value":"round_robin","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"urllc_priority","members":["ue_2","ue_6"],"scope":"URLLC traffic","context_approximation":false},{"parameter":"ue_scheduler_weights","value":1.0,"group":"other_traffic_reset","members":["ue_0","ue_1","ue_4","ue_8"],"scope":"Other traffic priorities reset to 1"}]},{"name":"URLLC priority 1.0","parts":[{"parameter":"scheduler_mode","value":"qos_deadline","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":1.0,"group":"urllc_priority","members":["ue_2","ue_6"],"scope":"URLLC traffic","context_approximation":false},{"parameter":"ue_scheduler_weights","value":1.0,"group":"other_traffic_reset","members":["ue_0","ue_1","ue_4","ue_8"],"scope":"Other traffic priorities reset to 1"}]},{"name":"URLLC priority 2.0","parts":[{"parameter":"scheduler_mode","value":"qos_deadline","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":2.0,"group":"urllc_priority","members":["ue_2","ue_6"],"scope":"URLLC traffic","context_approximation":false},{"parameter":"ue_scheduler_weights","value":1.0,"group":"other_traffic_reset","members":["ue_0","ue_1","ue_4","ue_8"],"scope":"Other traffic priorities reset to 1"}]},{"name":"URLLC priority 4.0","parts":[{"parameter":"scheduler_mode","value":"qos_deadline","group":"all_cells","members":[],"scope":"All cells"},{"parameter":"ue_scheduler_weights","value":4.0,"group":"urllc_priority","members":["ue_2","ue_6"],"scope":"URLLC traffic","context_approximation":false},{"parameter":"ue_scheduler_weights","value":1.0,"group":"other_traffic_reset","members":["ue_0","ue_1","ue_4","ue_8"],"scope":"Other traffic priorities reset to 1"}]}],"xapp:mec_offload_budget_qgbdt":[{"name":"Offloading probability 0%","parts":[{"parameter":"ue_offload_fraction","value":0.0,"group":"mec_workload","members":["ue_4"],"scope":"Budget-controlled MEC workload"}]},{"name":"Offloading probability 50%","parts":[{"parameter":"ue_offload_fraction","value":0.5,"group":"mec_workload","members":["ue_4"],"scope":"Budget-controlled MEC workload"}]}]}};
  /* ACTION_CATALOG_END */
  const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
  const unique = xs => [...new Set(xs)];
  const stable = x => JSON.stringify(Object.entries(x || {}).sort(([a],[b])=>a.localeCompare(b)));
  const allEffects = v => uniqueEffects([...v.matches.flatMap(m=>[...m.paths,...m.opposing_paths]),...v.side_effects]);
  const uniqueEffects = xs => [...new Map(xs.map(e=>[[e.parameter,e.kpi,e.contrast_id].join('|'),e])).values()];
  const activeEffect = e => e.direction !== 'no_change' && e.evidence !== 'no_observed_change';
  function defaults(plan) {
    // Listing combinations does not choose one, even when a primary app exists.
    return {};
  }
  function rangePressure(effect,range) {
    if(!range)return false;
    const finite=x=>typeof x==='number'&&Number.isFinite(x);
    if(finite(range.current)&&finite(effect.delta)&&!effect.scope_approximation&&!effect.context_approximation){
      const band=effect.delta_interval?.length===2&&effect.delta_interval.every(finite)?effect.delta_interval:[effect.delta,effect.delta];
      const low=range.current+Math.min(...band),high=range.current+Math.max(...band);
      return range.lower!=null&&(low<range.lower||low===range.lower&&!range.lower_inclusive)||range.upper!=null&&(high>range.upper||high===range.upper&&!range.upper_inclusive);
    }
    // No calibrated magnitude: pressure toward a constrained boundary remains
    // uncertain. Opposite intent words alone never imply an empty interval.
    if(range.lower==null&&range.upper==null)return ['increase','decrease'].includes(range.needed_direction)&&effect.direction!==range.needed_direction;
    return effect.direction==='increase'?range.upper!=null&&(range.domain?.[1]==null||range.upper<range.domain[1]||!range.upper_inclusive):effect.direction==='decrease'&&range.lower!=null&&(range.lower>(range.domain?.[0]??0)||!range.lower_inclusive);
  }
  function profileFor(c,v) {
    const index=Number(v.id.split(':preview:').pop()), profile=catalog.profiles[c.app_id]?.[index];
    return profile?.name===v.name ? profile : null;
  }
  function writes(graph,frame,c,v) {
    const profile=profileFor(c,v); if(!profile)return null;
    const controls=new Map(graph.edges.filter(e=>e.kind==='controls'&&e.source===c.app_id&&e.deployment!=='native_locked').map(e=>[e.target.slice(4),e.id]));
    const output=new Map();
    for(const part of profile.parts) {
      if(!controls.has(part.parameter))continue;
      const values=frame.states['rcp:'+part.parameter]?.by_entity || {};
      const members=part.members.length?part.members:Object.keys(values);
      const conditional=['efficiency_priority','eligible_cells','empty_cells','occupied_cells'].includes(part.group);
      for(const entity of members.length?members:['?']) {
        const key=part.parameter+'|'+entity,prev=output.get(key);
        // A runtime-selected subset may overwrite any baseline reset, not all UEs.
        const possible=conditional&&prev?unique([...prev.values,part.value]):[part.value];
        output.set(key,{parameter:part.parameter,entity,values:possible,
          uncertain:conditional||entity==='?',scope:part.scope,control_edge:controls.get(part.parameter)});
      }
    }
    return [...output.values()];
  }
  function review(graph,frame,plan,selection) {
    const result={schema:'oran.conflict-review.v1',snapshot:{...plan.snapshot},selected:[],conflicts:[],range_risks:[],coverage:[],deployment_enabled:false};
    if(frame?.slot!==plan.snapshot.slot){result.unavailable='The action scopes are not from the analyzed snapshot.';return result;}
    if(stable(graph.contract_sha256)!==stable(catalog.contracts)){result.unavailable='Action catalog differs from the loaded contracts.';return result;}
    const chosen=[];
    for(const [app,id] of Object.entries(selection)) {
      const c=plan.candidates.find(c=>c.app_id===app),v=c?.variants.find(v=>v.id===id);
      if(!v){result.unavailable='A selected preview is no longer available.';return result;}
      const w=writes(graph,frame,c,v);
      if(!w){result.unavailable='Action profile differs from the selected preview.';return result;}
      chosen.push({c,v,w,effects:allEffects(v)}); result.selected.push({app_id:app,variant_id:id});
    }
    const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
    const label=p=>nodes.get('rcp:'+p)?.label || p;
    const add=row=>{row.id='conflict:'+result.conflicts.length;row.edge_ids=unique(row.edge_ids);result.conflicts.push(row);};
    for(let i=0;i<chosen.length;i++)for(let j=i+1;j<chosen.length;j++) {
      const a=chosen[i],b=chosen[j],byParam=new Map();
      for(const x of a.w)for(const y of b.w) {
        if(x.parameter!==y.parameter || (x.entity!==y.entity&&x.entity!=='?'&&y.entity!=='?'))continue;
        if(x.values.length===1&&y.values.length===1&&same(x.values[0],y.values[0]))continue;
        const rows=byParam.get(x.parameter)||[];
        rows.push({entity:x.entity==='?'?y.entity:x.entity,left:x.values,right:y.values,uncertain:x.uncertain||y.uncertain,left_scope:x.scope,right_scope:y.scope});byParam.set(x.parameter,rows);
      }
      for(const [parameter,overlaps]of byParam) add({type:'direct',status:overlaps.some(o=>!o.uncertain)?'incompatible_writes':'scope_uncertain',
        apps:[a.c.app_id,b.c.app_id],parameters:[parameter],kpis:[],label:label(parameter),overlaps,
        edge_ids:[...a.w,...b.w].filter(w=>w.parameter===parameter).map(w=>w.control_edge)});
      // A cross-RCP adverse path to another candidate's requested primary KPI.
      for(const [source,target]of [[a,b],[b,a]])for(const goal of target.v.matches.filter(m=>m.primary_target??m.policy_goal_aligned)) {
        const victimParameters=unique(target.w.map(w=>w.parameter));
        const range=plan.ranges.find(r=>r.kpi===goal.kpi);
        const pressure=e=>range?rangePressure(e,range):e.direction!==goal.direction;
        const harmful=source.effects.filter(e=>e.kpi===goal.kpi&&activeEffect(e)&&pressure(e)&&victimParameters.some(p=>p!==e.parameter));
        if(!harmful.length)continue;
        const helpful=source.effects.some(e=>e.kpi===goal.kpi&&activeEffect(e)&&!pressure(e));
        const targetPaths=[...goal.paths,...goal.opposing_paths];
        const victimSupported=goal.paths.some(e=>activeEffect(e)&&!pressure(e)&&e.evidence==='resolved_average'&&!e.scope_approximation&&harmful.some(h=>h.parameter!==e.parameter));
        const uncertain=helpful || !victimSupported || harmful.some(e=>e.evidence!=='resolved_average'||e.scope_approximation);
        const targetControls=target.w.filter(w=>harmful.some(e=>w.parameter!==e.parameter));
        const targetEdges=targetPaths.filter(e=>targetControls.some(w=>w.parameter===e.parameter)).map(e=>e.edge_id);
        const contractEdges=targetControls.map(w=>'influence:'+w.parameter+':'+goal.kpi).filter(id=>graph.edges.some(e=>e.id===id));
        add({type:'indirect',status:uncertain?'uncertain_path':'measured_path',apps:[source.c.app_id,target.c.app_id],
          parameters:unique(harmful.map(e=>e.parameter)),kpis:[goal.kpi],label:nodes.get('kpi:'+goal.kpi)?.label||goal.kpi,
          needed_direction:goal.direction,effects:harmful,coupled_response:helpful,
          edge_ids:[...source.w.filter(w=>harmful.some(e=>e.parameter===w.parameter)).map(w=>w.control_edge),...harmful.map(e=>e.edge_id),...targetControls.map(w=>w.control_edge),...targetEdges,...contractEdges]});
      }
    }
    for(const source of chosen)for(const r of plan.ranges) {
      const harmful=source.effects.filter(e=>e.kpi===r.kpi&&activeEffect(e)&&rangePressure(e,r));
      if(harmful.length)result.range_risks.push({app_id:source.c.app_id,kpi:r.kpi,effects:harmful,
        at_boundary:harmful.some(e=>e.direction==='increase'?r.upper!=null&&r.current>=r.upper:r.lower!=null&&r.current<=r.lower),
        note:'Directional pressure only; no net bundle or interval breach is predicted.'});
    }
    result.coverage=plan.ranges.map(r=>({kpi:r.kpi,status:r.status,apps:chosen.filter(x=>x.v.matches.some(m=>m.kpi===r.kpi)).map(x=>x.c.app_id)}));
    result.note='Candidate action previews, not live decisions. Indirect conflicts are potential path interactions; joint-run validation is required.';
    return result;
  }
  function resolvedEstimate(s) {
    const intervals=s?.confidence_interval?[s.confidence_interval]:Object.values(s?.entity_uncertainty||{}).map(v=>v.confidence_interval);
    return intervals.some(ci=>ci&&(ci[0]>0||ci[1]<0));
  }
  function parameterKpiDashed(edge,estimate,{uncalibrated=false,uncertain=false}={}) {
    // The evidence filter applies only to RCP -> KPI influences.
    // xApp -> RCP contract/lock semantics never depend on gain confidence.
    return edge.kind==='influences'&&(uncertain||uncalibrated||!estimate||!resolvedEstimate(estimate));
  }
  function influencePresentation(edge,{contrastIds=null,uncalibrated=false}={}) {
    if(edge.kind!=='influences')return {estimates:[],summary:false,uncalibrated:false,measured:false,sensitivity:null,dashed:false};
    const all=edge.sensitivities?.length?edge.sensitivities:edge.sensitivity?[edge.sensitivity]:[];
    const estimates=contrastIds===null?all:all.filter(s=>contrastIds.includes(s.contrast_id));
    const missing=contrastIds!==null&&(!contrastIds.length||contrastIds.some(id=>!all.some(s=>s.contrast_id===id)));
    const calibrated=!uncalibrated&&!missing;
    // A relationship can be resolved for one categorical transition and not another.
    // The overview shows that conditional evidence without averaging action effects.
    // Action previews restrict this set to their own transitions before filtering.
    const summary=estimates.length>1;
    return {estimates,summary,uncalibrated:!calibrated,
      measured:calibrated&&estimates.length>0,
      sensitivity:calibrated&&!summary?estimates[0]||null:null,
      dashed:edge.kind==='influences'&&(!calibrated||!estimates.some(resolvedEstimate))};
  }
  function presentationForVariants(edge,variants,fallbackContrast=null) {
    const paths=v=>allEffects(v).filter(e=>e.edge_id===edge.id);
    const owners=variants.filter(v=>(v.path_edge_ids||[]).includes(edge.id)||paths(v).length);
    const uncalibrated=edge.kind==='influences'&&owners.some(v=>!paths(v).length||paths(v).some(e=>e.context_approximation));
    const contrastIds=owners.length?unique(owners.flatMap(paths).map(e=>e.contrast_id)):fallbackContrast?[fallbackContrast]:null;
    return influencePresentation(edge,{contrastIds,uncalibrated});
  }
  function intentCoverage(graph,plan,appIds) {
    // Related KPI ownership qualifies a group. Numerical interval compatibility
    // is resolved before enumeration, never by maximize/minimize string checks.
    const apps=new Map(graph.nodes.filter(n=>n.kind==='xapp').map(n=>[n.id,n]));
    const required=unique(plan.ranges.map(r=>r.kpi)),contributors=Object.fromEntries(required.map(k=>[k,[]]));
    const responsibilities=Object.fromEntries(appIds.map(id=>[id,[]]));
    for(const request of plan.ranges){
      for(const id of appIds){
        const related=Object.hasOwn(apps.get(id)?.target_kpis||{},request.kpi);
        if(related){contributors[request.kpi].push(id);responsibilities[id].push(request.kpi);}
      }
    }
    const covered=required.filter(k=>contributors[k].length),missing=required.filter(k=>!contributors[k].length);
    const unrelated=appIds.filter(id=>!responsibilities[id].length);
    const complete=required.length>0&&!missing.length;
    return {required_kpis:required,covered_kpis:covered,missing_kpis:missing,contributors,
      unrelated_apps:unrelated,relevant:appIds.length>0&&!unrelated.length,complete,
      eligible:complete&&!unrelated.length&&!plan.questions?.length&&!plan.ranges.some(r=>r.empty),
      basis:'kpi_ownership_and_intersected_ranges',attainment_verified:false};
  }
  function enumerateCombinations(graph,frame,plan,{showDashed=false}={}) {
    const result={schema:'oran.intent-combinations.v1',snapshot:{...plan.snapshot},show_dashed:showDashed,
      selected_by_default:false,deployment_enabled:false,groups:[],group_count:0,eligible_group_count:0,assignment_count:0,
      required_kpis:unique(plan.ranges.map(r=>r.kpi)),
      note:'Raw enumeration retains all matched candidates. Visible groups must relate to every KPI in the intersected active intent ranges, without unrelated apps. Objective direction labels do not determine interval compatibility or attainment; conflicts are action-space evidence.'};
    const check=review(graph,frame,plan,{});
    if(check.unavailable){result.unavailable=check.unavailable;return result;}
    if(plan.questions?.length||plan.ranges.some(r=>r.empty))return result;
    const entries=plan.candidates.map(c=>({...c,variants:c.variants.filter(Boolean)})).filter(c=>c.variants.length);
    const nodes=new Map(graph.edges.map(e=>[e.id,e])),singles=new Map(),pairs=new Map(),visibility=new Map(),groups=new Map(),summaries=new Map();
    for(const c of entries)for(const v of c.variants){
      const r=review(graph,frame,plan,{[c.app_id]:v.id});
      if(r.unavailable){result.unavailable=r.unavailable;return result;}
      singles.set(v.id,r);
    }
    for(let i=0;i<entries.length;i++)for(let j=i+1;j<entries.length;j++)for(const a of entries[i].variants)for(const b of entries[j].variants){
      const r=review(graph,frame,plan,{[entries[i].app_id]:a.id,[entries[j].app_id]:b.id});
      if(r.unavailable){result.unavailable=r.unavailable;return result;}
      pairs.set(a.id+'|'+b.id,r.conflicts);
    }
    function emit(chosen) {
      if(!chosen.length)return;
      const selection=Object.fromEntries(chosen.map(x=>[x.c.app_id,x.v.id])),variants=chosen.map(x=>x.v),conflicts=[];
      for(let i=0;i<chosen.length;i++)for(let j=i+1;j<chosen.length;j++)conflicts.push(...pairs.get(chosen[i].v.id+'|'+chosen[j].v.id));
      const range_risks=chosen.flatMap(x=>singles.get(x.v.id).range_risks);
      const support=unique([...conflicts.flatMap(c=>c.edge_ids),...range_risks.flatMap(r=>r.effects.map(e=>e.edge_id))]);
      const hidden=support.filter(id=>{
        const edge=nodes.get(id);if(!edge||edge.kind!=='influences')return false;
        const owners=variants.filter(v=>(v.path_edge_ids||[]).includes(id)||allEffects(v).some(e=>e.edge_id===id));
        const key=id+'|'+owners.map(v=>v.id).join('|');
        if(!visibility.has(key))visibility.set(key,presentationForVariants(edge,owners).dashed);
        return visibility.get(key);
      });
      const visible=project({conflicts,range_risks},showDashed,hidden);
      const covered=unique(variants.flatMap(v=>v.matches.map(m=>m.kpi))).filter(k=>result.required_kpis.includes(k));
      const full=result.required_kpis.length>0&&result.required_kpis.every(k=>covered.includes(k));
      const row={id:'assignment:'+result.assignment_count++,selection,covered_kpis:covered,full_coverage:full,
        direct_count:visible.conflicts.filter(c=>c.type==='direct').length,indirect_count:visible.conflicts.filter(c=>c.type==='indirect').length,
        range_risk_count:visible.range_risks.length,
        raw_direct_count:conflicts.filter(c=>c.type==='direct').length,raw_indirect_count:conflicts.filter(c=>c.type==='indirect').length};
      const app_ids=chosen.map(x=>x.c.app_id),key=app_ids.join('|');
      if(!groups.has(key)){groups.set(key,{id:'group:'+key,app_ids,assignments:[],assignment_count:0,full_coverage_count:0,direct_assignments:0,indirect_assignments:0,full_without_visible_conflicts:0});summaries.set(key,new Map());}
      const group=groups.get(key);group.assignments.push(row);group.assignment_count++;
      group.full_coverage_count+=Number(full);group.direct_assignments+=Number(row.direct_count>0);group.indirect_assignments+=Number(row.indirect_count>0);
      group.full_without_visible_conflicts+=Number(full&&row.direct_count===0&&row.indirect_count===0);
      const seen=new Map(),index=summaries.get(key);
      for(const conflict of visible.conflicts){
        const apps=conflict.type==='direct'?[...conflict.apps].sort():conflict.apps;
        const identity=conflict.type+'|'+apps.join('|');
        if(!index.has(identity))index.set(identity,{type:conflict.type,apps,parameters:new Set(),kpis:new Set(),statuses:new Set(),assignments:0,certain_assignments:0});
        const summary=index.get(identity);
        for(const p of conflict.parameters)summary.parameters.add(p);
        for(const k of conflict.kpis)summary.kpis.add(k);
        summary.statuses.add(conflict.status);
        seen.set(identity,(seen.get(identity)||false)||['incompatible_writes','measured_path'].includes(conflict.status));
      }
      for(const [identity,certain]of seen){const summary=index.get(identity);summary.assignments++;summary.certain_assignments+=Number(certain);}
    }
    function walk(index,chosen) {
      if(index===entries.length){emit(chosen);return;}
      walk(index+1,chosen);
      for(const v of entries[index].variants)walk(index+1,[...chosen,{c:entries[index],v}]);
    }
    walk(0,[]);
    result.groups=[...groups.values()].sort((a,b)=>a.app_ids.length-b.app_ids.length||a.app_ids.join('|').localeCompare(b.app_ids.join('|')));
    for(const group of result.groups){
      group.conflict_summaries=[...summaries.get(group.app_ids.join('|')).values()].map(s=>({...s,parameters:[...s.parameters],kpis:[...s.kpis],statuses:[...s.statuses],action_dependent:s.assignments<group.assignment_count}));
      group.intent_coverage=intentCoverage(graph,plan,group.app_ids);
    }
    result.group_count=result.groups.length;
    result.eligible_group_count=result.groups.filter(g=>g.intent_coverage.eligible).length;
    return result;
  }
  function project(review,showDashed=false,hiddenEdges=[]) {
    const hidden=new Set(hiddenEdges);
    const conflicts=showDashed?review.conflicts:review.conflicts.filter(c=>c.type==='direct'||!['scope_uncertain','uncertain_path'].includes(c.status)&&!c.edge_ids.some(id=>hidden.has(id)));
    const range_risks=showDashed?review.range_risks:review.range_risks.map(r=>({...r,effects:r.effects.filter(e=>e.evidence==='resolved_average'&&!e.scope_approximation&&!hidden.has(e.edge_id))})).filter(r=>r.effects.length);
    return {...review,conflicts,range_risks,view:{show_dashed:showDashed}};
  }
  return {review,defaults,catalog,resolvedEstimate,parameterKpiDashed,influencePresentation,presentationForVariants,intentCoverage,rangePressure,enumerateCombinations,selectionKey:stable,project,intentCases};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=SmoConflicts;
