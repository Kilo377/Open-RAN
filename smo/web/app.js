/* Structural graph presentation only. The browser does not load action templates. */
if(typeof module!=='undefined'&&module.exports)module.exports=require('./legacy_action_review.js');
var SmoGraph = (() => {
  'use strict';
  // Bundles and their graph/frame records are immutable JSON snapshots. Cache
  // their exact serialization by object identity rather than repeatedly
  // serializing the calibrated graph during every render.
  const snapshotFingerprints=new WeakMap();
  const fingerprint=value=>{
    if(!value||typeof value!=='object')return JSON.stringify(value??null);
    if(!snapshotFingerprints.has(value))snapshotFingerprints.set(value,JSON.stringify(value));
    return snapshotFingerprints.get(value);
  };
  function sourceIdentity(bundle){
    const s=bundle?.source||{};
    return JSON.stringify([s.mode??null,s.run_id??null,s.source_sha256??null,s.name??null,
      s.deployment_job_id??null,s.evaluation_origin_id??null]);
  }
  function decisionPreviewContextKey(bundle,frame){
    return JSON.stringify([sourceIdentity(bundle),bundle?.graph_sha256??null,frame?.slot??null]);
  }
  function quantificationContextKey(bundle,frame){
    return JSON.stringify([sourceIdentity(bundle),fingerprint(bundle?.graph),fingerprint(frame)]);
  }
  function frozenIntentReference(row,frame){
    const finite=x=>typeof x==='number'&&Number.isFinite(x);
    const at=(a,k)=>finite(a.snapshot?.[k])?a.snapshot[k]:Infinity;
    const anchors=(row.anchors||[]).filter(a=>finite(a.current)).slice().sort((a,b)=>{
      for(const key of ['time_ms','slot'])if(at(a,key)!==at(b,key))return at(a,key)<at(b,key)?-1:1;
      const left=String(a.intent_id||''),right=String(b.intent_id||'');
      return left<right?-1:left>right?1:a.current-b.current;
    });
    const current=frame?.states?.['kpi:'+row.kpi]?.value;
    return anchors.length?anchors[0].current:finite(row.current)?row.current:finite(current)?current:null;
  }
  const intentCases = [
    {id:'ue_power_mec_budget_band',source:'User-defined intent',
      text:'所有 UE 的总平均功率降低至少 2%，同时 MEC 总预期计算需求不超过 1.8 GFLOP/s。',
      kpis:['ue_power_w','mec_expected_gflops'],
      baselineNote:'功率降幅以点击分析时的当前值为基准。'},
    {id:'ue_power_mec_delay_band',source:'User-defined intent',
      text:'所有 UE 的总平均功率降低至少 1%，同时 MEC 卸载任务平均端到端时延不超过 15 ms。',
      kpis:['ue_power_w','mec_offloaded_latency_ms'],
      baselineNote:'功率降幅以点击分析时的当前值为基准。'},
    {id:'bs_power_mec_delay_band',source:'User-defined intent',
      text:'基站总平均功率降低至少 2%，同时 MEC 卸载任务平均端到端时延不超过 15 ms。',
      kpis:['ap_power_w','mec_offloaded_latency_ms'],
      baselineNote:'功率降幅以点击分析时的当前值为基准。'},
    {id:'c04',source:'C-04 / I-15',
      text:'提高全网下行吞吐量，同时降低所有基站的总功率。',
      kpis:['network_goodput_mbps','ap_power_w'],
      note:''},
    {id:'c06',source:'C-06, scheduling subset',
      text:'提高全网下行吞吐量，同时降低 URLLC 数据包的超时与丢弃事件率。',
      kpis:['network_goodput_mbps','urllc_failure_events_per_s'],
      note:'请使用存在 URLLC 压力的场景。'},
    {id:'c02',source:'C-02, M1 mechanism adaptation',
      text:'降低所有 UE 的总功率，同时降低 MEC 业务的总预期计算需求。',
      kpis:['ue_power_w','mec_expected_gflops'],
      note:'X-13 目前仅控制部分 MEC 业务。'},
    {id:'i16',source:'I-16, request form from C-08',
      text:'降低所有 UE 的总功率，同时降低 MEC 卸载任务的平均端到端时延。',
      kpis:['ue_power_w','mec_offloaded_latency_ms'],
      note:'最新的 X-12-MEC 模型仍需独立仿真验证。'},
    {id:'lc02',source:'LC-02, M5 interval adaptation',
      text:'要求同一观测窗口内的全网下行吞吐量，同时严格高于和严格低于其当前值。',
      kpis:['network_goodput_mbps'],
      note:'同一 KPI 在同一观测窗口内的要求互相矛盾。'},
    {id:'c03',source:'C-03 / I-19, negative-control adaptation',needsThreshold:true,
      text:'在保持全网下行吞吐量的同时，将基站总功率降至经过验证的可达最低值以下。',
      kpis:['ap_power_w','network_goodput_mbps'],
      note:'分析前，请填写经过本网络验证的基站功率限值。'},
  ];
  function validatePresetEdit(edit) {
    if(!edit||typeof edit.text!=='string'||!edit.text.trim())throw new Error('请输入完整的意图内容。');
    if(edit.text.trim().length>2000)throw new Error('意图内容不能超过 2000 个字符。');
    const tag=edit.tag??'';
    if(typeof tag!=='string')throw new Error('备注标签请填写文字，或留空。');
    if(tag.trim().length>120)throw new Error('备注标签不能超过 120 个字符。');
    // Free-form memo only. It never supplies classifications or KPI ranges.
    return {text:edit.text.trim(),tag:tag.trim()};
  }
  function customIntentCases(edits={},deletedIds=[]) {
    const deleted=new Set(Array.isArray(deletedIds)?deletedIds:[]);
    return intentCases.filter(original=>!deleted.has(original.id)).map(original=>{
      const base={...original,kpis:[...original.kpis]};
      if(!edits||!Object.prototype.hasOwnProperty.call(edits,base.id))return base;
      let edit;try{edit=validatePresetEdit(edits[base.id]);}catch{return base;}
      if(edit.text===base.text&&edit.tag===(base.tag||''))return base;
      if(edit.text===base.text)return {...base,tag:edit.tag,customized:true};
      // Edited text must be interpreted afresh; old KPI hints and guards may no
      // longer describe it. Saved active intents are separate immutable records.
      return {id:base.id,source:'Browser preset edit',...edit,kpis:[],customized:true,
        baselineNote:'点击分析后，根据当前快照解析 KPI 与目标范围。'};
    });
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
    // A relationship may exist in one measured context and not another.
    // The graph shows this conditional evidence without selecting an action.
    const summary=estimates.length>1;
    const measured=calibrated&&estimates.some(s=>typeof s.estimate==='number'&&Number.isFinite(s.estimate)||s.entity_estimates&&Object.values(s.entity_estimates).some(v=>typeof v==='number'&&Number.isFinite(v)));
    const declared=!!Object.keys(edge.objectives||{}).length;
    const confidenceResolved=estimates.some(resolvedEstimate);
    const observedZero=measured&&estimates.every(s=>s.zero_in_both||s.fit_diagnostics?.assessment==='constant_response_observed'||s.estimate===0&&s.confidence_interval?.every(v=>v===0));
    return {estimates,summary,uncalibrated:!calibrated,
      measured,declared,confidenceResolved,observedZero,
      sensitivity:calibrated&&!summary?estimates[0]||null:null,
      dashed:!measured&&!declared};
  }
  function influenceMagnitude(estimates) {
    if(!estimates.length)return null;
    const finite=v=>typeof v==='number'&&Number.isFinite(v);
    const normalized=estimates.every(s=>finite(s.importance?.signed_reference_importance));
    // Display magnitudes only. Keep each signed estimate intact for dynamics.
    // Never average unrelated control transitions or combine gain units.
    if(!normalized&&(!estimates.every(s=>finite(s.estimate))||new Set(estimates.map(s=>s.unit)).size>1))return null;
    const values=estimates.map(s=>Math.abs(normalized?s.importance.signed_reference_importance:s.estimate));
    return {lower:Math.min(...values),upper:Math.max(...values),basis:normalized?'normalized_importance':'gain_magnitude',unit:normalized?'':estimates[0].unit||''};
  }
  function gainLabelLayout(items,minY,maxY,minGap=19){
    if(!items.length)return [];
    const rows=items.map(x=>({...x})).sort((a,b)=>a.y-b.y||a.id.localeCompare(b.id));
    const gap=rows.length>1?Math.min(minGap,(maxY-minY)/(rows.length-1)):0;
    for(let i=0;i<rows.length;i++)rows[i].labelY=Math.max(rows[i].y,minY,i?rows[i-1].labelY+gap:minY);
    rows[rows.length-1].labelY=Math.min(rows[rows.length-1].labelY,maxY);
    for(let i=rows.length-2;i>=0;i--)rows[i].labelY=Math.min(rows[i].labelY,rows[i+1].labelY-gap);
    return rows;
  }
  function constraintIssuesForRecord(issues,id) {
    return (issues||[]).filter(issue=>issue.intent_ids?.includes(id));
  }
  function hasBlockingConstraints(result) {
    return (result.constraint_issues||[]).some(issue=>issue.blocking);
  }
  function intentFeedback(result,{submitted=false}={}) {
    if(result.questions?.length)return submitted
      ? {text:result.questions.join(' '),problem:true}
      : {text:'已有意图需要补充说明，请查看联合分析结果。',problem:false};
    const blocked=hasBlockingConstraints(result);
    if(blocked)return {text:(submitted?'本次意图已解析并保存':'已更新观测')+'；共同约束需要处理，请查看标注的意图。',problem:false};
    if(!submitted)return {text:'已更新观测，原意图基准保持不变。',problem:false};
    const prefix='本次意图已解析并保存。';
    if(result.candidates?.length)return {text:prefix+'已更新覆盖全部 KPI 的组合。',problem:false};
    const ranges=result.ranges||[];
    return {text:prefix+(ranges.some(r=>r.current==null)?'当前缺少匹配所需的测量值。':ranges.every(r=>r.status==='within_range')?'当前值已满足这些目标范围。':'当前图中没有关联覆盖这些 KPI 的 xApp 组合。'),problem:false};
  }
  function previewGroups(app) {
    const stable=value=>value&&typeof value==='object'?Array.isArray(value)?value.map(stable):Object.fromEntries(Object.keys(value).sort().map(k=>[k,stable(value[k])])):value;
    const groups=new Map();
    for(const row of app?.actions||[]){
      const key=JSON.stringify(stable({parameter:row.parameter,current:row.current,proposed:row.proposed,changed:row.changed}));
      if(!groups.has(key))groups.set(key,{...row,scopes:[]});
      groups.get(key).scopes.push({cell_id:row.cell_id,ue_id:row.ue_id||null});
    }
    return [...groups.values()];
  }
  function batchCatalogCompatible(batch,catalog) {
    if(!batch||!catalog||!Array.isArray(batch.members)||!Array.isArray(catalog.groups))return false;
    const keys=groups=>groups.map(g=>[...g.app_ids].sort().join('|')).sort();
    // A retained empirical batch cannot replace a newly expanded graph catalog.
    return JSON.stringify(keys(batch.members))===JSON.stringify(keys(catalog.groups));
  }
  function initialSelectionSummary(catalog) {
    const s=catalog?.initial_selection;
    if(!['influence_coverage_v1','semantic_influence_coverage_v2'].includes(s?.method))return null;
    const semantic=s.method==='semantic_influence_coverage_v2';
    return {primary:s.primary_group_count,reserve:s.reserve_group_count,total:s.input_group_count,
      text:`${semantic?'能力方向＋统计影响初选':'统计影响初选'} · 门槛 ${Number((s.config.coverage_threshold*100).toFixed(2))}% · 原始 ${s.input_group_count} 组 → 主候选 ${s.primary_group_count} 组 · 备选 ${s.reserve_group_count} 组${s.excluded_group_count?' · 反向直接冲突剔除 '+s.excluded_group_count+' 组':''}。仅评估主候选；导出列表包含备选与剔除记录。`};
  }
  function quantificationGroups(review,catalog,showDashed=false) {
    const independent=review?.[showDashed?'quantification_catalog_with_uncertain':'quantification_catalog'];
    if(independent)return independent.groups||[];
    // Compatibility for older exported reviews. Live service supplies a
    // separately enumerated structural catalog, not the admitted group list.
    const groups=new Map();
    for(const [key,disposition]of [['groups','candidate'],['reserve_groups','reserve'],['excluded_groups','excluded']])
      for(const group of catalog?.[key]||[])if(!groups.has(group.id))groups.set(group.id,{...group,deployment_disposition:disposition});
    return [...groups.values()];
  }
  function retainedEvidenceCompatible(batch,catalog,graph,activeIds) {
    if(!batch?.context?.contracts||!graph?.contract_sha256||!catalog||!activeIds?.length)return false;
    const ids=x=>[...x].sort().join('|');
    const stable=x=>JSON.stringify(Object.entries(x).sort(([a],[b])=>a.localeCompare(b)));
    const ranges=x=>(x||[]).map(r=>[r.kpi,r.lower,r.upper,r.lower_inclusive,r.upper_inclusive]).sort(([a],[b])=>a.localeCompare(b));
    return ids(batch.active_intent_ids||[])===ids(activeIds)
      && batch.context.metric_scope===graph.metric_scope
      && stable(batch.context.contracts)===stable(graph.contract_sha256)
      && JSON.stringify(ranges(batch.ranges))===JSON.stringify(ranges(catalog.ranges));
  }
  const metricDisplayValue=(key,value)=>key==='urllc_failure_fraction'&&typeof value==='number'?value*100:value;
  const metricDisplayUnit=(key,unit)=>key==='urllc_failure_fraction'?'%':unit;
  function evaluationPresentation(e){
    if(!e?.after?.complete)return {tone:'unavailable',label:'数据不足'};
    if(e.no_regression===false)return {tone:'regressed',label:'存在目标退步'};
    return e.after.all_satisfied?{tone:'met',label:'全部目标达标'}:
      e.after.score>0?{tone:'partial',label:'部分改善，尚未全部达标'}:{tone:'neutral',label:'无净改善，尚未全部达标'};
  }
  return {resolvedEstimate,parameterKpiDashed,influencePresentation,influenceMagnitude,gainLabelLayout,sourceIdentity,decisionPreviewContextKey,quantificationContextKey,frozenIntentReference,intentCases,validatePresetEdit,customIntentCases,constraintIssuesForRecord,hasBlockingConstraints,intentFeedback,previewGroups,batchCatalogCompatible,initialSelectionSummary,quantificationGroups,retainedEvidenceCompatible,metricDisplayValue,metricDisplayUnit,evaluationPresentation};
})();

// Read-only conflict proxies. A reference drive is NOT a KPI forecast.
var SmoConflictRisk = (() => {
  'use strict';
  const method='intent_control_dynamics_v4';
  const finite=v=>typeof v==='number'&&Number.isFinite(v);
  const stable=v=>v&&typeof v==='object'?Array.isArray(v)?v.map(stable):Object.fromEntries(Object.keys(v).sort().map(k=>[k,stable(v[k])])):v;
  const equal=(a,b)=>JSON.stringify(stable(a))===JSON.stringify(stable(b));
  const sign=v=>v===0?0:Math.sign(v);
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const pairs=a=>a.flatMap((x,i)=>a.slice(i+1).map(y=>[x,y]));
  const value=(frame,p)=>frame.states?.['rcp:'+p]?.by_entity||{};
  function conflictCounts(summaries){
    return Object.fromEntries(['direct','indirect','implicit'].map(type=>[type,new Set(summaries.filter(s=>s.type===type&&new Set(s.apps||[]).size>=2).map(s=>[...new Set(s.apps)].sort().join('|'))).size]));
  }
  function members(row,frame){
    if(row.members?.length)return row.members;
    const part=row.parameter_component;
    if(part==='all_cells')return Object.keys(value(frame,'cell_sleep_mode'));
    if(part==='mec_workload'||part==='mec_priority')return Object.keys(value(frame,'ue_offload_fraction'));
    if(part==='drx_enabled_population')return Object.keys(value(frame,'ue_drx_configs')).filter(u=>value(frame,'ue_drx_configs')[u]?.enabled);
    if(part==='urllc_priority')return Object.keys(frame.states?.['kpi:urllc_failure_fraction']?.by_entity||{});
    return [];
  }
  const resolved=row=>row.confidence_interval?.[0]>0||row.confidence_interval?.[1]<0;
  const observedZero=row=>row.estimate===0&&row.confidence_interval?.[0]===0&&row.confidence_interval?.[1]===0;
  function intentRole(direction,row,current){
    if(direction===null)return 'context_unavailable';
    if(direction===0)return 'neutral';
    if(!finite(current))return 'context_unavailable';
    const low=row.lower,high=row.upper;
    if(finite(low)&&low===high&&current===low)return 'against_interval';
    if(finite(low)&&current<=low)return direction>0?'toward_interval':'against_interval';
    if(finite(high)&&current>=high)return direction<0?'toward_interval':'against_interval';
    const down=finite(low)?current-low:Infinity,up=finite(high)?high-current:Infinity;
    if(down===up)return 'inside_interval_no_unique_direction';
    return (down<up?direction<0:direction>0)?'consuming_margin':'building_margin';
  }
  function project(edge,base,target,frame,showDashed,actionContext=null){
    const measured=(edge.sensitivities?.length?edge.sensitivities:edge.sensitivity?[edge.sensitivity]:[]).filter(r=>finite(r.estimate));
    // A weight measured under QoS-deadline scheduling is not an additional
    // latency-aware weight. Select its measured operating context before
    // summing population directions; legacy unconditioned rows are unchanged.
    const all=measured.filter(row=>{
      const expected=row.conditioning?.scheduler_mode;
      if(!expected)return true;
      const ids=members(row,frame).filter(u=>!equal(base[u],target[u]));
      if(!ids.length)return true;
      return ids.every(u=>{
        const action=actionContext?.actions?.find(a=>a.parameter==='ue_scheduler_weights'&&a.ue_id===u);
        if(!action?.cell_id)return false;
        const modes=actionContext.target?.scheduler_mode||value(frame,'scheduler_mode');
        return modes[action.cell_id]===expected;
      });
    });
    // Numerical confidence stays attached to the reference weight. It does
    // not remove a measured control chain or turn an unknown effect into zero.
    const rows=all;
    const changed=Object.keys({...base,...target}).filter(u=>!equal(base[u],target[u]));
    if(!changed.length)return {drive:0,direction:0,status:'unchanged',notes:[],calibrations:[],
      action_coverage:{complete:true,changed_entities:[],covered_entities:[],uncovered_entities:[],uncovered_components:{}}};
    const notes=new Set(),pieces=[],used=[],components=[],covered=new Set(),coveredFields=new Map();
    const parameter=edge.source.slice(4);
    const actionCoverage=()=>{
      const uncovered_entities=changed.filter(u=>!covered.has(u)),uncovered_components={};
      if(parameter==='ue_drx_configs')for(const u of changed){
        const fields=Object.keys({...base[u],...target[u]}).filter(k=>!equal(base[u]?.[k],target[u]?.[k])&&!coveredFields.get(u)?.has(k));
        if(fields.length)uncovered_components[u]=fields;
      }
      return {complete:!uncovered_entities.length&&!Object.keys(uncovered_components).length,
        changed_entities:[...changed],covered_entities:changed.filter(u=>covered.has(u)&&!uncovered_components[u]),
        uncovered_entities,uncovered_components};
    };
    const contextNote=row=>{
      const expected=row.reference_scheduler_mode||(row.parameter_component==='mec_priority'?'latency_aware':row.parameter_component==='urllc_priority'?'qos_deadline':'round_robin');
      if(Object.values(value(frame,'scheduler_mode')).some(mode=>mode!==expected))notes.add('reference_context_shift');
      if(Object.values(value(frame,'cell_sleep_mode')).some(mode=>mode!=='active'))notes.add('reference_context_shift');
    };
    if(parameter==='cell_sleep_mode'){
      const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
      const profiles=all.filter(r=>r.kind==='finite_contrast'&&object(r.from)&&object(r.to));
      const modes=new Set(['active','micro_sleep','light_sleep','deep_sleep']);
      const actualKeys=Object.keys({...base,...target}).sort(),matches=[];
      for(const row of profiles){
        const ids=row.members,fromKeys=Object.keys(row.from).sort(),toKeys=Object.keys(row.to).sort();
        if(!Array.isArray(ids)||!ids.length||new Set(ids).size!==ids.length||
          !equal([...ids].sort(),actualKeys)||!equal(fromKeys,actualKeys)||!equal(toKeys,actualKeys))continue;
        if(![...Object.values(row.from),...Object.values(row.to)].every(mode=>modes.has(mode)))continue;
        const forward=ids.every(id=>equal(base[id],row.from[id])&&equal(target[id],row.to[id]));
        const reverse=ids.every(id=>equal(base[id],row.to[id])&&equal(target[id],row.from[id]));
        if(forward||reverse)matches.push({row,step:forward?1:-1});
      }
      if(matches.length){
        const drive=matches[0].row.estimate*matches[0].step;
        if(matches.some(m=>m.row.estimate*m.step!==drive))return {drive:null,direction:null,status:'unsupported',
          notes:['ambiguous_joint_sleep_profile_reference'],calibrations:matches.map(m=>m.row.contrast_id)};
        const {row,step}=matches[0];contextNote(row);
        if(step<0)notes.add('common_reference_contrast');
        if(observedZero(row))notes.add('observed_zero_reference_weight');else if(!resolved(row))notes.add('uncertain_gain_direction');
        notes.add('exact_joint_sleep_profile_reference');notes.add('local_reference_drive_not_kpi_change');
        return {drive,direction:sign(drive),status:'joint_profile_reference_projection',notes:[...notes],
          calibrations:matches.map(m=>m.row.contrast_id),
          components:[{kind:'exact_joint_cell_sleep_profile_reference_weight',gain:row.estimate,parameter_step:step,
            reference_drive:drive,calibration_id:row.contrast_id,members:[...row.members],
            from:structuredClone(row.from),to:structuredClone(row.to)}]};
      }
      if(profiles.length)notes.add('joint_sleep_profile_not_applicable');
    }
    // Prefer an explicitly measured structured profile when the actual action
    // matches it. Do not add its constituent scalar fits a second time.
    if(parameter==='ue_drx_configs')for(const row of all.filter(r=>r.kind==='finite_contrast'&&r.from&&r.to&&typeof r.from==='object'&&typeof r.to==='object')){
      const keys=Object.keys(row.from),ids=members(row,frame);
      if(!keys.length||!equal([...keys].sort(),Object.keys(row.to).sort())||!ids.some(u=>changed.includes(u)))continue;
      if(ids.every(u=>keys.every(k=>coveredFields.get(u)?.has(k))))continue;
      const steps=[];let applicable=true;
      for(const u of ids){
        if(equal(base[u],target[u])){steps.push(0);continue;}
        if(!base[u]||!target[u]||Object.keys({...base[u],...target[u]}).some(k=>!keys.includes(k)&&!equal(base[u][k],target[u][k]))){applicable=false;break;}
        const forward=keys.every(k=>equal(base[u][k],row.from[k])&&equal(target[u][k],row.to[k]));
        const reverse=keys.every(k=>equal(base[u][k],row.to[k])&&equal(target[u][k],row.from[k]));
        if(!forward&&!reverse){applicable=false;break;}
        steps.push(forward?1:-1);
      }
      if(!applicable)continue;
      contextNote(row);const step=mean(steps);
      pieces.push(row.estimate*step);used.push(row.contrast_id);
      components.push({kind:'structured_profile_reference_weight',gain:row.estimate,parameter_step:step,reference_drive:row.estimate*step,calibration_id:row.contrast_id,from:row.from,to:row.to});
      ids.forEach(u=>{covered.add(u);if(!coveredFields.has(u))coveredFields.set(u,new Set());keys.forEach(k=>coveredFields.get(u).add(k));});
      if(!steps.every(v=>v===steps[0]))notes.add('population_direction_projection');
      if(steps.some(v=>v<0))notes.add('common_reference_contrast');
      if(observedZero(row))notes.add('observed_zero_reference_weight');else if(!resolved(row))notes.add('uncertain_gain_direction');
    }
    const rangeRows=rows.filter(r=>r.kind!=='finite_contrast');
    for(const row of rangeRows){
      const ids=members(row,frame),component=parameter==='ue_drx_configs'?(row.scalar_component||'inactivity_ms'):null;
      if(!ids.length||!ids.some(u=>changed.includes(u)))continue;
      const old=ids.map(u=>component?base[u]?.[component]:base[u]);
      const next=ids.map((u,i)=>component&&coveredFields.get(u)?.has(component)?old[i]:component?target[u]?.[component]:target[u]);
      if(!old.every(finite)||!next.every(finite))continue;
      if(old.every((v,i)=>v===next[i]))continue;
      contextNote(row);
      const domain=row.fit_domain||row.sampled_domain;
      if(domain&&[...old,...next].some(v=>v<domain[0]||v>domain[1]))notes.add('outside_reference_domain');
      // Orthogonal projection onto the measured common-population direction.
      // This is a risk feature, not an identified per-UE causal derivative.
      const steps=next.map((v,i)=>v-old[i]),step=mean(steps);
      const residual=Math.hypot(...steps.map(d=>d-step));
      if(residual>1e-9)notes.add('population_direction_projection');
      if(observedZero(row))notes.add('observed_zero_reference_weight');else if(!resolved(row))notes.add('uncertain_gain_direction');
      if(row.fit_diagnostics?.assessment==='limited_first_order_approximation')notes.add('limited_reference_fit');
      pieces.push(row.estimate*step);used.push(row.contrast_id);components.push({kind:'gain_times_step',gain:row.estimate,parameter_step:step,reference_drive:row.estimate*step,calibration_id:row.contrast_id});ids.forEach(u=>{covered.add(u);if(component){if(!coveredFields.has(u))coveredFields.set(u,new Set());coveredFields.get(u).add(component);}});
    }
    // Compose measured category differences while retaining their uncertainty.
    const categorical=all.filter(r=>r.kind==='finite_contrast'&&typeof r.from==='string'&&typeof r.to==='string');
    if(categorical.length){
      const anchors=[...new Set(categorical.map(r=>r.from))];
      if(anchors.length===1){
        const anchor=anchors[0],potentials={[anchor]:{estimate:0,confidence_interval:[0,0]}};let ambiguous=false;
        for(const row of categorical){if(row.to in potentials&&!equal(potentials[row.to],row))ambiguous=true;potentials[row.to]=row;}
        const ids=members(categorical[0],frame),steps=[],needed=[];
        let admissible=!ambiguous&&ids.length>0,uncertain=false;
        for(const u of ids){
          if(base[u]===target[u]){steps.push(0);continue;}
          const left=potentials[base[u]],right=potentials[target[u]];
          if(!left||!right){admissible=false;break;}
          const delta=right.estimate-left.estimate;
          const envelope=left.confidence_interval&&right.confidence_interval?
            [right.confidence_interval[0]-left.confidence_interval[1],right.confidence_interval[1]-left.confidence_interval[0]]:null;
          const directionResolved=envelope&&(envelope[0]>0||envelope[1]<0);
          const sameObservedResponse=delta===0&&envelope?.[0]===0&&envelope?.[1]===0;
          uncertain ||= !directionResolved&&!sameObservedResponse;
          if(sameObservedResponse)notes.add('observed_zero_reference_weight');
          steps.push(delta);needed.push(left,right);
        }
        if(admissible&&(!uncertain||!edge.statistical_selection||edge.statistical_selection.action_context_policy==='retain_valid_measured')){
          pieces.push(mean(steps));components.push({kind:'common_reference_category_difference',reference_drive:mean(steps),transitions:ids.filter(u=>base[u]!==target[u]).map(u=>({entity:u,from:base[u],to:target[u]}))});needed.filter(r=>r.contrast_id).forEach(r=>used.push(r.contrast_id));ids.forEach(u=>covered.add(u));
          if(ids.some(u=>base[u]!==anchor&&base[u]!==target[u]))notes.add('common_reference_contrast');
          if(!steps.every(v=>v===steps[0]))notes.add('population_direction_projection');
          if(uncertain)notes.add('uncertain_gain_direction');
        }else if(!edge.statistical_selection&&parameter==='cell_sleep_mode'&&ids.length){
          const row=rows.find(r=>r.kind==='finite_contrast'&&r.from==='active'&&/sleep/.test(r.to)&&r.estimate!==0);
          const steps=row&&ids.map(u=>base[u]===target[u]?0:base[u]==='active'&&/sleep/.test(target[u])?1:target[u]==='active'&&/sleep/.test(base[u])?-1:null);
          if(steps&&steps.every(finite)&&steps.some(v=>v!==0))return {drive:row.estimate*mean(steps),direction:sign(row.estimate*mean(steps)),status:'sleep_family_reference_weight',
            notes:[...notes,'sleep_family_reference_weight','unmeasured_sleep_profile','reference_context_approximation','local_reference_drive_not_kpi_change',...(!resolved(row)?['uncertain_gain_direction']:[])],calibrations:[row.contrast_id],
            components:[{kind:'binary_sleep_reference_weight',gain:row.estimate,parameter_step:mean(steps),reference_drive:row.estimate*mean(steps),calibration_id:row.contrast_id,assumption:'active_vs_sleep_family_not_profile_specific'}],
            action_coverage:{...actionCoverage(),unmeasured_profile:true}};
        }
      }
    }
    if(parameter==='ue_drx_configs'&&changed.some(u=>Object.keys({...base[u],...target[u]}).some(k=>!equal(base[u]?.[k],target[u]?.[k])&&!coveredFields.get(u)?.has(k))))notes.add('uncovered_structured_components');
    if(!pieces.length){
      return {drive:null,direction:null,status:'unsupported',notes:[...notes,'reference_direction_unavailable'],calibrations:[],action_coverage:actionCoverage()};
    }
    if(changed.some(u=>!covered.has(u)))notes.add('uncovered_population_members');
    const drive=pieces.reduce((a,b)=>a+b,0);
    notes.add('local_reference_drive_not_kpi_change');
    return {drive,direction:sign(drive),status:'reference_projection',notes:[...notes],calibrations:[...new Set(used)],components,action_coverage:actionCoverage()};
  }
  function screenedProjection(audit,edge,base,target,frame){
    // This is an applicability audit, never an alternate gain projector.
    // A screened measurement may be omitted from the active KG model, while
    // an unmeasured action/profile remains unknown. No rejected coefficient
    // or confidence interval contributes to a drive or a score here.
    const changed=Object.keys({...base,...target}).filter(u=>!equal(base[u],target[u]));
    const parameter=edge?.source?.slice(4),structured=parameter==='ue_drx_configs';
    const required=new Map(changed.map(u=>[u,new Set(structured?
      Object.keys({...base[u],...target[u]}).filter(k=>!equal(base[u]?.[k],target[u]?.[k])):['*'])]));
    const covered=new Map(changed.map(u=>[u,new Set()])),used=[],notes=new Set();
    const result=excluded=>({excluded,calibrations:[...new Set(used)],
      changed_entities:changed,covered_components:Object.fromEntries([...covered].map(([u,fields])=>[u,[...fields]])),
      context_notes:[...notes],coefficients_used:false,physical_zero_assumed:false});
    if(!changed.length||!audit||audit.source!==edge.source||audit.target!==edge.target)return result(false);
    for(const entry of audit.excluded_estimates||[]){
      const row=entry.calibration;
      if(!entry.screened_effect_eligible||!['interval_includes_zero','observed_zero'].includes(entry.reason)||
        !row?.projection_units?.valid||row.parameter!==parameter||row.contrast_id!==entry.contrast_id)continue;
      const ids=members(row,frame).filter(u=>required.has(u));
      if(!ids.length)continue;
      const additions=[];
      if(row.kind==='finite_contrast'){
        if(row.from==null||row.to==null)continue;
        if(structured){
          if(typeof row.from!=='object'||typeof row.to!=='object')continue;
          const keys=Object.keys(row.from);
          if(!keys.length||!equal([...keys].sort(),Object.keys(row.to).sort()))continue;
          for(const u of ids){
            if(!base[u]||!target[u])continue;
            const forward=keys.every(k=>equal(base[u][k],row.from[k])&&equal(target[u][k],row.to[k]));
            const reverse=keys.every(k=>equal(base[u][k],row.to[k])&&equal(target[u][k],row.from[k]));
            if(forward||reverse)additions.push([u,keys.filter(k=>required.get(u).has(k))]);
          }
        }else for(const u of ids){
          if(equal(base[u],row.from)&&equal(target[u],row.to)||equal(base[u],row.to)&&equal(target[u],row.from))
            additions.push([u,['*']]);
        }
      }else if(['range_linear_fit','local_derivative'].includes(row.kind)){
        const component=structured?(row.scalar_component||'inactivity_ms'):null;
        const units=row.projection_units;
        const domain=units.fields?.fit_domain||units.fields?.sampled_domain||
          (finite(row.parameter_levels?.low)&&finite(row.parameter_levels?.high)&&finite(units.coordinate_factor)?
            [row.parameter_levels.low,row.parameter_levels.high].map(v=>v*units.coordinate_factor):null);
        if(!Array.isArray(domain)||domain.length!==2||!domain.every(finite)||domain[0]>domain[1])continue;
        for(const u of ids){
          const left=component?base[u]?.[component]:base[u],right=component?target[u]?.[component]:target[u];
          if(!finite(left)||!finite(right)||left===right||Math.min(left,right)<domain[0]||Math.max(left,right)>domain[1])continue;
          if(!component||required.get(u).has(component))additions.push([u,[component||'*']]);
        }
      }
      if(!additions.some(([,fields])=>fields.length))continue;
      additions.forEach(([u,fields])=>fields.forEach(k=>covered.get(u).add(k)));
      used.push(entry.contrast_id);
      // Keep the existing first-order population projection's context-shift
      // semantics. This is a disclosed omission, not a borrowed contextual gain.
      const expected=row.reference_scheduler_mode;
      if(expected&&Object.values(value(frame,'scheduler_mode')).some(mode=>mode!==expected))notes.add('reference_context_shift');
    }
    return result([...required].every(([u,fields])=>[...fields].every(k=>covered.get(u).has(k))));
  }
  function intentContext(row,frame){
    const current=frame.states?.['kpi:'+row.kpi]?.value;
    const anchors=(row.anchors||[]).map(a=>a.current).filter(finite);
    const baseline=SmoGraph.frozenIntentReference(row,frame);
    const bounds=[row.lower,row.upper].filter(finite);
    const floor=Math.max(finite(baseline)?Math.abs(baseline)*1e-6:0,1e-9);
    const distances=finite(baseline)?bounds.map(b=>Math.abs(b-baseline)):[];
    const anchoredAtBoundary=distances.some(d=>d<=floor);
    let scale,basis;
    if(distances.length&&!anchoredAtBoundary){scale=Math.max(Math.min(...distances),floor);basis='frozen_target_distance_or_tolerance';}
    else if(finite(baseline)&&Math.abs(baseline)>floor){scale=Math.abs(baseline);basis='frozen_baseline_magnitude';}
    else {scale=Math.max(...bounds.map(Math.abs),1);basis='bound_or_one_unit_fallback';}
    const empty=!!row.empty||(finite(row.lower)&&finite(row.upper)&&(row.lower>row.upper||row.lower===row.upper&&(row.lower_inclusive===false||row.upper_inclusive===false)));
    const margins=finite(current)?[finite(row.lower)?current-row.lower:null,finite(row.upper)?row.upper-current:null].filter(finite):[];
    const margin=margins.length?Math.min(...margins):null;
    const weight=margin===null?1:1/(1+Math.max(0,margin)/scale);
    return {kpi:row.kpi,baseline:finite(baseline)?baseline:null,current:finite(current)?current:null,
      lower:row.lower??null,upper:row.upper??null,lower_inclusive:row.lower_inclusive!==false,upper_inclusive:row.upper_inclusive!==false,
      scale,scale_basis:basis,margin,weight,weight_basis:'1/(1+nonnegative_margin/scale)',empty,
      context_missing:!finite(current),baseline_fallback:!anchors.length};
  }
  function aggregate(chains){
    const numeric=chains.filter(c=>finite(c.drive)),nonzero=numeric.some(c=>c.drive!==0);
    const incomplete=chains.some(c=>!finite(c.drive));
    // Retain a nonzero known projection, but never turn an entirely unknown
    // active effect into zero merely because another component is unchanged.
    const drive=incomplete&&!nonzero?null:numeric.reduce((n,c)=>n+c.drive,0);
    const directions=[...new Set(chains.filter(c=>c.direction!==null&&c.direction!==0).map(c=>c.direction))];
    const direction=drive!==null?sign(drive):directions.length===1?directions[0]:null;
    return {drive,direction,known:numeric.length,total:chains.length,
      partial:incomplete||chains.some(c=>c.notes?.some(n=>n!=='local_reference_drive_not_kpi_change'))};
  }
  function review(graph,frame,groups,ranges,preview,{showDashed=false,normalizationContexts=null}={}){
    if(preview?.slot!==frame?.slot)return {method,status:'unavailable',reason:'matching_decision_preview_unavailable',groups:[]};
    const uniqueRanges=[...new Map(ranges.map(r=>[r.kpi,r])).values()],keys=uniqueRanges.map(r=>r.kpi);
    if(!keys.length)return {method,status:'unavailable',reason:'intent_kpis_unavailable',groups:[]};
    const contexts=normalizationContexts||uniqueRanges.map(r=>intentContext(r,frame));
    if(contexts.some(c=>c.empty))return {method,status:'unavailable',reason:'empty_intent_interval',groups:[],intent_contexts:contexts};
    const base={},apps=new Map(),controls=graph.edges.filter(e=>e.kind==='controls'&&e.deployment!=='native_locked');
    for(const p of new Set(controls.map(e=>e.target.slice(4))))base[p]=structuredClone(value(frame,p));
    const observations=new Set((preview.apps||[]).map(a=>a.observation_slot).filter(Number.isInteger));
    if(observations.size>1)return {method,status:'unavailable',reason:'inconsistent_public_observation',groups:[]};
    for(const app of preview.apps||[]){
      const proposed=app.status==='proposed'&&app.actions?.length;
      const hold=app.status==='no_action'&&app.inference_invoked===true&&!app.actions?.length;
      let valid=!!(proposed||hold)&&!app.action_contract_failures?.length;
      const seen=new Set();
      for(const a of app.actions||[]){
        const p=a.parameter,id=a.ue_id||a.cell_id,key=p+'|'+id;
        if(!controls.some(e=>e.source===app.app_id&&e.target==='rcp:'+p)||seen.has(key)||a.current==null||a.proposed==null||!base[p]||base[p][id]!=null&&!equal(base[p][id],a.current))valid=false;
        seen.add(key);
        if(base[p]&&base[p][id]==null&&a.current!=null)base[p][id]=structuredClone(a.current);
      }
      apps.set(app.app_id,{...app,valid,hold});
    }
    const chains=[],ids=[...new Set(groups.flatMap(g=>g.app_ids))];
    for(const id of ids){
      const app=apps.get(id),target=structuredClone(base);
      if(app?.valid)for(const a of app.actions||[])target[a.parameter][a.ue_id||a.cell_id]=structuredClone(a.proposed);
      for(const p of new Set(controls.filter(e=>e.source===id).map(e=>e.target.slice(4))))for(const kpi of keys){
        const edge=graph.edges.find(e=>e.kind==='influences'&&e.source==='rcp:'+p&&e.target==='kpi:'+kpi);
        if(!edge)continue;
        if(!showDashed&&SmoGraph.influencePresentation(edge).dashed)continue;
        const estimate=app?.valid?project(edge,base[p],target[p],frame,showDashed,{target,actions:app.actions||[]}):{drive:null,direction:null,status:'proposal_unavailable',notes:['proposal_unavailable'],calibrations:[]};
        const range=uniqueRanges.find(r=>r.kpi===kpi),context=contexts.find(r=>r.kpi===kpi);
        chains.push({app_id:id,parameter:p,kpi,influence_edge_id:edge.id,current:structuredClone(base[p]),proposed:app?.valid?structuredClone(target[p]):null,
          ...estimate,normalized_drive:finite(estimate.drive)?estimate.drive/context.scale:null,
          intent_role:intentRole(estimate.direction,range,context.current),intent_range:structuredClone(range)});
      }
    }
    const result=[];
    for(const group of groups){
      const selected=chains.filter(c=>group.app_ids.includes(c.app_id)),vectors=[];
      for(const id of group.app_ids)for(const context of contexts){
        const own=selected.filter(c=>c.app_id===id&&c.kpi===context.kpi);
        const unmeasuredPath=graph.edges.some(e=>e.kind==='influences'&&e.target==='kpi:'+context.kpi&&!own.some(c=>c.influence_edge_id===e.id)&&controls.some(c=>c.source===id&&c.target===e.source)&&apps.get(id)?.actions?.some(a=>'rcp:'+a.parameter===e.source&&!equal(a.current,a.proposed)));
        const contributions=unmeasuredPath?[...own,{drive:null,direction:null,notes:['reference_direction_unavailable']}]:own;
        const a=!apps.get(id)?.valid?{drive:null,direction:null,partial:true}:contributions.length?aggregate(contributions):{drive:0,direction:0,partial:false};
        vectors.push({app_id:id,kpi:context.kpi,...a,normalized_drive:finite(a.drive)?a.drive/context.scale:null,
          intent_role:intentRole(a.direction,uniqueRanges.find(r=>r.kpi===context.kpi),context.current),
          chain_parameters:own.map(c=>c.parameter),unmeasured_path:unmeasuredPath,no_visible_graph_path:!own.length&&!unmeasuredPath,proposal_mode:apps.get(id)?.mode||null});
      }
      const interactions=[];let numerator=0,denominator=0;
      for(const [a,b]of pairs(group.app_ids))for(const context of contexts){
        const left=vectors.find(v=>v.app_id===a&&v.kpi===context.kpi),right=vectors.find(v=>v.app_id===b&&v.kpi===context.kpi);
        // No relevant path on either side cannot form a shared KPI interaction.
        if(left.no_visible_graph_path||right.no_visible_graph_path)continue;
        const x=left.normalized_drive,y=right.normalized_drive;
        const known=finite(x)&&finite(y),strength=known?Math.abs(x)+Math.abs(y):null;
        const opposing=left.direction!==null&&right.direction!==null?left.direction*right.direction<0:null;
        const cancellation=known?Math.max(0,strength-Math.abs(x+y)):null;
        const weightedStrength=known?context.weight*strength:null,weightedCancellation=known?context.weight*cancellation:null;
        if(known){numerator+=weightedCancellation;denominator+=weightedStrength;}
        interactions.push({apps:[a,b],kpi:context.kpi,left,right,opposing,cancellation,
          strength,weight:context.weight,weighted_cancellation:weightedCancellation,weighted_strength:weightedStrength,
          partial:left.partial||right.partial,status:known?'reference_magnitudes':opposing===null?'unknown':'direction_only'});
      }
      const known=selected.filter(c=>finite(c.drive)),unknownInteractions=interactions.filter(p=>p.cancellation===null);
      const missing=group.app_ids.filter(id=>!apps.get(id)?.valid);
      const incomplete=missing.length>0||selected.some(c=>!finite(c.drive))||vectors.some(v=>v.unmeasured_path)||unknownInteractions.length>0||contexts.some(c=>c.context_missing);
      const approximate=selected.some(c=>c.notes.some(n=>n!=='local_reference_drive_not_kpi_change'));
      const partialEvidence=selected.some(c=>c.notes.some(n=>['uncertain_gain_direction','uncertain_joint_profile_filtered','uncovered_structured_components','uncovered_population_members','unmeasured_sleep_profile'].includes(n)));
      const noGainEvidence=!selected.length&&group.app_ids.some(id=>apps.get(id)?.actions?.some(a=>!equal(a.current,a.proposed)));
      const unresolved=incomplete||noGainEvidence;
      const score=denominator>0?100*Math.min(1,Math.max(0,numerator/denominator)):
        unresolved?null:0;
      const summaries=group.conflict_summaries||group.conflicts;
      const contributors=interactions.filter(p=>p.weighted_cancellation>0).map(p=>({apps:p.apps,kpi:p.kpi,
        weighted_cancellation:p.weighted_cancellation,score_contribution:denominator>0?100*p.weighted_cancellation/denominator:0})).sort((a,b)=>b.score_contribution-a.score_contribution);
      result.push({group_id:group.group_id||group.id,app_ids:group.app_ids,score,score_range:[0,100],
        score_aggregation:'intent_weighted_control_drive_cancellation',score_status:unresolved||partialEvidence?'partial_evidence':approximate?'reference_approximation':'reference_score',
        score_numerator:numerator,score_denominator:denominator,chains:selected,app_vectors:vectors,interactions,contributors,
        structural_counts:summaries?conflictCounts(summaries):null,count_basis:summaries?'catalog_topology_only_not_score':null,
        coverage:{scope:'visible_intent_control_chains',known_chains:known.length,total_chains:selected.length,
          comparable_pairs:interactions.length-unknownInteractions.length,total_pairs:interactions.length,
          partial_chains:selected.filter(c=>c.notes.some(n=>n!=='local_reference_drive_not_kpi_change')).length,
          missing_proposal_apps:missing},
        intent_direction:{against:vectors.filter(v=>v.intent_role==='against_interval').length,
          consuming_margin:vectors.filter(v=>v.intent_role==='consuming_margin').length,
          toward:vectors.filter(v=>v.intent_role==='toward_interval').length,
          unknown:vectors.filter(v=>v.intent_role==='context_unavailable').length}});
    }
    return {method,status:'ready',snapshot:{slot:frame.slot,time_ms:frame.time_ms},intent_contexts:contexts,
      source:'graph_calibrations_and_recorded_shadow_proposals',evidence_filter:showDashed?'including_uncalibrated_paths':'measured_weights_and_declared_paths',
      score_semantics:'reference_control_opposition_not_attainment_or_probability',
      kpi_forecast_performed:false,observed_after_kpis_used:false,selection_performed:false,groups:result};
  }
  return {method,review,project,screenedProjection,intentRole,intentContext,conflictCounts};

})();

if(typeof document!=='undefined') (() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const state = { bundle:null, index:0, all:false, selected:null, focus:'', playing:false, timer:null, polling:null, contextViews:{}, intent:null, activeIntents:[], constraintIssues:[], invalidatedKpis:[], graphGroup:null, intentVersion:0, intentBusy:false, intentFrame:null, combinationCatalog:null, conflictView:null, showDashed:false, presetEdits:{}, deletedPresetIds:[], intentPresets:SmoGraph.customIntentCases(), editingPreset:null };
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (n, digits=2) => typeof n === 'number' && Number.isFinite(n) ? n.toLocaleString('en-US', {maximumFractionDigits:digits}) : '—';
  const metricValue=(key,value,digits=4)=>fmt(SmoGraph.metricDisplayValue(key,value),digits);
  const metricUnit=(key,unit)=>SmoGraph.metricDisplayUnit(key,unit);
  const time = slot => typeof slot === 'number' ? `${fmt(slot * (state.bundle.source.slot_duration_ms || .5),1)} ms` : '—';
  const frame = () => state.bundle.source.frames[state.index] || {states:{}};
  const nodeState = n => frame().states[n.id] || {};
  const graph = () => state.bundle.graph;
  const population = () => graph().metric_scope === 'ue_population.v1';
  const fixedBeforeSlots = () => {
    const slots=state.bundle.runtime_environment?.origin_observe_slots;
    return Number.isInteger(slots)&&slots>0?slots:null;
  };
  const statisticsLabel = () => state.bundle.runtime_environment?.statistics?.label||'';
  const windowDuration = slots => fmt(slots*(state.bundle.source.slot_duration_ms||.5)/1000,3)+' s';
  const evaluationWindows = (slots,beforeSlots=fixedBeforeSlots()) => beforeSlots==null
    ? `前后各 ${slots} slots（${windowDuration(slots)}）`
    : `部署前 ${beforeSlots} slots（${windowDuration(beforeSlots)}） / 部署后 ${slots} slots（${windowDuration(slots)}）`;
  const find = id => graph().nodes.find(n => n.id === id) || graph().edges.find(e => e.id === id);
  const section = (title, html) => `<section class="inspector-section"><h3>${esc(title)}</h3>${html}</section>`;
  const disclosure = (title, html) => `<details class="inspector-section"><summary>${esc(title)}</summary>${html}</details>`;
  const line = (label, value) => `<div class="state-line"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
  const pretty = v => typeof v === 'object' ? JSON.stringify(v) : String(v);
  const effects = e => e.sensitivities?.length ? e.sensitivities : e.sensitivity ? [e.sensitivity] : [];
  const effectNumber = v => typeof v !== 'number' ? 'per UE' : v !== 0 && (Math.abs(v) < .001 || Math.abs(v) >= 1e5) ? v.toExponential(2) : fmt(v,4);
  const signedEffect = v => typeof v === 'number' && v > 0 ? '+'+effectNumber(v) : effectNumber(v);
  const magnitudeText = estimates => {
    const m=SmoGraph.influenceMagnitude(estimates);
    return !m?'—':m.lower===m.upper?effectNumber(m.lower):effectNumber(m.lower)+'–'+effectNumber(m.upper);
  };
  const magnitudeNote = estimates => SmoGraph.influenceMagnitude(estimates)?.basis==='normalized_importance'
    ? 'Absolute normalized influence over the tested control span; a range covers multiple measured changes.'
    : 'Absolute measured gain; a range covers multiple changes with the same unit.';
  const contrastLabel = s => s.contrast_label || ({weight_mec_population_range:'MEC priority · Latency-aware',weight_urllc_population_range:'URLLC priority · QoS-deadline',scheduler_latency:'RR → Latency-aware',scheduler_qos:'RR → QoS-deadline',ap_sleep:'Active → Micro-sleep',offload_population_range:'MEC offloading',drx_population_range:'DRX inactivity',drx_cycle_population_range:'DRX cycle',drx_profile_80ms_1ms:'DRX profile 40/8 → 80/1 ms',ap_micro_sleep:'Active → Micro-sleep',ap_light_sleep:'Active → Light-sleep',ap_deep_sleep:'Active → Deep-sleep',throughput_priority_component_range:'Throughput priority · Latency-aware'}[s.contrast_id] || (s.contrast_id || s.scope).replaceAll('_',' '));
  const resolved = SmoGraph.resolvedEstimate;
  const zeroEffect = s => s?.kind === 'range_linear_fit' ? s.fit_diagnostics?.assessment === 'constant_response_observed' : s?.zero_in_both === true || s?.estimate === null && Object.keys(s.entity_estimates || {}).length > 0 && Object.values(s.entity_estimates).every(v => v === 0);
  const invalidatedKpi = key => state.invalidatedKpis.includes(key);
  function kpiGoal(node) {
    if(invalidatedKpi(node.key))return null;
    const request=state.intent?.ranges.find(r=>r.kpi===node.key);
    if(request)return request.needed_direction==='increase'?{text:'↑ Higher',title:'Intent target: increase this KPI'}:request.needed_direction==='decrease'?{text:'↓ Lower',title:'Intent target: decrease this KPI'}:{text:'↔ Range',title:'Intent target: stay within the requested interval'};
    return {minimize:{text:'↓ Lower',title:'Reference objective: lower values'},maximize:{text:'↑ Higher',title:'Reference objective: higher values'},upper_bound:{text:'≤ Limit',title:'Reference objective: respect a compute-demand limit'}}[node.preferred_direction]||null;
  }

  function edgePresentation(edge) {
    const selectedContext=state.intent?null:state.contextViews[edge.source]||null;
    return SmoGraph.influencePresentation(edge,{contrastIds:selectedContext?[selectedContext]:null});
  }
  function contextSelect(parameter) {
    if(state.intent)return '';
    const choices=new Map(graph().edges.filter(e=>e.source===parameter&&e.kind==='influences').flatMap(e=>effects(e)).map(s=>[s.contrast_id,s]));
    if(choices.size<=1)return '';
    const active=state.contextViews[parameter]||'';
    return `<label class="sr-only" for="estimate-context">Measured change</label><select id="estimate-context" class="estimate-select" data-parameter="${esc(parameter)}"><option value="" ${!active?'selected':''}>All measured changes</option>${[...choices].map(([id,s])=>`<option value="${esc(id)}" ${id===active?'selected':''}>${esc(contrastLabel(s))}</option>`).join('')}</select>`;
  }
  function contextTable(estimates) {
    return `<table class="effect-context-table"><thead><tr><th>Change</th><th>ΔKPI / change</th><th>Confidence interval</th></tr></thead><tbody>${estimates.map(s=>`<tr><td>${esc(contrastLabel(s))}<small>${esc(s.unit)}</small></td><td>${esc(signedEffect(s.estimate))}<small>${resolved(s)?'Resolved':zeroEffect(s)?'No observed change':'Uncertain'}</small></td><td>${esc((s.confidence_interval||[]).map(effectNumber).join(' … ')||'—')}<small>${Number.isFinite(s.confidence_level)?esc(Math.round(s.confidence_level*100)+'%'):''}</small></td></tr>`).join('')}</tbody></table>`;
  }
  function magnitudeTable(estimates) {
    return `<table class="effect-context-table"><thead><tr><th>Change</th><th>Impact</th></tr></thead><tbody>${estimates.map(s=>`<tr><td>${esc(contrastLabel(s))}</td><td title="${esc(magnitudeNote([s]))}">${esc(magnitudeText([s]))}<small>${resolved(s)?'Resolved':zeroEffect(s)?'No observed change':'Uncertain'}</small></td></tr>`).join('')}</tbody></table>`;
  }
  const rangeText = r => `${r.lower_inclusive?'[':'('}${r.lower==null?'−∞':metricValue(r.kpi,r.lower)}, ${r.upper==null?'+∞':metricValue(r.kpi,r.upper)}${r.upper_inclusive&&r.upper!=null?']':')'} ${metricUnit(r.kpi,r.unit)}`;
  const intentErrorTranslations = {
    'The language model must return an object containing ranges and questions.':'语言模型返回的格式有误，需要包含 KPI 区间和澄清问题。',
    'Expected up to eight KPI ranges.':'每次最多支持八个 KPI 区间。',
    'Invalid clarification questions.':'语言模型返回的澄清问题格式有误。',
    'No KPI range was understood. Name a KPI and the range you want.':'未识别到 KPI 要求，请说明目标指标及期望方向或范围。',
    'The model returned an unsupported KPI or interval field.':'语言模型返回了不支持的 KPI 或区间字段。',
    'Interval endpoints must declare whether they are included.':'KPI 区间需要明确是否包含边界值。',
    'Unsupported interval endpoint.':'不支持该 KPI 区间的边界格式。',
    'Enter a valid DeepSeek API key.':'请输入有效的 DeepSeek API 密钥。',
    'Invalid model name.':'模型名称无效。',
    'Describe your KPI intent in 1–2,000 characters.':'请输入一至两千字以内的 KPI 意图。',
    'Describe a KPI intent in 1–2,000 characters.':'请输入一至两千字以内的 KPI 意图。',
    'Connect DeepSeek or set DEEPSEEK_API_KEY in the service environment.':'请先连接 DeepSeek。',
    'An intent is already being interpreted. Wait for it to finish.':'已有意图正在解析，请等待完成。',
    'DeepSeek interpretation failed. Check the API key, model and connection; no substitute interpretation was used.':'DeepSeek 解析失败，请检查 API 密钥、模型名称和连接。',
    'Invalid current-value multiplier.':'当前值的相对变化倍数无效。',
    'Requested relative target is out of range.':'请求的相对目标超出了支持范围。',
    'Select up to 32 distinct saved intents.':'最多保留三十二条不同的有效意图。',
    'Active intents support up to eight distinct KPIs.':'有效意图合计最多支持八种 KPI。',
    'A saved intent is unavailable in this network context. Clear it or restore its ledger.':'保存的意图在当前网络上下文中不可用，请清空意图或恢复对应记录。',
    'DeepSeek returned an invalid interval twice. Please make the KPI and range explicit.':'DeepSeek 连续两次返回无效区间，请明确 KPI 及其目标范围。',
    'The intent broker snapshot context is not verified.':'尚未验证语言服务的数据上下文，请刷新后重试。',
    'The local DeepSeek session could not interpret this request. Check its connection or KPI bounds.':'DeepSeek 未能解析该请求，请检查连接或 KPI 目标范围。',
    'The local DeepSeek session is unavailable.':'DeepSeek 会话当前不可用，请检查模型连接。',
    'The local intent broker returned an oversized response.':'语言服务返回的数据过大，请简化意图后重试。',
    'Intent matching requires the population graph view.':'意图匹配需要使用群体统计视图。',
    'Select a valid simulation snapshot.':'请选择有效的仿真快照。',
    'This simulation snapshot is no longer available. Refresh and try again.':'该仿真快照已不可用，请刷新后重试。',
    'Do you mean URLLC packet latency or offloaded MEC task latency?':'你希望降低 URLLC 数据包时延，还是 MEC 卸载任务时延？',
  };
  const chineseKpis={ap_power_w:'基站总功率',ue_power_w:'UE 总平均功率',network_goodput_mbps:'全网下行吞吐量',ue_energy_j:'UE 累计能量（历史口径）',mec_expected_gflops:'MEC 总预期计算需求',mec_offloaded_latency_ms:'MEC 卸载任务平均端到端时延',task_latency_ms:'全部 MEC 任务平均端到端时延',urllc_failure_events_per_s:'URLLC 超时与丢弃事件率',urllc_failure_fraction:'URLLC 数据包错误率',urllc_packet_latency_ms:'URLLC 数据包平均时延'};
  function intentMessage(text,problem=false) {
    let translated=intentErrorTranslations[text]||text;
    const unavailable=text.match(/^Current (.+) is unavailable/);
    if(unavailable){const key=state.bundle?.graph.nodes.find(n=>n.kind==='kpi'&&n.label===unavailable[1])?.key;translated='当前快照缺少'+(chineseKpis[key]||unavailable[1])+'数据，请选择包含该指标记录的数据源后再使用相对目标。';}
    if(!/[\u4e00-\u9fff]/.test(text)&&/contradictory|strictly above.*strictly below/i.test(text))translated='意图对同一 KPI 的要求互相矛盾，请明确期望提高还是降低。';
    $('intent-message').textContent=translated;$('intent-message').classList.toggle('problem',problem);$('intent-feedback').hidden=!text||(!problem&&/^(已更新观测|本次意图已解析并保存)/.test(translated));
  }
  function renderIntentCaseNote() {
    const preset=state.intentPresets.find(c=>c.id===$('intent-case').value);
    const note=$('intent-case-note');note.hidden=!preset;
    if(!preset){note.textContent='';return;}
    const states=state.bundle?.source.frames[state.index]?.states||{};
    const missing=state.bundle?preset.kpis.filter(k=>states['kpi:'+k]?.value==null).map(k=>chineseKpis[k]||find('kpi:'+k)?.label||k):[];
    const floor=preset.id==='c06'&&states['kpi:urllc_failure_events_per_s']?.value===0?'当前快照中的 URLLC 失败事件率已为零。':'';
    note.textContent=[preset.needsThreshold?preset.note:'',missing.length?'当前数据缺少：'+missing.join('、')+'。':'',floor].filter(Boolean).join(' ');note.hidden=!note.textContent;
  }
  function clearIntentReview() {
    const hadReview=!!(state.intent||state.intentBusy);
    state.intentVersion++;state.intentBusy=false;state.intent=null;state.graphGroup=null;
    state.conflictView=null;state.intentFrame=null;state.combinationCatalog=null;
    $('intent-submit').textContent='分析';intentMessage('');
    if(hadReview){state.selected=null;state.focus='';$('app-focus').value='';render();}
    else renderIntentCaseNote();
  }
  const intentContextId=()=>state.bundle.intent_context_id||state.bundle.source.name;
  const intentStorageKey=()=> 'oran.smo.active-intents:'+intentContextId();
  function saveIntentIds(ids) { try{localStorage.setItem(intentStorageKey(),JSON.stringify(ids));}catch{} }
  function migrateIntentIds() {
    const migration=state.bundle.intent_migration;
    if(!migration?.token||typeof migration.from_source_name!=='string'||!Array.isArray(migration.record_ids))return;
    const marker='oran.smo.intent-migration.v1:'+migration.token;
    try{
      if(localStorage.getItem(marker)!==null)return;
      if(localStorage.getItem(intentStorageKey())===null){
        const previous=JSON.parse(localStorage.getItem('oran.smo.active-intents:'+migration.from_source_name)||'[]');
        if(Array.isArray(previous)){
          const available=new Set(migration.record_ids);
          localStorage.setItem(intentStorageKey(),JSON.stringify([...new Set(previous.filter(id=>available.has(id)))]));
        }
      }
      localStorage.setItem(marker,'complete');
    }catch{}
  }
  function applyIntentResult(result) {
    state.intent=result;state.intentRestoreFailed=false;state.activeIntents=result.active_intents||[];
    state.constraintIssues=result.constraint_issues||[];state.invalidatedKpis=result.invalidated_kpis||[];
    saveIntentIds(state.activeIntents.map(r=>r.id));
    state.combinationCatalog=null;state.graphGroup=null;state.conflictView=null;
    state.selected=null;state.focus='';$('app-focus').value='';
  }
  function clearAllIntents() {
    state.activeIntents=[];state.constraintIssues=[];state.invalidatedKpis=[];state.intentRestoreFailed=false;saveIntentIds([]);clearIntentReview();render();
  }
  async function reviewSavedIntents(ids) {
    if(!ids.length){clearAllIntents();return;}
    stop();const version=++state.intentVersion;
    state.intentBusy=true;state.intent=null;state.combinationCatalog=null;state.intentFrame=structuredClone(frame());
    intentMessage('正在计算共同允许区间…');render();
    try{
      const result=await postIntent('./api/intent/review',{active_intent_ids:ids,slot:state.intentFrame.slot??null});
      if(version!==state.intentVersion)return;
      applyIntentResult(result);
      const feedback=SmoGraph.intentFeedback(result);intentMessage(feedback.text,feedback.problem);
    }catch(e){if(version===state.intentVersion){state.intentRestoreFailed=true;intentMessage(e.message,true);}}
    finally{if(version===state.intentVersion){state.intentBusy=false;render();}}
  }
  function intentRangeTable(ranges, {baseline=false}={}) {
    return `<table class="intent-kpi-table"><thead><tr><th scope="col">KPI</th><th scope="col">期望范围</th><th scope="col">${baseline?'提出时基准':'当前观测'}</th></tr></thead><tbody>${ranges.map(r=>`<tr><th scope="row">${esc(chineseKpis[r.kpi]||r.label)}</th><td><code>${esc(r.empty?'∅':rangeText(r))}</code>${invalidatedKpi(r.kpi)?'<small>旧能耗意图需重新解析为功率</small>':''}</td><td><code>${esc(metricValue(r.kpi,r.current)+' '+metricUnit(r.kpi,r.unit))}</code>${!baseline?`<small>${esc(invalidatedKpi(r.kpi)?'保留原始 J 口径，未换算为 W':r.empty?'区间交集为空':r.current==null?'数据不可用':r.status==='within_range'?'位于共同区间内':'尚未进入共同区间')}</small>`:''}</td></tr>`).join('')}</tbody></table>`;
  }
  function renderIntentHistory() {
    $('intent-history').hidden=!state.activeIntents.length;
    $('intent-active-count').textContent=state.activeIntents.length+' 项';
    $('intent-refresh').disabled=state.intentBusy;
    $('intent-records').innerHTML=state.activeIntents.map(record=>{
      const issues=SmoGraph.constraintIssuesForRecord(state.constraintIssues,record.id);
      return `<li class="${issues.length?'intent-record-needs-attention':''}"><div class="intent-record-heading"><div class="intent-record-title"><div class="intent-record-status"><span class="intent-status-active">约束生效中</span>${issues.some(issue=>issue.blocking)?'<span class="intent-status-attention">待处理</span>':''}</div><strong>${esc(record.text)}</strong></div><button class="text-button" type="button" data-remove-intent="${esc(record.id)}" aria-label="${esc('停用意图：'+record.text)}" title="停用后保留原始记录及基准" ${state.intentBusy?'disabled':''}>停用</button></div>${issues.map(issue=>`<p class="intent-record-warning">${esc(issue.message)}</p>`).join('')}${intentRangeTable(record.ranges,{baseline:true})}<small class="intent-anchor-time">基准快照 ${esc(fmt((record.snapshot.time_ms||0)/1000,3)+' s')}</small></li>`;
    }).join('');
    for(const button of $('intent-records').querySelectorAll('[data-remove-intent]'))button.addEventListener('click',()=>reviewSavedIntents(state.activeIntents.filter(r=>r.id!==button.dataset.removeIntent).map(r=>r.id)));
  }
  function intentStatus() {
    const s=state.bundle?.intent_service || {},offline=location.protocol==='file:';
    const initial=state.bundle?.candidate_selection;
    const selectionActive=['influence_coverage_v1','semantic_influence_coverage_v2'].includes(initial?.method);
    $('initial-selection-method').hidden=!selectionActive;
    $('initial-selection-method').textContent=selectionActive?`组合初选：${initial.method==='semantic_influence_coverage_v2'?'能力方向＋统计影响':'统计影响覆盖'} · ${Number((initial.coverage_threshold*100).toFixed(2))}% 门槛`:'';
    $('llm-status').textContent=offline?'需要在线语言服务':`DeepSeek · ${s.verified?'已连接':s.configured?'密钥已配置':'请配置 API 密钥'}`;
    $('llm-settings').hidden=offline;
    $('intent-submit').disabled=offline || state.intentBusy || !!state.editingPreset;
    if (document.activeElement!==$('llm-model')) $('llm-model').value=s.model || 'deepseek-v4-pro';
    if(offline){
      $('intent-message').replaceChildren();$('intent-feedback').hidden=false;
      const a=document.createElement('a');a.href=s.url || 'http://127.0.0.1:53505/';a.textContent='打开在线页面进行意图分析';$('intent-message').append(a);
    }
  }
  const viewedGroup = () => state.combinationCatalog?.groups.find(g=>g.id===state.graphGroup);
  const intentCandidates = () => state.combinationCatalog?.candidates||state.intent?.candidates||[];
  const groupConflicts = group => group?.conflicts||group?.conflict_summaries||[];
  function displayedConflicts() {
    const groups=viewedGroup()?[viewedGroup()]:state.combinationCatalog?.groups||[];
    const rows=groups.flatMap(groupConflicts);
    return [...new Map(rows.map(c=>[JSON.stringify([c.type,c.apps,c.parameters,c.kpis]),c])).values()];
  }
  function renderConflictLegend(conflicts) {
    $('conflict-legend').hidden=false;
    for(const type of ['direct','indirect','implicit']) {
      const key=$(type+'-conflict-legend'),present=conflicts.some(c=>c.type===type);
      key.hidden=false;
      key.classList.toggle('muted',!present);
      key.dataset.hasConflicts=String(present);
      key.title=present?'当前视图包含此类冲突风险':'当前视图没有此类冲突风险；仅展示配色';
    }
  }
  function renderIntent() {
    renderIntentHistory();renderIntentCaseNote();intentStatus();syncPresetControls();
    $('intent-results').hidden=!state.intent;
    $('intent-clear').hidden=!state.intent&&!state.intentBusy&&!state.activeIntents.length&&!state.intentRestoreFailed;
    $('combination-catalog').hidden=!state.intent;
    $('intent-legend').hidden=!state.intent?.candidates?.length;
    if(!state.intent){renderConflictLegend([]);return;}
    state.combinationCatalog=state.showDashed?state.intent.combination_catalog_with_uncertain:state.intent.combination_catalog;
    $('intent-legend').hidden=!intentCandidates().length;
    if(state.graphGroup&&!viewedGroup())state.graphGroup=null;
    $('intent-path-view').textContent=viewedGroup()?'Viewing one combination':'All candidate paths';
    state.conflictView={analysis_basis:'knowledge_graph',conflicts:displayedConflicts(),runtime_proposals_used:false};
    renderConflictLegend(state.conflictView.conflicts);
    const stale=state.intent.snapshot.slot!==frame().slot;
    $('intent-snapshot').hidden=!stale;
    $('intent-snapshot').textContent=stale?'观测已变化；点击“更新观测”重新评估，旧基准保持不变。':'';
    $('intent-ranges').innerHTML=intentRangeTable(state.intent.ranges);
    const blocked=SmoGraph.hasBlockingConstraints(state.intent);
    $('intent-constraints').hidden=!blocked;
    $('intent-constraints').textContent=blocked?'联合分析已暂停。请处理上方标注的意图；原始记录与基准保持不变。':'';
    $('intent-candidates').innerHTML=intentCandidates().map(c=>`<span class="intent-candidate">${esc(c.label)}</span>`).join('');
    renderCombinationCatalog();
  }
  let conflictQuantification=null,conflictQuantificationRequestKey=null,conflictQuantificationBusy=false,conflictQuantificationVersion=0,conflictQuantificationTimer=null;
  let quantificationPanelView=null;
  const uncalibratedEnvironment=()=>{
    const statistics=state.bundle?.runtime_environment?.statistics;
    return statistics?.kind==='uncalibrated'||statistics?.current_calibration===false;
  };
  const quantificationMethod=()=>$('conflict-method')?.value||'goal_obstruction_v2';
  function quantificationKey(){
    if(!state.intent||state.intentBusy||state.intent.snapshot?.slot!==frame().slot)return null;
    return JSON.stringify([SmoGraph.quantificationContextKey(state.bundle,frame()),
      quantificationMethod(),state.activeIntents.map(r=>r.id).sort(),state.intent.ranges,state.showDashed,
      'all_covered',SmoGraph.quantificationGroups(state.intent,state.combinationCatalog,state.showDashed).map(g=>g.id),
      'current_decision']);
  }
  const currentQuantification=()=>quantificationKey()===conflictQuantificationRequestKey?conflictQuantification:null;
  function decisionPacket(response,expectedFrame){
    if(!response.decision_frame&&!response.decision_preview)return null;
    const f=response.decision_frame,p=response.decision_preview;
    const stable=v=>v&&typeof v==='object'?Array.isArray(v)?v.map(stable):Object.fromEntries(Object.keys(v).sort().map(k=>[k,stable(v[k])])):v;
    if(!f||!p||f.slot!==response.preview_slot||p.slot!==f.slot||response.snapshot?.slot!==f.slot)
      throw Error('量化快照包不完整，未执行评分。');
    if(!expectedFrame||JSON.stringify(stable(expectedFrame))!==JSON.stringify(stable(f)))
      throw Error('网络快照已更新，请刷新网络快照并更新观测后重新量化。');
    return {frame:f,preview:p};
  }
  async function requestConflictQuantification(request,frames,post,trustedInputs=false){
    const endpoint=trustedInputs?'./api/conflicts/inputs':'./api/conflicts/quantify';
    const result=await post(endpoint,request);
    // A replay telemetry frame need not be a RIC decision boundary. Reuse
    // only an earlier, exact recorded decision and its own parameter/KPI frame;
    // never interpolate actions or borrow post-deployment evaluation results.
    const slot=result.suggested_slot,aligned=frames.find(f=>f.slot===slot);
    if(result.reason!=='matching_decision_preview_unavailable'||!Number.isInteger(slot)||slot>request.slot||!aligned)return result;
    const previous=await post(endpoint,{...request,slot});
    if(previous.source_id!==result.source_id||previous.preview_slot!==slot||previous.snapshot?.slot!==slot||!previous.groups?.length)return result;
    const requested=frames.find(f=>f.slot===request.slot);
    return {...previous,requested_snapshot:{slot:request.slot,time_ms:requested?.time_ms??null},
      snapshot_alignment:{policy:'nearest_earlier_recorded_decision',requested_slot:request.slot,decision_slot:slot,
        lag_slots:request.slot-slot,lag_ms:requested?.time_ms==null?null:requested.time_ms-aligned.time_ms,
        interpolation_performed:false}};
  }
  function conflictQuantificationEngine(method){
    if(method==='net_intent_obstruction_v1')return SmoIntentObstruction;
    return method.startsWith('goal_obstruction_')?SmoGoalObstruction:SmoConflictRisk;
  }
  function ensureConflictQuantification(force=false){
    const key=quantificationKey();
    if(!key||location.protocol==='file:'||(!force&&key===conflictQuantificationRequestKey))return;
    const structuralGroups=SmoGraph.quantificationGroups(state.intent,state.combinationCatalog,state.showDashed);
    if(!structuralGroups.length)return;
    conflictQuantificationRequestKey=key;conflictQuantification=null;conflictQuantificationBusy=true;
    const version=++conflictQuantificationVersion;clearTimeout(conflictQuantificationTimer);
    const request={active_intent_ids:state.activeIntents.map(r=>r.id),slot:frame().slot,
      observe_slots:500,show_dashed:state.showDashed,group_scope:'all_covered'};
    const bundle=state.bundle,frames=bundle.source.frames,method=quantificationMethod();
    const expectedSource=bundle.source.run_id||bundle.source.source_sha256||bundle.source.name;
    conflictQuantificationTimer=setTimeout(async()=>{
      try{
        const response=await requestConflictQuantification(request,frames,postIntent,!!bundle.intent_service?.conflict_input_schema);
        if(response.source_id!==expectedSource)throw Error('网络来源已变化，请刷新网络快照后重新量化。');
        if(bundle.graph_sha256&&response.graph_sha256!==bundle.graph_sha256)throw Error('图谱增益版本已变化，请刷新图谱后重新量化。');
        let result=response;
        if(response.groups?.length){
          const expectedFrame=frames.find(f=>f.slot===response.preview_slot),packet=decisionPacket(response,expectedFrame);
          const decisionFrame=packet?.frame||expectedFrame;
          const shadow=packet?.preview||await postIntent('./api/xapps/preview',{slot:response.preview_slot});
          if(shadow.source_id!==response.source_id)throw Error('拟动作与量化快照来自不同网络，请刷新网络快照后重新量化。');
          if(bundle.graph_sha256&&shadow.graph_sha256!==bundle.graph_sha256)throw Error('拟动作与量化使用的图谱版本不一致，请刷新图谱。');
          if(response.groups.some(g=>!structuralGroups.some(s=>s.id===g.group_id)))throw Error('量化组合范围已变化，请更新观测。');
          const riskGroups=response.groups.map(g=>({...g,conflict_summaries:g.conflict_summaries||structuralGroups.find(s=>s.id===g.group_id)?.conflict_summaries||[]}));
          const engine=conflictQuantificationEngine(method);
          const risk=engine.review(bundle.graph,decisionFrame,riskGroups,
            response.groups[0].kpis.map(k=>k.range),shadow,{method,showDashed:request.show_dashed,arbitration:response.arbitration_reference,normalization:response.influence_normalization});
          result={...risk,schema:risk.schema||'oran.intent-control-dynamics.v4',preview_slot:response.preview_slot,
            source_id:response.source_id,active_intent_ids:response.active_intent_ids,input_sha256:response.input_sha256,
            requested_snapshot:response.requested_snapshot,snapshot_alignment:response.snapshot_alignment,
            deployment_results_used:false,deployment_batch_used:false,
            reference_response_diagnostics_method:response.method,
            arbitration_reference:response.arbitration_reference,
            influence_normalization:response.influence_normalization,
            group_scope:response.group_scope,quantification_scope:response.quantification_scope,
            report_slots:response.report_slots,show_dashed:request.show_dashed};
          if(packet)Object.assign(result,{decision_frame:packet.frame,decision_preview:packet.preview,
            decision_frame_sha256:response.decision_frame_sha256,decision_preview_sha256:response.decision_preview_sha256,
            snapshot_packet_schema:response.snapshot_packet_schema,snapshot_packet_sha256:response.snapshot_packet_sha256});
        }
        if(version===conflictQuantificationVersion&&key===quantificationKey())conflictQuantification=result;
      }catch(error){if(version===conflictQuantificationVersion)conflictQuantification={status:'unavailable',error:error.message,groups:[]};}
      finally{if(version===conflictQuantificationVersion){conflictQuantificationBusy=false;render();}}
    },150);
  }
  const quantificationReasons={intent_kpis_unavailable:'尚无意图目标 KPI',kpi_scale_unavailable:'目标 KPI 缺少公共参考尺度；非比例 KPI 的零基准需指定尺度',kpi_observation_unavailable:'意图目标 KPI 缺少当前观测',empty_intent_interval:'意图区间为空，请处理共同约束',population_scope_mismatch:'动作范围与群体校准不符',nonuniform_population_action:'群体参数并非共同变化',
    environment_calibration_pending:'新环境增益待标定',
    network_snapshot_changed_during_capture:'网络快照在读取时已更新，请刷新后重试',
    unmeasured_structured_change:'该配置变化尚未测量',unmeasured_categorical_transition:'该策略切换尚未测量',outside_measured_domain:'超出已测参数范围',
    uncertain_gain_hidden:'当前增益方向不确定，虚线已隐藏',missing_gain_uncertainty:'缺少增益区间',gain_not_measured:'尚无对应增益',ambiguous_calibration:'存在多个适用增益',
    operating_context_shift:'运行条件偏离校准场景',additive_multi_parameter_approximation:'多参数叠加近似',gain_direction_uncertain:'增益方向不确定',
    response_window_mismatch:'评估窗与增益校准窗不同',limited_linear_fit:'一阶拟合能力有限',proposal_unavailable:'缺少本轮拟动作',scenario_limit:'仲裁分支超过计算上限',
    matching_decision_preview_unavailable:'当前时刻未记录决策快照',inconsistent_public_observation:'拟动作的观测时刻不一致',parameter_observation_mismatch:'拟动作与图中参数观测不一致',unsupported_resource_arbitration:'该共享参数尚未配置随机仲裁',unsupported_mitigation_mode:'当前控制器未使用此方法要求的随机仲裁',empirical_observational_approximation:'执行数据回归近似，非受控因果增益',same_seed_validation_only:'仅同种子验证',empirical_validation_failed:'回归尚未通过留出验证',empirical_context_mismatch:'回归适用场景不匹配',empirical_unidentified_direction:'数据尚未识别这个动作方向',empirical_operating_point_outside_span:'当前工作点不在已观测子空间内',empirical_outside_observed_range:'超出回归数据的参数范围',empirical_horizon_mismatch:'回归观察窗口不匹配'};
  function renderConflictQuantificationStatus(){
    const data=currentQuantification();
    $('conflict-method').setAttribute('aria-busy',String(conflictQuantificationBusy));
    $('conflict-method').dataset.state=data?.error?'error':data?.method?'success':'default';
    $('conflict-quantification-refresh').disabled=conflictQuantificationBusy||!quantificationKey()||location.protocol==='file:';
    $('conflict-quantification-export').disabled=!data?.method;
    $('conflict-align-snapshot').hidden=data?.suggested_slot==null;
    $('conflict-quantification-status').textContent=conflictQuantificationBusy?'正在计算…':data?.error||data?.label||quantificationReasons[data?.reason]||
      (data?.method?`当前决策 ${time(data.preview_slot)}`+(data.snapshot_alignment?` · 采用此前 ${fmt(data.snapshot_alignment.lag_ms,2)} ms 的已记录拟动作`:'')+` · ${data.groups.length} 个组合 · 独立量化，不读取部署结果`:'等待同一时刻的意图与拟动作');
    const alert=$('quantification-alert');
    alert.hidden=!(data?.error||data?.reason);
    alert.textContent=data?.error||quantificationReasons[data?.reason]||data?.reason||'';
    renderQuantificationPanel();
  }
  function quantifiedGroup(group){return currentQuantification()?.groups?.find(g=>g.group_id===group.id);}
  function renderGoalObstruction(q,group){
    const strengthMethod=currentQuantification()?.method==='goal_obstruction_v2';
    const counts=q.structural_counts||SmoConflictRisk.conflictCounts(group.conflict_summaries||[]);
    const reasons={proposal_unavailable:'缺少有效拟动作',incomplete_scheduler_bundle:'下行调度策略与 UE 优先级的原子动作包不完整',unsupported_arbitration_resource:'此共享参数尚未定义受支持的仲裁方式',arbitration_outcome_limit:'仲裁分支超过计算上限',reference_space_limit:'完整策略子集超过计算上限',reference_projection_incomplete:'增益或单位不足'};
    const missing=q.projection_coverage?.unknown_links||[],screened=(q.projection_coverage?.excluded_links||[]).filter(r=>r.action_context);
    const link=r=>`${find('rcp:'+r.parameter)?.label||r.parameter} → ${chineseKpis[r.kpi]||r.kpi}`;
    const num=n=>n==null?'—':n!==0&&Math.abs(n)<.0001?n.toExponential(2):fmt(n,4);
    const scale=k=>{const c=currentQuantification()?.intent_contexts.find(c=>c.kpi===k);return `${num(c?.scale)} ${find('kpi:'+k)?.unit||''}`;};
    return `<div class="group-quantification"><p class="quant-score" title="${strengthMethod?'目标受阻强度':'目标增益损失比例'} · KG 参考近似"><strong>${q.score==null?'—':fmt(q.score,1)}</strong><span>/ 100</span></p>`+
      (q.score==null?`<p class="quant-warning">${esc(missing.length?`缺少 ${missing.length} 条动作增益`:reasons[q.reason]||q.reason||'证据不足')}</p>`:
        q.model_coverage_complete===false?`<p class="quant-warning">部分量化 · KPI 覆盖 ${q.projection_coverage.covered_kpis}/${q.projection_coverage.required_kpis}</p>`:
        screened.length?`<p class="quant-caption">${screened.length} 项动作关系已筛除</p>`:'')+
      '<details><summary>量化详情</summary>'+
      `<p>${q.no_active_push?'当前保留关系中未量化到主动推动目标的作用。':''}${strengthMethod?'目标受阻强度':'目标增益损失比例'}${q.score_status==='reference_approximation'?' · 参考近似':''}</p>`+
      (graph().statistical_filter?.enabled?'<p>仅在显著 P–K 关系范围内量化；被排除关系不代表物理影响为零。</p>':'')+
      (q.projection_coverage?.kpis||[]).filter(r=>r.excluded&&!r.supported).map(r=>`<p class="quant-warning">${esc(chineseKpis[r.kpi]||r.kpi)}：当前动作没有保留的有效增益。该分数仅反映其余已知关系，不代表完整零风险。</p>`).join('')+
      screened.map(r=>`<p class="quant-caption">已筛除：${esc(link(r))} · 当前动作有测量记录，但未通过显著性筛选；该估计不计分。</p>`).join('')+
      missing.map(r=>`<p class="quant-warning">无法量化：${esc(link(r))} · ${r.notes.includes('incompatible_or_missing_gain_units')?'增益单位缺失或不兼容':'缺少适用于当前动作的增益'}。</p>`).join('')+
      '<p>以完整 xApp 子集能提供的正向参考作用为基准，计算整个组合执行后的作用损失。共享参数按随机仲裁后的状态计入一次。</p>'+
      (q.goal_components||[]).map(c=>`<div class="quant-detail"><strong>${esc(chineseKpis[c.kpi]||c.kpi)}</strong><p>可用正向作用 ${num(c.potential)} · 组合期望作用 ${num(c.expected_goal_drive)}</p>`+
        (strengthMethod?`<p>期望作用损失 ${num(c.expected_reference_loss)} · 意图余量系数 ${num(c.intent_margin_factor)} · 受阻强度 ${num(c.obstruction_strength)}</p>`:`<p>受阻比例 ${c.expected_loss_fraction==null?'—':fmt(100*c.expected_loss_fraction,1)+'%'}</p>`)+
        `<p>公共尺度 ${esc(scale(c.kpi))} · KPI 价值权重 1 · 作用无量纲</p>${c.no_active_push?'<p class="quant-warning">当前保留关系中未量化到主动推动作用；零受阻不代表能够满足意图。</p>':''}</div>`).join('')+
      (strengthMethod?`<p>综合受阻强度 B = ${num(q.raw_obstruction_strength)}，取各目标受阻强度的最大值；评分 = 100B / (1+B)，越低越好。</p><p>余量系数 = 1 / (1 + 当前非负归一化余量)。不除以各组合自己的正向能力，不截断反向作用的损失量。</p>`:'<p>综合评分取各意图 KPI 期望损失比例的最大值。</p>')+
      '<p>作用已按各 KPI 的共同尺度归一化，不是物理单位下的 KPI 变化量。</p>'+
      `<p class="quant-counts">潜在位置：直接 ${counts.direct} 处 · 间接 ${counts.indirect} 处${counts.implicit?' · 隐式 '+counts.implicit+' 处':''}。处数不参与评分。</p>`+
      `<p>参考完整策略子集 ${q.reference_subset_count||0} 组 · 组合仲裁结果 ${q.actual_outcomes?.length||0} 种。</p>`+
      (q.held_lease_state_not_reconstructed?'<p class="quant-warning">当前包含已启用的 xApp；未重建持有中的仲裁租约。</p>':'')+
      '<p class="quant-caption">仅评估这一时刻的拟动作，按新一轮均匀随机仲裁求期望；不是后续闭环窗口的预测，也不是意图失败概率。实测结果独立展示。</p></details></div>';
  }
  function renderNetObstruction(q,group){
    const num=n=>n==null?'—':n!==0&&Math.abs(n)<.0001?n.toExponential(2):fmt(n,4);
    const signed=n=>n==null?'—':(n>0?'+':'')+num(n);
    const counts=q.structural_counts||SmoConflictRisk.conflictCounts(group.conflict_summaries||[]);
    const reasons={proposal_unavailable:'缺少有效拟动作',reference_projection_incomplete:'当前动作缺少适用增益或单位',reference_projector_unavailable:'参考投影引擎不可用',
      kpi_scale_unavailable:'目标 KPI 缺少公共尺度',kpi_observation_unavailable:'目标 KPI 缺少当前观测',arbitration_outcome_limit:'仲裁分支超过计算上限'};
    return `<div class="group-quantification"><p class="quant-score" title="净目标受阻 · KG 参考近似"><strong>${q.score==null?'—':fmt(q.score,1)}</strong><span>/ 100</span></p>`+
      (q.score==null?`<p class="quant-warning">${esc(reasons[q.reason]||q.reason||'证据不足')}</p>`:
        q.model_coverage_complete===false?'<p class="quant-warning">部分模型覆盖；该分数仅反映已有参考关系。</p>':'')+
      '<details><summary>量化详情</summary><p>当前联合拟动作经随机仲裁求期望，再按物理范围与意图区间计算净参考作用。</p>'+
      (q.goal_components||[]).map(c=>`<div class="quant-detail"><strong>${esc(chineseKpis[c.kpi]||c.kpi)}</strong>`+
        `<p>归一化净作用 ${signed(c.reference_progress)} · 公共尺度 ${esc(num(c.kpi_scale))} ${esc(find('kpi:'+c.kpi)?.unit||'')}</p></div>`).join('')+
      `<p>净参考作用 U = ${signed(q.reference_effect)}；有负作用时取最差项，否则等权平均。受阻强度 D = ${num(q.raw_obstruction_strength)}，评分 = 100D / (1+D)。</p>`+
      `<p class="quant-counts">潜在位置：直接 ${counts.direct} 处 · 间接 ${counts.indirect} 处${counts.implicit?' · 隐式 '+counts.implicit+' 处':''}。处数不参与评分。</p>`+
      `<p>联合仲裁结果 ${q.arbitration_outcome_count??0} 种。</p>`+
      (q.held_lease_state_not_reconstructed?'<p class="quant-warning">未重建已启用 xApp 持有中的仲裁租约。</p>':'')+
      '<p class="quant-caption">作用无量纲。零受阻不等于意图达标；这不是闭环 KPI 预测或失败概率，实际 Metric 独立展示。</p></details></div>';
  }
  function renderGroupQuantification(group){
    const data=currentQuantification();
    const q=quantifiedGroup(group);
    if(!q)return '<p class="quant-unavailable">'+esc(conflictQuantificationBusy?'量化中…':currentQuantification()?.error||quantificationReasons[currentQuantification()?.reason]||'当前快照尚未量化')+'</p>';
    if(currentQuantification()?.method==='net_intent_obstruction_v1')return renderNetObstruction(q,group);
    if(currentQuantification()?.method.startsWith('goal_obstruction_'))return renderGoalObstruction(q,group);
    const name=id=>find(id)?.label||id;
    const signed=n=>n==null?'未量化':(n>0?'+':'')+(n!==0&&Math.abs(n)<.0001?n.toExponential(2):fmt(n,4));
    const coverage=q.coverage,contexts=currentQuantification().intent_contexts||[];
    const counts=q.structural_counts||SmoConflictRisk.conflictCounts(group.conflict_summaries||[]);
    const notes={population_direction_projection:'非统一动作投影到已测群体方向',uncovered_structured_components:'其余配置字段未计入参考影响',uncovered_population_members:'部分 UE 影响未覆盖',outside_reference_domain:'超出参考范围，沿用一阶权重近似',uncertain_gain_direction:'增益符号不确定',observed_zero_reference_weight:'此参考条件下未观察到变化，按中性权重计入',uncertain_joint_profile_filtered:'联合配置响应不确定，已按虚线设置排除；保留可用单字段近似',limited_reference_fit:'局部拟合较粗',common_reference_contrast:'共用基准策略的对比近似',sleep_family_reference_weight:'以 active／sleep 类别沿用已测参考权重',unmeasured_sleep_profile:'未区分具体休眠档位的影响强度',reference_context_shift:'当前上下文偏离参考标定条件',reference_context_approximation:'同类动作的上下文近似',uncertain_gain_hidden:'对应参考权重不确定，已按隐藏虚线设置排除',reference_direction_unavailable:'缺少此动作方向的参考增益',local_reference_drive_not_kpi_change:'参考影响，不是 KPI 变化预测',proposal_unavailable:'缺少有效的已记录拟动作'};
    const finiteComponent=c=>typeof c.gain==='number'&&typeof c.parameter_step==='number';
    const roles={toward_interval:'推动进入意图区间',against_interval:'偏离意图方向',consuming_margin:'消耗意图安全余量',building_margin:'增加意图安全余量',inside_interval_no_unique_direction:'在区间内，无唯一改善方向',neutral:'本轮中性',context_unavailable:'方向上下文不足'};
    const direction=q.intent_direction;
    return `<div class="group-quantification"><p class="quant-score"><span>综合冲突风险</span><strong>${q.score==null?'—':fmt(q.score,1)}</strong><span>/ 100</span></p>`+
      `<p class="quant-caption">意图加权控制动力对抗 · ${q.score_status==='partial_evidence'?'部分证据':q.score_status==='reference_approximation'?'参考近似':'参考量化'} · 链路 ${coverage.known_chains}/${coverage.total_chains}</p>`+
      (direction.against||direction.consuming_margin?`<p class="quant-warning">${direction.against?direction.against+' 项参考影响偏离意图方向':''}${direction.against&&direction.consuming_margin?'；':''}${direction.consuming_margin?direction.consuming_margin+' 项消耗安全余量':''}</p>`:'')+
      `<details><summary>量化详情</summary>`+
      '<p>先汇总各 xApp 对意图 KPI 的带符号影响，再以意图尺度归一化，统一计算动力抵消比例。不按直接、间接类型分别打分。</p>'+
      `<p>数值覆盖 ${coverage.comparable_pairs}/${coverage.total_pairs} 对；${coverage.partial_chains} 条链路含参考近似。零分表示已量化部分未出现方向抵消，不代表意图达标。</p>`+
      `<p class="quant-counts">组合记录的潜在位置：直接 ${counts.direct} 处 · 间接 ${counts.indirect} 处${counts.implicit?' · 隐式 '+counts.implicit+' 处':''}。处数不参与评分。</p>`+
      (coverage.missing_proposal_apps.length?`<p class="quant-warning">缺少 ${esc(coverage.missing_proposal_apps.map(name).join('、'))} 的有效拟动作，未将未知影响计作零。</p>`:'')+
      contexts.map(c=>`<div class="quant-detail"><strong>${esc(chineseKpis[c.kpi]||c.kpi)}</strong><p>归一化尺度 ${esc(fmt(c.scale,4))} · 意图权重 ${esc(fmt(c.weight,4))}${c.margin==null?'':' · 当前边界余量 '+esc(fmt(c.margin,4))}</p></div>`).join('')+
      (q.contributors.length?q.contributors.slice(0,3).map(c=>`<div class="quant-detail"><strong>${esc(c.apps.map(name).join(' ↔ '))}</strong><p>${esc(chineseKpis[c.kpi]||c.kpi)} · 对综合风险贡献 ${esc(fmt(c.score_contribution,2))} 分</p></div>`).join(''):'<p>已量化影响中暂无反向动力抵消。</p>')+
      q.app_vectors.filter(v=>!v.no_visible_graph_path).map(v=>`<div class="quant-detail"><strong>${esc(name(v.app_id))} → ${esc(chineseKpis[v.kpi]||v.kpi)}</strong><p>归一化参考影响 ${esc(signed(v.normalized_drive))} · ${esc(roles[v.intent_role])}${v.partial?' · 部分证据':''}</p></div>`).join('')+
      q.chains.map(l=>`<div class="quant-chain"><span>${esc(name(l.app_id))} → ${esc(name('rcp:'+l.parameter))} → ${esc(chineseKpis[l.kpi]||name('kpi:'+l.kpi))}</span><code>增益 × 拟参数变化 → 参考影响 ${esc(signed(l.drive))}${l.drive==null?'':' '+esc(find('kpi:'+l.kpi)?.unit||'')}</code>${(l.components||[]).filter(c=>finiteComponent(c)).map(c=>`<small>权重 ${esc(signed(c.gain))} × 参数步长 ${esc(signed(c.parameter_step))}</small>`).join('')}<small>期望 ${esc(rangeText(l.intent_range))} · ${esc(roles[l.intent_role])}</small><small>${esc(l.notes.map(n=>notes[n]||n).join('；'))}</small></div>`).join('')+
      '<p class="quant-caption">分数衡量参考控制动力的对抗，不是 KPI 变化预测、达标程度或失败概率。实测结果独立展示。</p></details></div>';
  }
  function renderQuantificationPanel(){
    const groups=SmoGraph.quantificationGroups(state.intent,state.combinationCatalog,state.showDashed);
    const data=currentQuantification(),body=$('quantification-body');
    const signature=JSON.stringify([state.bundle.graph_sha256,!!state.intent,uncalibratedEnvironment(),conflictQuantificationBusy,
      groups.map(g=>[g.id,g.deployment_disposition]),$('quantification-full').open]);
    // Status polling must not rebuild an unchanged, potentially large evidence table.
    if(quantificationPanelView?.data===data&&quantificationPanelView.signature===signature)return;
    quantificationPanelView={data,signature};
    // Disclosure is presentation only: all structural groups are still quantified.
    const open=new Set([...body.querySelectorAll('tr[data-quantification-group]')].filter(r=>r.querySelector('details')?.open).map(r=>r.dataset.quantificationGroup));
    $('quantification-count').innerHTML=`<span>全部组合</span><strong>${groups.length}</strong><span>组</span>`;
    $('quantification-full').hidden=!groups.length;
    $('quantification-full-summary').textContent=`完整组合量化（${groups.length} 组）`;
    $('quantification-summary').textContent=!state.intent?'解析意图后独立量化覆盖目标的组合。':
      uncalibratedEnvironment()?'当前场景参数–KPI 增益待标定；保留组合与拟动作，缺失分数显示 —。':
      '使用当前意图、同一快照拟动作及图谱增益；不读取部署后的 Metric。';
    if(!groups.length){body.innerHTML='<tr><td colspan="3">'+(state.intent?'当前意图没有完整的图谱关联组合。':'等待意图。')+'</td></tr>';return;}
    const dispositions={excluded:'已从部署候选剔除',reserve:'备选，未进入部署候选',not_selected:'未进入部署候选'};
    const row=group=>{
      const q=quantifiedGroup(group),counts=q?.structural_counts||SmoConflictRisk.conflictCounts(group.conflict_summaries||group.conflicts||[]);
      const names=group.app_ids.map(id=>find(id)?.label||id).join(' + '),disposition=group.deployment_disposition||'candidate';
      const evidence=q?.score==null?(uncalibratedEnvironment()?'参数–KPI 增益待标定':q?.reason==='reference_projection_incomplete'?'当前动作缺少适用增益':quantificationReasons[q?.reason]||q?.reason||data?.error||'等待量化'):'同快照拟动作与一阶增益';
      return `<tr data-quantification-group="${esc(group.id)}" data-deployment-excluded="${disposition!=='candidate'}"><td><strong>${esc(names)}</strong>${dispositions[disposition]?`<span class="quantification-excluded">${dispositions[disposition]}</span>`:''}</td><td>${renderGroupQuantification(group)}</td><td><p>直接 ${counts.direct} 处 · 间接 ${counts.indirect} 处${counts.implicit?' · 隐式 '+counts.implicit+' 处':''}</p><p class="quant-unavailable">${esc(evidence)}</p></td></tr>`;
    };
    body.innerHTML=$('quantification-full').open?groups.map(g=>row(g)).join(''):'';
    for(const r of body.querySelectorAll('tr[data-quantification-group]'))if(open.has(r.dataset.quantificationGroup)){const detail=r.querySelector('details');if(detail)detail.open=true;}
  }
  function renderCombinationCatalog() {
    const openDetails=new Map();
    for(const row of $('catalog-body').querySelectorAll('tr')){
      const group=row.querySelector('[data-view-group]')?.dataset.viewGroup;
      if(group)for(const detail of row.querySelectorAll('details')){
        const summary=detail.querySelector('summary')?.textContent;
        if(summary)openDetails.set(group+'|'+summary,detail.open);
      }
    }
    const catalog=state.combinationCatalog,batch=matchingBatch(),groups=catalog?.groups||[];
    const retained=activeBatch(),measuredBatch=batch||(SmoGraph.retainedEvidenceCompatible(retained,catalog,state.bundle.graph,state.activeIntents.map(r=>r.id))?retained:null);
    const initial=SmoGraph.initialSelectionSummary(catalog);
    const blocked=SmoGraph.hasBlockingConstraints(state.intent),needsClarification=!!state.intent.questions?.length;
    const countReady=!!batch||(!needsClarification&&!blocked&&!!catalog);
    $('catalog-count').classList.toggle('has-count',countReady);
    if(countReady)$('catalog-count').innerHTML=`<span>${initial?'主候选':'可能组合'}</span><strong>${groups.length}</strong><span>组</span>`;
    else $('catalog-count').textContent=needsClarification?'意图待澄清':blocked?'共同约束待处理':'请更新观测';
    $('catalog-export').disabled=!groups.length&&!catalog?.excluded_groups?.length;
    $('catalog-export').textContent=batch?'导出实测 CSV':'导出列表';
    $('catalog-note').textContent=batch?`本批次 ${groups.length} 个组合 · 量化独立分析当前控制链；右侧独立展示实际部署前后效果。`:state.bundle.scenario_view?.manual_evaluation_enabled===false?'候选组合覆盖全部目标 KPI · 当前压力场景提供图谱与冲突分析，尚未配置对应的部署评估。':'候选组合覆盖全部目标 KPI；点击“评估全部组合”获得每组实际效果。';
    if(initial)$('catalog-note').textContent=initial.text+' '+$('catalog-note').textContent;
    if(!batch&&measuredBatch)$('catalog-note').textContent=(initial?.text||'')+` 下表复用原批次 ${measuredBatch.total} 组中已有的对应实测结果。`;
    renderActionExclusions(catalog);
    const missingDirections=catalog?.direction_missing_kpis||[];
    const emptyMessage=catalog?.direct_action_filter?.removed_primary_count?'主候选已全部因反向直接冲突被剔除。'
      :missingDirections.length?'尚无已确认能力方向覆盖：'+missingDirections.map(k=>chineseKpis[k]||find('kpi:'+k)?.label||k).join('、')+'。关联组合保留在备选中。'
      :graph().statistical_filter?.enabled?'仅按显著关系筛选后，暂无覆盖全部意图的主候选；这不表示网络在理论上不可实现。'
      :catalog?.selection_algorithm==='semantic_influence_coverage_v2'?'尚无覆盖全部目标能力方向的主候选；未确认的组合保留在备选中。'
      :'当前没有关联覆盖全部 KPI 的组合。';
    $('catalog-body').innerHTML=(!batch&&needsClarification)?'<tr><td colspan="3">'+esc('意图待澄清：'+state.intent.questions.join(' '))+'</td></tr>':(!batch&&blocked)?'<tr><td colspan="3">共同约束尚未解决，暂不枚举组合。</td></tr>':!countReady?'<tr><td colspan="3">请更新观测以读取图结构分析。</td></tr>':!groups.length?'<tr><td colspan="3">'+esc(emptyMessage)+'</td></tr>':groups.map(g=>{
      const apps=g.app_ids.map(id=>intentCandidates().find(c=>c.app_id===id)||find(id)).sort((a,b)=>a.label.localeCompare(b.label));
      const member=measuredBatch?.members.find(m=>m.group_id===g.id&&[...m.app_ids].sort().join('|')===[...g.app_ids].sort().join('|'));
      const outcome=memberOutcome(member,(measuredBatch?.ranges||[]).map(r=>r.kpi));
      const observed=member?.comparison?.result_evaluation;
      const counts=quantifiedGroup(g)?.structural_counts||SmoConflictRisk.conflictCounts(g.conflict_summaries||g.conflicts||[]);
      const displayOutcome=outcome==='unmet'&&observed?.no_regression===true?(observed.after.score>0?'partial':'neutral'):outcome;
      const badges={satisfied:'✓ 全部达标',partial:'↗ 部分改善，尚未全部达标',neutral:'— 无改善，尚未全部达标',unmet:'× 未达标',insufficient:'? 数据不足',pending:'待评估',running:'评估中',failed:'! 运行失败',cancelled:'已停止',invalid:'! 约束无效'};
      return `<tr data-combination-group="${esc(g.id)}" data-outcome="${displayOutcome}" data-regression="${member?.comparison?.result_evaluation?.no_regression===false}" class="${state.graphGroup===g.id?'catalog-viewed':''}"><td><span class="catalog-outcome-badge">${badges[displayOutcome]}</span><button type="button" class="catalog-group-button" data-view-group="${esc(g.id)}" aria-label="${esc('查看组合：'+apps.map(c=>c.label).join(' + '))}" aria-pressed="${state.graphGroup===g.id}"><strong>${esc(apps.map(c=>c.label).join(' + '))}</strong></button><details class="combination-extra"><summary>组合详情</summary>${member?`<small class="batch-member-status">${esc(memberPhases[member.phase]||member.phase)}${member.worker_label?' · '+esc(member.worker_label):''}</small>`:''}${member?.phase==='complete'?`<button type="button" class="text-button catalog-deploy" data-view-result="${esc(member.job_id)}" data-result-group="${esc(g.id)}">查看实测回放</button>`:''}${groupCoverage(g)}</details></td><td class="combination-risk">${renderGroupQuantification(g)}<p class="combination-conflict-counts">直接 ${counts.direct} 处 · 间接 ${counts.indirect} 处${counts.implicit?' · 隐式 '+counts.implicit+' 处':''}</p></td><td class="combination-metric">${renderMeasuredMember(member)}</td></tr>`;
    }).join('');
    for(const row of $('catalog-body').querySelectorAll('tr')){
      const group=row.querySelector('[data-view-group]')?.dataset.viewGroup;
      if(group)for(const detail of row.querySelectorAll('details')){
        const key=group+'|'+detail.querySelector('summary')?.textContent;
        if(openDetails.has(key))detail.open=openDetails.get(key);
      }
    }
    for(const button of $('catalog-body').querySelectorAll('[data-view-group]'))button.addEventListener('click',()=>{state.graphGroup=state.graphGroup===button.dataset.viewGroup?null:button.dataset.viewGroup;state.selected=null;state.focus='';$('app-focus').value='';render();});
    for(const button of $('catalog-body').querySelectorAll('[data-view-result]'))button.addEventListener('click',()=>viewDeploymentResult(button.dataset.viewResult,button.dataset.resultGroup));
  }
  function renderActionExclusions(catalog){
    const info=catalog?.direct_action_filter,groups=catalog?.excluded_groups||[];
    $('action-filter-note').hidden=!info;
    $('action-filter-note').textContent=!info?'':info.status!=='ready'?'反向直接冲突过滤：缺少一致的决策快照，暂未剔除。':`已确认反向直接冲突：剔除 ${groups.length} 组 · 决策 ${time(info.decision_slot)}`+(info.lag_ms>0?` · 使用此前 ${fmt(info.lag_ms,2)} ms 的记录`:'')+((info.unknown_apps.length||info.unresolved_shared_actions.length)?' · 未确认的动作方向保留':'');
    $('action-exclusions').hidden=!groups.length;
    $('action-exclusions-summary').textContent=`查看 ${groups.length} 组剔除原因`;
    const name=id=>find(id)?.label||id;
    $('action-exclusions-body').innerHTML=groups.map(g=>`<div class="action-exclusion"><strong>${esc(g.app_ids.map(name).join(' + '))}</strong>`+
      g.action_exclusion.pairs.flatMap(pair=>pair.evidence.map(e=>`<p>${esc(name('rcp:'+e.parameter))}${e.component?' · '+esc(e.component):''} · ${esc(e.entity)}：${esc(fmt(e.current,4))} → ${e.actions.map(a=>`${esc(name(a.app_id))} ${esc(fmt(a.proposed,4))} ${a.direction==='increase'?'↑':'↓'}`).join(' / ')}</p>`)).join('')+'</div>').join('');
  }
  function quantificationText(row){
    const q=row.quantification;
    if(!q?.complete)return '<small>满足程度：'+(row.after?'数据不足':'待部署后观测')+'</small>';
    const signed=v=>(v>0?'+':'')+fmt(v,2)+'%';
    let text='';
    if(q.mode==='directional_change')text=`相对部署前${q.direction===1?'提升':'降低'} ${q.favorable_change_percent==null?'—':signed(q.favorable_change_percent)}`;
    else if(q.target_progress_percent!=null)text=`要求完成度 ${fmt(q.target_progress_percent,2)}%`;
    else if(q.mode==='maintain_range')text='部署前已达标 · '+(q.satisfied?'保持达标':'部署后越界');
    else text=q.satisfied?'已进入期望区间':'尚未进入期望区间';
    if(q.mode!=='directional_change'&&q.relative_change_percent!=null)text+=` · 相对变化 ${signed(q.relative_change_percent)}`;
    if(q.mode==='directional_change'&&q.favorable_change_percent==null)text+=` · 绝对改善 ${fmt(q.favorable_delta,4)} ${row.unit}`;
    const margin=q.after_margin;
    const detail=margin==null?'':margin<0?`距区间差 ${fmt(-margin,4)} ${row.unit}`:margin===0?(q.satisfied?'位于目标边界':'严格边界未满足'):`区间内余量 ${fmt(margin,4)} ${row.unit}`;
    return `<small class="intent-quantification">${esc(text)}</small>${detail?'<small>'+esc(detail)+'</small>':''}`;
  }
  function renderMeasuredMember(member){
    if(!member)return '<span class="muted">尚未评估</span>';
    if(member.phase!=='complete')return `<span class="muted">${esc(memberPhases[member.phase]||member.phase)}</span>${member.error?`<small>${esc(member.error)}</small>`:''}`;
    const comparison=member.comparison;
    if(!comparison)return '<span class="muted">实测数据不可用</span>';
    const result=comparison.after,evaluation=comparison.result_evaluation;
    const keys=new Set((evaluation?.metrics||[]).map(m=>m.kpi));
    return renderObservedResult(evaluation)+(member.baseline_matches_reference===false?'<p class="quant-warning">基线有差异</p>':'')+
      '<details class="result-kpi-details"><summary>指标详情</summary>'+(comparison.rows||[]).filter(r=>keys.has(r.kpi)).map(row=>{
      const a=row.before,b=row.after,m=evaluation.metrics.find(m=>m.kpi===row.kpi),unit=metricUnit(row.kpi,row.unit);
      const scoreUnit=m.score_unit==='percentage_points'?' 个百分点':'%';
      const status=b?.status==='satisfied'?'达标':b?.status==='unavailable'?'数据不足':'未达标';
      const extrema=b?.measurement?.constraint_uses_extrema;
      return `<div class="batch-kpi"><strong>${esc(chineseKpis[row.kpi]||row.label)}</strong><code>${esc(metricValue(row.kpi,a?.value))} → ${esc(metricValue(row.kpi,b?.value))} ${esc(unit)}</code>`+
        `<small>意图方向改善 ${esc(resultScoreValue(m.improvement_score))}${esc(scoreUnit)}${m.regression?' · 退步不可抵消':''}</small>`+
        `<small>剩余违背 ${esc(metricValue(row.kpi,m.after.distance))} ${esc(unit)} · 归一化 ${esc(resultScoreValue(m.after.normalized_violation))} 分</small>`+
        (m.target_progress_percent!=null?`<small>区间差距缩减 ${esc(fmt(m.target_progress_percent,2))}%</small>`:'')+
        `<small>${esc(rangeText(row))} · ${esc(status)}${b?.strict_boundary_violation?' · 严格端点未满足':''}</small>`+
        (extrema?`<small>改善分按窗口均值；达标按整窗范围：${esc(metricValue(row.kpi,b.measurement.minimum))}–${esc(metricValue(row.kpi,b.measurement.maximum))} ${esc(unit)}</small>`:'')+
        (a?.measurement?.sample_count!=null||b?.measurement?.sample_count!=null?`<small>样本 ${esc(fmt(a?.measurement?.sample_count,0))} → ${esc(fmt(b?.measurement?.sample_count,0))}</small>`:'')+'</div>';
    }).join('')+'</details>';
  }
  function resultScoreValue(value){
    if(value==null)return '—';
    if(value!==0&&Math.abs(value)<.01)return (value>0?'+':'−')+'<0.01';
    return (value>0?'+':'')+fmt(value,2);
  }
  function renderObservedResult(evaluation){
    if(!evaluation)return '<p class="quant-unavailable">意图改善分尚未生成</p>';
    const after=evaluation.after,before=evaluation.before,view=SmoGraph.evaluationPresentation(evaluation);
    const missingScale=evaluation.metrics.filter(m=>m.scale==null);
    const regressions=(evaluation.regression_kpis||[]).map(k=>chineseKpis[k]||k).join('、');
    const passed=evaluation.metrics.filter(m=>m.after.satisfied===true).length;
    const stateLabel=view.tone==='regressed'?'目标退步':view.tone==='partial'?'部分改善':view.tone==='neutral'?'无改善':view.label;
    return `<div class="observed-result" data-result="${view.tone}"><p class="observed-score" title="意图改善分"><strong>${esc(resultScoreValue(after.score))}</strong></p>`+
      (after.complete?`<p class="observed-caption">${esc(stateLabel)} · 达标 ${passed}/${evaluation.metrics.length}</p><p class="observed-change" title="各意图目标归一化剩余违背的最大值">违背 ${esc(fmt(evaluation.violation?.after?.max,2))}</p>`:`<p class="quant-warning">${esc(missingScale.length?'缺少尺度：'+missingScale.map(m=>chineseKpis[m.kpi]||m.kpi).join('、'):'意图目标数据不足')}</p>`)+
      '</div>';
  }
  function groupCoverage(group) {
    const coverage=group.intent_coverage||{},keys=coverage.required_kpis||[];
    if(!group.coverage_paths)return '';
    const direction=group.initial_selection?.direction_coverage||state.combinationCatalog?.groups.find(g=>g.id===group.id)?.initial_selection?.direction_coverage;
    const names=ids=>ids.map(id=>intentCandidates().find(c=>c.app_id===id)?.label||find(id)?.label||id).join(' + ');
    const directionNames={increase:'提高',decrease:'降低',maintain:'保持'};
    const semantic=direction?`<details class="catalog-coverage"><summary>能力方向覆盖 ${Object.values(direction).filter(r=>r.supporting_apps.length).length}/${Object.keys(direction).length}</summary><ul>${Object.entries(direction).map(([k,r])=>`<li><strong>${esc(chineseKpis[k]||find('kpi:'+k)?.label||k)} · ${esc(r.required.map(d=>directionNames[d]||d).join(' / '))}</strong><span>${esc(r.supporting_apps.length?names(r.supporting_apps):'尚无已确认能力方向')}</span></li>`).join('')}</ul></details>`:'';
    return semantic+`<details class="catalog-coverage"><summary>${direction?'拓扑覆盖':'KPI 覆盖'} ${coverage.covered_count}/${coverage.required_count}</summary><ul>${keys.map(key=>{
      const label=chineseKpis[key]||find('kpi:'+key)?.label||key;
      const traces=(group.coverage_paths[key]||[]).flatMap(row=>row.paths.map(path=>
        (intentCandidates().find(c=>c.app_id===row.app_id)?.label||find(row.app_id)?.label||row.app_id)+' → '+(find('rcp:'+path.parameter)?.label||path.parameter)+' → '+label));
      return `<li><strong>${esc(label)}</strong>${traces.map(text=>`<span>${esc(text)}</span>`).join('')}</li>`;
    }).join('')}</ul></details>`;
  }
  let deploymentState={enabled:false,latest:null,latest_batch:null},deploymentBusy=false,deploymentError='',deploymentPolling=false,lastBatchRenderKey=null;
  const activeJob=()=>deploymentState.latest;
  const activeBatch=()=>deploymentState.latest_batch;
  const matchingBatch=()=>{const b=activeBatch();return b&&b.active_intent_ids?.slice().sort().join(',')===state.activeIntents.map(r=>r.id).sort().join(',')&&SmoGraph.batchCatalogCompatible(b,state.combinationCatalog)?b:null;};
  const managedComparison=()=>{
    const batch=matchingBatch();
    if(batch){const member=batch.members.find(m=>m.group_id===state.graphGroup);return {...batch,comparison:member?.comparison||null,selected_names:member?.selected_names||null};}
    return null;
  };
  const deploymentPhases={queued:'已排队',initializing:'初始化仿真',collecting_baseline:'采集部署前基线',deploying:'等待 RIC 执行动作',observing:'观察部署后效果',evaluating:'计算实测指标',finished:'保存数据',saving:'保存数据',complete:'全部组合评估完成',complete_with_failures:'批次结束，部分组合运行失败',failed:'运行失败',cancelled:'已停止',connection_lost:'连接中断，任务与数据保留',reconnecting:'重新连接中',running:'逐组评估中',pausing:'正在暂停并保存',paused:'批次已暂停'};
  const memberPhases={pending:'待评估',running:'正在评估',complete:'已评估',failed:'运行失败',cancelled:'已停止'};
  function memberOutcome(member,required=[]){
    if(!member)return 'pending';
    if(member.phase!=='complete')return ['pending','running','failed','cancelled'].includes(member.phase)?member.phase:'pending';
    const valid=new Set(['satisfied','below_range','above_range','outside_range']);
    const c=member.comparison,rows=c?.rows||[];
    if(c?.status==='invalid_constraints'||rows.some(r=>['empty_range','legacy_kpi'].includes(r.after?.status)))return 'invalid';
    const keys=required.length?required:rows.map(r=>r.kpi),metrics=keys.map(k=>rows.find(r=>r.kpi===k));
    if(!keys.length||metrics.some(r=>!r||!valid.has(r.before?.status)||!valid.has(r.after?.status)))return 'insufficient';
    return metrics.every(r=>r.after.status==='satisfied')?'satisfied':'unmet';
  }
  function batchOutcomeCounts(batch){
    const counts={satisfied:0,unmet:0,insufficient:0,pending:0,running:0,failed:0,cancelled:0,invalid:0};
    const required=[...new Set((batch.ranges||[]).map(r=>r.kpi))];
    for(const member of batch.members||[])counts[memberOutcome(member,required)]++;
    return counts;
  }
  const deploymentRunning=()=>{const b=activeBatch();return b?!['complete','complete_with_failures','paused'].includes(b.phase):!!activeJob()&&!['complete','failed','cancelled'].includes(activeJob().phase);};
  function renderDeployment(){
    const batch=activeBatch(),job=activeJob(),running=deploymentRunning(),jobs=deploymentState.running_jobs||[],capacity=deploymentState.concurrency||1;
    const groups=state.combinationCatalog?.groups||[],count=groups.length;
    $('deployment-panel').dataset.state=deploymentBusy?'loading':deploymentError||batch?.phase==='complete_with_failures'?'error':batch?.phase==='complete'?'success':'default';
    $('deployment-selection').textContent=batch&&!matchingBatch()&&state.intent?`历史批次 ${batch.total} 组 · 当前候选 ${count} 组`:`${capacity} 台 AGX 并发`;
    const origin=state.bundle.evaluation_origin,beforeSlots=fixedBeforeSlots(),observeSlots=Number($('comparison-slots').value);
    const originReady=origin?.status==='ready'&&origin.observe_slots===(beforeSlots??observeSlots);
    const validWindow=Number.isInteger(observeSlots)&&observeSlots>=1&&observeSlots<=20000;
    const button=$('deployment-start');button.disabled=!originReady||!validWindow||deploymentBusy||running||state.intentBusy||!count||!deploymentState.enabled||location.protocol==='file:'||SmoGraph.hasBlockingConstraints(state.intent||{});
    button.textContent=deploymentBusy?'提交中…':running?'全部组合评估中…':`评估全部 ${count} 个组合`;button.setAttribute('aria-busy',String(deploymentBusy));
    $('deployment-stop').hidden=!batch||!running;$('deployment-stop').disabled=deploymentBusy||!!batch?.pause_requested;
    $('deployment-stop').textContent=batch?.pause_requested?'正在暂停并保存…':'暂停批次';
    $('deployment-resume').hidden=!batch||!['paused','complete_with_failures'].includes(batch.phase);$('deployment-resume').disabled=deploymentBusy;
    $('deployment-reconnect').hidden=!jobs.some(j=>j.phase==='connection_lost')&&job?.phase!=='connection_lost';$('deployment-reconnect').disabled=deploymentBusy;
    $('evaluation-origin').disabled=deploymentBusy||running||!deploymentState.enabled||location.protocol==='file:';
    $('evaluation-origin-status').textContent=originReady?'起点已对齐'+(beforeSlots==null?'':` · 部署前固定 ${beforeSlots} slots`):deploymentState.enabled?'起点未对齐':'此场景暂未配置部署评估。';
    $('deployment-result').hidden=true;
    $('deployment-export-json').hidden=!batch;
    if(batch)$('deployment-export-json').href='./api/deployment/batch/results?batch_id='+encodeURIComponent(batch.id)+'&format=json';
    const windowSlots=batch?.observe_slots||Number($('comparison-slots').value)||500;
    const counts=batch?.counts;
    $('deployment-status').textContent=deploymentError||(!deploymentState.enabled?'未配置仿真设备。':batch?
      `${deploymentPhases[batch.phase]||batch.phase} · ${counts.complete}/${batch.total}${counts.failed+counts.cancelled?' · 失败 '+(counts.failed+counts.cancelled):''}${batch.error?' · '+batch.error:''}`:
      `${deploymentState.target} · 每组独立复位、同种子 · ${evaluationWindows(windowSlots)}。`);
    const originAudit=batch?.evaluation_origin_audit;
    $('deployment-audit').textContent='';
    if(originAudit?.status==='verified')$('deployment-audit').textContent+=` · ${originAudit.verified_runs}/${originAudit.completed_runs} 组起点已校验`;
    else if(originAudit?.status==='legacy_unaligned')$('deployment-audit').textContent+=' · 历史批次：原意图快照与实验起点未对齐，原始记录保留';
    if(batch?.controlled_comparison?.status==='ready')$('deployment-audit').textContent+=` · ${batch.controlled_comparison.reference.matched_completed_runs} 组部署前基准一致`;
    else if(batch?.controlled_comparison?.reason==='evaluation_baseline_mismatch')$('deployment-audit').textContent+=' · 部署前基准不一致，不宜直接横向比较';
    $('deployment-outcomes').hidden=!batch;
    if(batch){
      const outcomes=batchOutcomeCounts(batch);
      const labels={satisfied:'达标',unmet:'未达标',insufficient:'数据不足',pending:'待评估',running:'评估中',failed:'运行失败',cancelled:'已停止',invalid:'约束无效'};
      $('deployment-outcomes').innerHTML=Object.entries(labels).filter(([key])=>['satisfied','unmet','insufficient'].includes(key)||outcomes[key]>0).map(([key,label])=>`<span class="outcome-${key}">${label}<strong>${outcomes[key]}</strong>组</span>`).join('')+(batch.members.some(m=>m.comparison?.result_evaluation?.no_regression===false)?`<span class="outcome-unmet">目标退步<strong>${batch.members.filter(m=>m.comparison?.result_evaluation?.no_regression===false).length}</strong>组</span>`:'')+'';
    }
    $('deployment-workers').innerHTML=(deploymentState.workers||[]).map(worker=>{
      const current=jobs.find(j=>j.worker_id===worker.id);
      return `<div role="listitem" class="deployment-worker${current?' is-busy':''}"><strong>${esc(worker.label)}</strong><span>${current?esc(current.selected_names.join(' + ')+' · '+(deploymentPhases[current.phase]||current.phase)+(current.slot!=null?' · '+current.slot+'/'+current.total_slots+' slots':'')):'空闲'}</span></div>`;
    }).join('');
    $('deployment-progress').hidden=!batch;
    if(batch){$('deployment-progress').max=batch.total;$('deployment-progress').value=counts.complete+counts.failed+counts.cancelled;}
    if(running&&batch)$('comparison-slots').value=String(batch.observe_slots);
    $('comparison-slots').disabled=running;
    $('comparison-validated-window').disabled=running;
    $('comparison-window-duration').textContent='slots = '+windowDuration(Number($('comparison-slots').value));
  }
  async function refreshDeployment(){
    if(deploymentPolling||location.protocol==='file:')return;
    deploymentPolling=true;
    try{
      const response=await fetch('./api/deployment',{cache:'no-store'});const result=await response.json();
      if(!response.ok)throw new Error(result.error||'无法读取部署状态');
      deploymentState=result;
      if(deploymentError==='无法读取运行状态；请检查本机 SMO 服务。')deploymentError='';
      const batchKey=result.latest_batch?result.latest_batch.id+'|'+result.latest_batch.members.map(m=>m.phase+':'+m.job_id).join(','):null;
      if(state.intent&&batchKey!==lastBatchRenderKey){ensureConflictQuantification();renderConflictQuantificationStatus();renderCombinationCatalog();lastBatchRenderKey=batchKey;}renderComparison();
    }catch(e){deploymentError='无法读取运行状态；请检查本机 SMO 服务。';renderDeployment();}
    finally{deploymentPolling=false;}
  }
  async function prepareEvaluationOrigin(){
    if(deploymentBusy||deploymentRunning())return;
    stop();deploymentBusy=true;deploymentError='';renderDeployment();
    try{
      const bundle=await postIntent('./api/evaluation/origin',{observe_slots:Number($('comparison-slots').value)});
      initialize(bundle);state.index=0;
      if(state.activeIntents.length)await reviewSavedIntents(state.activeIntents.map(r=>r.id));
      render();
    }catch(e){deploymentError=e.message;}
    finally{deploymentBusy=false;renderComparison();renderDeployment();}
  }
  async function startDeployment(){
    if(deploymentBusy||deploymentRunning())return;
    stop();clearTimeout(comparisonTimer);comparisonVersion++;comparisonBusy=false;
    deploymentBusy=true;deploymentError='';renderDeployment();
    try{
      const batch=await postIntent('./api/deployment/batch/start',{active_intent_ids:state.activeIntents.map(r=>r.id),slot:frame().slot,observe_slots:Number($('comparison-slots').value),show_dashed:state.showDashed});
      deploymentState.latest_batch=batch;await refreshDeployment();
    }catch(e){deploymentError=e.message;}
    finally{deploymentBusy=false;renderComparison();if(state.intent)renderCombinationCatalog();}
  }
  async function stopDeployment(){
    if(!activeBatch()||deploymentBusy)return;
    deploymentBusy=true;deploymentError='';renderDeployment();
    try{deploymentState.latest_batch=await postIntent('./api/deployment/batch/pause',{batch_id:activeBatch().id});}
    catch(e){deploymentError=e.message;}
    finally{deploymentBusy=false;renderDeployment();}
  }
  async function resumeDeployment(){
    if(!activeBatch()||deploymentBusy)return;
    deploymentBusy=true;deploymentError='';renderDeployment();
    try{deploymentState.latest_batch=await postIntent('./api/deployment/batch/resume',{batch_id:activeBatch().id});}
    catch(e){deploymentError=e.message;}
    finally{deploymentBusy=false;renderDeployment();}
  }
  async function viewDeploymentResult(jobId,groupId){
    try{await postIntent('./api/deployment/view',{job_id:jobId});state.graphGroup=groupId;await load();if(state.activeIntents.length)await reviewSavedIntents(state.activeIntents.map(r=>r.id));state.graphGroup=groupId;render();}
    catch(e){deploymentError=e.message;renderDeployment();}
  }
  let comparisonResult=null, comparisonSession=null, comparisonBusy=false, comparisonVersion=0,
      comparisonContext=null, comparisonRequestKey=null, comparisonTimer=null, comparisonError='', comparisonSlotsDirty=false;
  const comparisonScope=()=>intentContextId()+'|'+(state.bundle.source.run_id||state.bundle.source.source_sha256||'')+'|'+state.activeIntents.map(r=>r.id).sort().join(',');
  const comparisonStorageKey=()=> 'oran.smo.intent-comparison.v1:'+comparisonScope();
  function ensureComparison() {
    if(deploymentState.enabled||managedComparison()||state.bundle.scenario_view?.manual_evaluation_enabled===false)return;
    if(!state.activeIntents.length||state.intentBusy||location.protocol==='file:'||typeof frame().slot!=='number')return;
    const context=comparisonScope(),key=context+'|'+frame().slot;
    if(context!==comparisonContext){
      comparisonContext=context;comparisonSession=null;comparisonResult=null;comparisonError='';
      try{comparisonSession=localStorage.getItem(comparisonStorageKey());}catch{}
    }
    if(key===comparisonRequestKey)return;
    comparisonRequestKey=key;clearTimeout(comparisonTimer);
    comparisonTimer=setTimeout(()=>requestComparison(false),140);
  }
  async function requestComparison(capture=false) {
    if(managedComparison()){await refreshDeployment();return;}
    if(!state.activeIntents.length||state.intentBusy)return;
    const slot=frame().slot,context=comparisonScope(),version=++comparisonVersion;
    comparisonRequestKey=context+'|'+slot;comparisonBusy=true;comparisonError='';renderComparison();
    try{
      const slots=Number($('comparison-slots').value);
      if(!Number.isInteger(slots)||slots<1||slots>20000)throw new Error('观察长度须为 1–20,000 个整数 slots。');
      const start=capture||!comparisonSession;
      const result=await postIntent(start?'./api/intent/comparison/start':'./api/intent/comparison/review',start?
        {active_intent_ids:state.activeIntents.map(r=>r.id),slot,observe_slots:slots}:{comparison_id:comparisonSession,slot});
      if(version!==comparisonVersion||comparisonScope()!==context)return;
      comparisonSession=result.id;comparisonResult={...result,reviewed_slot:slot};
      try{localStorage.setItem(comparisonStorageKey(),result.id);}catch{}
      if(capture)comparisonSlotsDirty=false;
      if(!comparisonSlotsDirty)$('comparison-slots').value=String(result.observe_slots);
    }catch(error){if(version===comparisonVersion)comparisonError=error.message;}
    finally{if(version===comparisonVersion){comparisonBusy=false;renderComparison();}}
  }
  function renderComparison() {
    const show=!!state.activeIntents.length&&state.bundle.scenario_view?.manual_evaluation_enabled!==false;$('intent-comparison').hidden=!show;if(!show)return;
    const managed=managedComparison();
    const current=managed?(managed.comparison||null):(comparisonResult?.reviewed_slot===frame().slot&&comparisonContext===comparisonScope()?comparisonResult:null);
    renderDeployment();
    $('comparison-heading').textContent='全部组合部署评估';
    $('comparison-capture').hidden=true;$('comparison-refresh').hidden=true;
    const showDetail=!!managed?.comparison;
    $('comparison-summary').hidden=!showDetail;$('comparison-body').closest('table').hidden=!showDetail;$('comparison-intents').hidden=!showDetail;
    $('intent-comparison').dataset.comparisonId=current?.id||'';
    const labels={awaiting_deployment:'尚未部署',observing:'观察中',observing_data:'等待窗口数据写入',complete:'窗口完成',complete_with_missing_kpis:'窗口完成 · 数据不足',invalid_constraints:'目标约束待处理',snapshot_before_baseline:'当前快照早于基线'};
    $('comparison-state').textContent=comparisonBusy?'读取中…':comparisonError?'待处理':(managed&&!managed.comparison?deploymentPhases[managed.phase]:labels[current?.status])||'尚未部署';
    $('comparison-capture').disabled=!!managed||state.intentBusy||comparisonBusy||location.protocol==='file:';
    $('comparison-refresh').disabled=state.intentBusy||comparisonBusy||(!managed&&!comparisonSession)||location.protocol==='file:';
    $('comparison-message').textContent=comparisonError;
    const slots=current?.observe_slots||Number($('comparison-slots').value)||500;
    const baseline=current?.baseline_window;
    const beforeSlots=Number.isInteger(baseline?.window_slots)?baseline.window_slots:baseline&&Number.isInteger(baseline.slot)&&Number.isInteger(baseline.window_start_slot)
      ? baseline.slot-baseline.window_start_slot
      : managed?managed.context?.runtime_environment?.origin_observe_slots??null:fixedBeforeSlots();
    const span=w=>w?`slots ${w.window_start_slot+1}–${w.slot}`:'待观察';
    $('comparison-note').textContent=evaluationWindows(slots,beforeSlots)+' · 仅比较意图目标 · 期望范围保持解析时的基准'+(current?` · 部署前 ${span(current.baseline_window)}`:'')+(current?.after_window?` · 部署后 ${span(current.after_window)}`:current?.application_slot!=null?` · 从动作应用 slot ${current.application_slot} 开始`:' · 等待真实执行回执');
    if(managed)$('comparison-note').textContent=managed.comparison?'当前查看：'+managed.selected_names.join(' + ')+` · 每组${evaluationWindows(slots,beforeSlots)}，目标范围保持不变。`:`每组${evaluationWindows(managed.observe_slots,beforeSlots)} · 实测结果逐组填入下方组合表。`;
    if(current?.status==='observing')$('comparison-note').textContent+=` · 尚余 ${current.remaining_slots} slots`;
    if(current?.application_evidence?.length)$('comparison-note').textContent+=' · 已应用：'+[...new Set(current.application_evidence.map(r=>find(r.app_id)?.label||r.app_id))].join('、');
    const stat=value=>value?`${value.satisfied}/${value.required} 项`:'待观察';
    const before=current?.before,after=current?.after,q=current?.quality;
    const pct=v=>v==null?'—':fmt(v*100,2)+'%';
    const outcomes={insufficient_data:'数据不足',maintained_feasibility:'前后均达标',reached_intent:'全部目标达标',violated_intent:'由达标变为未达标',closer:'更接近目标',further:'远离目标',unchanged:'差距未变'};
    $('comparison-summary').innerHTML=`<span>部署前达标<strong>${esc(stat(before))}</strong></span><span>部署后达标<strong>${esc(after?stat(after):current?.status==='awaiting_deployment'?'尚未部署':'待观察')}</strong></span><span>最大归一化差距<strong>${esc(pct(before?.worst_normalized_gap))} → ${esc(pct(after?.worst_normalized_gap))}</strong></span><span>差距缩小<strong>${esc(q?.improvement_percentage_points==null?'—':(q.improvement_percentage_points>0?'+':'')+fmt(q.improvement_percentage_points,2)+' 百分点')}</strong><small>${esc(after?outcomes[q?.state]||'':'待部署后评价')}</small></span>`;
    if(matchingBatch())$('comparison-summary').innerHTML=`<span>部署前达标<strong>${esc(stat(before))}</strong></span><span>部署后达标<strong>${esc(stat(after))}</strong></span>`;
    const evaluation=current?.result_evaluation;
    if(evaluation)$('comparison-summary').innerHTML=`<span>部署前基准分<strong>${esc(resultScoreValue(evaluation.before.score))}</strong></span><span>意图改善分<strong>${esc(resultScoreValue(evaluation.after.score))}</strong><small>${esc(SmoGraph.evaluationPresentation(evaluation).label)}</small></span><span>最大剩余违背<strong>${esc(fmt(evaluation.violation?.after?.max,2))} 分</strong><small>仅统计意图涉及的 KPI</small></span>`;
    const states={satisfied:'达标',below_range:'未满足下限约束',above_range:'未满足上限约束',outside_range:'超出上下限',unavailable:'数据不足',empty_range:'目标区间为空',legacy_kpi:'旧 J 口径需重解析'};
    const reasons={exact_window_data_unavailable:'缺少完整等长窗口',exact_window_source_unavailable:'缺少逐时隙数据',missing_or_reset_counter:'计数器缺失或重置',incomplete_task_event_log:'任务事件记录不完整',unresolved_mature_tasks:'存在未决的成熟任务',no_mature_task_samples:'没有成熟任务样本',missing_task_profile:'任务观察时限不可用',unsupported_window_metric:'此指标尚无窗口统计',no_valid_completed_packet_samples:'没有有效的已完成分组',no_valid_offered_packet_samples:'没有有效的到达分组'};
    const reading=(row,key)=>{
      if(!row)return '<code>—</code><small>尚无部署后观测</small>';
      const m=row.measurement||{};
      return `<code>${esc(metricValue(key,row.value))}</code><small class="${row.status==='satisfied'?'satisfied':['outside_range','above_range','below_range'].includes(row.status)?'unmet':''}">${esc(states[row.status]||row.status)}</small>`+
        (m.reason?`<small>${esc(reasons[m.reason]||m.reason)}</small>`:'')+
        (m.minimum!=null?`<small>窗口范围 ${esc(metricValue(key,m.minimum))}–${esc(metricValue(key,m.maximum))}</small>`:'')+
        (m.aggregation==='matched_mature_birth_cohort'?`<small>成熟任务 ${m.generated} · 完成 ${m.sample_count} · 失败 ${m.failed} · 未决 ${m.pending}</small><small>新任务到达窗 ${esc(fmt(m.cohort_exposure_ms,1))} ms · 观察时限 ${esc(fmt(m.deadline_guard_ms,1))} ms</small>`:m.sample_count!=null?`<small>${m.aggregation==='completion_weighted_interval_mean'?'完成分组':'窗口事件'} ${esc(fmt(m.sample_count,0))}</small>`:'');
    };
    const gap=row=>!row?'—':`${pct(row.normalized_gap)}${row.strict_boundary_violation?'（严格边界未满足）':''}`;
    const basis={before_window_magnitude:'部署前窗口值',nonzero_requested_bound:'非零目标边界',measurement_resolution:'测量分辨率',normalization_unavailable:'无有效尺度'};
    $('comparison-body').innerHTML=(current?.result_evaluation?.metrics||[]).map(m=>{
      const row=current.rows.find(r=>r.kpi===m.kpi),unit=metricUnit(row.kpi,row.unit);
      return `<tr><th scope="row">${esc(chineseKpis[row.kpi]||row.label)}<small>${esc(unit)}</small></th><td><code>${esc(row.empty?'∅':rangeText(row))}</code></td><td>${reading(row.before,row.kpi)}</td><td>${reading(row.after,row.kpi)}</td><td><code>改善 ${esc(resultScoreValue(m.improvement_score))}${m.score_unit==='percentage_points'?' 个百分点':'%'}</code><small>${m.regression?'目标退步，不可用其他收益抵消':'相对部署前未退步'}</small><small>剩余违背 ${esc(metricValue(row.kpi,m.after.distance))} ${esc(unit)} · ${esc(fmt(m.after.normalized_violation,2))} 分</small>${m.target_progress_percent==null?'':`<small>区间差距缩减 ${esc(fmt(m.target_progress_percent,2))}%</small>`}${row.after?.measurement?.constraint_uses_extrema?'<small>改善分按窗口均值；区间约束按整窗最小/最大值判定</small>':''}</td></tr>`;
    }).join('');
    $('comparison-intents').innerHTML=(current?.intents||[]).map(row=>`<div class="comparison-intent-row"><strong>${esc(row.text)}</strong><p>部署前 ${esc(stat(row.before))} → 部署后 ${esc(stat(row.after))}${row.after?' · '+esc(outcomes[row.quality.state]||''):''}</p></div>`).join('');
  }
  let decisionPreview=null, previewBusy=false, previewVersion=0, previewRequestKey=null, previewResultKey=null, previewTimer=null;
  const currentPreview = () => previewResultKey===SmoGraph.decisionPreviewContextKey(state.bundle,frame())&&
    (decisionPreview?.requested_slot??decisionPreview?.slot)===frame().slot?decisionPreview:null;
  const appPreview = id => currentPreview()?.apps?.find(a=>a.app_id===id);
  const previewLabels={ue_offload_fraction:'MEC offload',ue_drx_configs:'DRX cycle/on/inactivity',scheduler_mode:'Downlink scheduler',ue_scheduler_weights:'Scheduling weight',cell_sleep_mode:'BS sleep mode'};
  function previewValue(value,field) {
    if(value===null||value===undefined)return '未记录';
    if(field==='ue_offload_fraction')return fmt(value*100,1)+'%';
    if(field==='ue_drx_configs')return value.enabled===false?'Disabled':`${fmt(value.cycle_ms)}/${fmt(value.on_duration_ms)}/${fmt(value.inactivity_ms)} ms`;
    if(field==='scheduler_mode')return {round_robin:'RR',latency_aware:'Latency-aware',qos_deadline:'QoS-deadline'}[value]||String(value);
    return typeof value==='object'?JSON.stringify(value):String(value).replaceAll('_',' ');
  }
  function previewScope(row) {
    const ues=[...new Set(row.scopes.filter(s=>s.ue_id).map(s=>s.ue_id.replace('ue_','')))];
    const cells=[...new Set(row.scopes.map(s=>s.cell_id?.replace('cell_','')))];
    return ues.length?'UE '+ues.join(','):'Cell '+cells.join(',');
  }
  function previewStatus(app) {
    if(!app)return previewBusy?'读取拟动作…':location.protocol==='file:'?'拟动作需连接 SMO':'此快照未记录拟动作';
    if(app.status==='no_action')return '本轮无新动作';
    if(app.status==='invalid_proposal')return '拟动作未通过控制契约';
    if(app.status!=='proposed')return '输入不足，无法预演';
    return `拟动作 · ${app.mode==='state_clone'?'状态副本':'首次启用'}`;
  }
  function ensureDecisionPreview() {
    const slot=frame().slot;
    if(location.protocol==='file:'||typeof slot!=='number')return;
    const bundle=state.bundle,key=SmoGraph.decisionPreviewContextKey(bundle,frame());
    const expectedSource=bundle.source.run_id||bundle.source.source_sha256||bundle.source.name;
    if(previewRequestKey===key)return;
    previewRequestKey=key;previewBusy=true;
    const version=++previewVersion;clearTimeout(previewTimer);
    previewTimer=setTimeout(async()=>{
      try{
        const result=await postIntent('./api/xapps/preview',{slot});
        if(result.source_id!==expectedSource)throw Error('拟动作来源已变化，请刷新网络快照。');
        if(bundle.graph_sha256&&result.graph_sha256!==bundle.graph_sha256)throw Error('图谱版本已变化，请刷新网络快照。');
        if(version===previewVersion&&key===SmoGraph.decisionPreviewContextKey(state.bundle,frame())){
          if(decisionPreview&&JSON.stringify(decisionPreview)!==JSON.stringify(result))conflictQuantificationRequestKey=null;
          decisionPreview=result;previewResultKey=key;
        }
      }
      catch(e){if(version===previewVersion&&key===SmoGraph.decisionPreviewContextKey(state.bundle,frame())){
        decisionPreview={slot,reason:e.message,apps:[]};previewResultKey=key;
      }}
      finally{if(version===previewVersion){previewBusy=false;render();}}
    },120);
  }
  function renderDecisionPreview() {
    const offline=location.protocol==='file:',data=currentPreview();
    $('preview-run').disabled=previewBusy||offline||typeof frame().slot!=='number';
    $('preview-run').textContent=previewBusy?'读取中…':'更新拟动作';
    $('preview-status').textContent=previewBusy?'读取拟动作…':offline?'拟动作需连接本地 SMO':data?.apps?.length?`拟动作 ${time(data.slot)} · 未下发${data.requested_slot!==undefined&&data.requested_slot!==data.slot?' · 最近已采集快照':''}`:data?'此快照未记录拟动作':'';
    $('preview-status').title=data?.reason||data?.snapshot_note||'当前值来自各 xApp 同一决策快照的已交付报告。';
  }
  function exportCombinationCatalog() {
    if(matchingBatch()){const a=document.createElement('a');a.href='./api/deployment/batch/results?batch_id='+encodeURIComponent(matchingBatch().id)+'&format=csv';a.download='all-combination-results.csv';a.click();return;}
    const catalog=state.combinationCatalog,groups=[...(catalog?.groups||[]),...(catalog?.reserve_groups||[]),...(catalog?.excluded_groups||[])];if(!groups.length)return;
    const reasons={opposed_direct_control_actions:'同一参数控制动作方向相反',weak_influence_coverage:'部分目标 KPI 影响偏弱',low_marginal_contribution:'增量影响较小',meaningful_coverage_and_policy_diversity:'影响覆盖与策略保留',unquantified_evidence_preserved:'待补充校准，保留',empty_primary_fallback:'回退完整枚举',unconfirmed_capability_direction:'尚未确认覆盖全部目标方向',unsupported_capability_direction:'能力方向未覆盖全部目标'};
    const rows=[['xApp组合','候选池','初选原因','图上冲突风险','反向直接冲突证据']];
    for(const group of groups){
      const apps=group.app_ids.map(id=>intentCandidates().find(c=>c.app_id===id)).sort((a,b)=>a.label.localeCompare(b.label));
      rows.push([apps.map(c=>c.label).join(' + '),({reserve:'备选',excluded:'已剔除'})[group.initial_selection?.pool]||'主候选',reasons[group.initial_selection?.reason]||'',groupConflictLines(group).join('；'),group.action_exclusion?JSON.stringify(group.action_exclusion):'']);
    }
    const csv='\ufeff'+rows.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='xapp-combination-conflicts.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function postIntent(path,data) {
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result=await response.json();if(!response.ok)throw new Error(result.error || `HTTP ${response.status}`);return result;
  }
  const presetStorageKey='oran.smo.intent-presets.v1';
  let presetRevision=null,presetSyncBusy=false;
  async function syncPresetStore(){
    if(location.protocol==='file:')return;
    presetSyncBusy=true;syncPresetControls();
    try{
      const response=await fetch('./api/intent/presets',{cache:'no-store'});
      if(!response.ok)throw new Error('无法读取服务端预设');
      let saved=await response.json();
      if(saved.revision===0&&(Object.keys(state.presetEdits).length||state.deletedPresetIds.length)){
        saved=await postIntent('./api/intent/presets',{revision:0,edits:state.presetEdits,deleted_ids:state.deletedPresetIds});
      }
      const selected=$('intent-case').value,draft=$('intent-text').value;
      presetRevision=saved.revision;state.presetEdits=saved.edits;state.deletedPresetIds=saved.deleted_ids;
      state.intentPresets=SmoGraph.customIntentCases(saved.edits,saved.deleted_ids);renderPresetOptions(selected);$('intent-text').value=draft;
      try{localStorage.setItem(presetStorageKey,JSON.stringify({schema:'oran.intent-presets.v1',edits:saved.edits,deleted_ids:saved.deleted_ids}));}catch{}
    }catch(error){intentMessage('预设暂未同步到服务端：'+error.message,true);}
    finally{presetSyncBusy=false;syncPresetControls();}
  }
  async function persistPresetData(edits,deletedIds) {
    if(location.protocol!=='file:'){
      if(presetRevision===null)throw new Error('预设尚未同步，请刷新页面后再保存。');
      const saved=await postIntent('./api/intent/presets',{revision:presetRevision,edits,deleted_ids:deletedIds});presetRevision=saved.revision;
    }
    try{localStorage.setItem(presetStorageKey,JSON.stringify({schema:'oran.intent-presets.v1',edits,deleted_ids:deletedIds}));}
    catch{if(location.protocol==='file:')throw new Error('浏览器未能保存预设。请检查浏览器存储权限后重试。');}
  }
  function syncPresetControls() {
    $('intent-preset-edit').disabled=presetSyncBusy||state.intentBusy||!!state.editingPreset||!state.intentPresets.some(p=>p.id===$('intent-case').value);
    $('intent-preset-edit').setAttribute('aria-expanded',String(!!state.editingPreset));
    for(const id of ['intent-preset-save','intent-preset-delete','intent-preset-reset','intent-preset-cancel'])$(id).disabled=presetSyncBusy;
    $('intent-preset-edit-text').readOnly=presetSyncBusy;$('intent-preset-edit-tag').readOnly=presetSyncBusy;
    $('intent-case').disabled=presetSyncBusy||state.intentBusy||!!state.editingPreset;
    $('intent-text').readOnly=!!state.editingPreset;
    $('intent-preset-trash').hidden=!state.deletedPresetIds.length;
    $('intent-preset-deleted-note').textContent='已删除 '+state.deletedPresetIds.length+' 个预设';
    $('intent-preset-undo-delete').disabled=presetSyncBusy||state.intentBusy||!!state.editingPreset;
  }
  function presetEditorMessage(message,problem=false) {
    const el=$('intent-preset-edit-message');el.textContent=message;el.hidden=!message;
    el.classList.toggle('problem',problem);
    $('intent-preset-edit-text').setAttribute('aria-invalid',String(problem));
  }
  function renderPresetOptions(selected='') {
    const select=$('intent-case');select.replaceChildren();
    const manual=document.createElement('option');manual.value='';manual.textContent='手动输入';select.append(manual);
    for(const preset of state.intentPresets){const option=document.createElement('option');option.value=preset.id;option.textContent=(preset.tag?preset.tag+' · ':'')+preset.text;select.append(option);}
    select.value=selected;syncPresetControls();
  }
  function closePresetEditor(focus=true) {
    state.editingPreset=null;$('intent-preset-editor').hidden=true;presetEditorMessage('');
    syncPresetControls();intentStatus();
    if(focus)$('intent-preset-edit').focus();
  }
  async function savePresetChange(reset=false) {
    const id=state.editingPreset;if(!id||presetSyncBusy)return;
    presetSyncBusy=true;syncPresetControls();
    try {
      const next={...state.presetEdits};
      if(reset)delete next[id];
      else {
        const edit=SmoGraph.validatePresetEdit({text:$('intent-preset-edit-text').value,tag:$('intent-preset-edit-tag').value});
        const original=SmoGraph.intentCases.find(p=>p.id===id);
        if(edit.text===original.text&&edit.tag===(original.tag||''))delete next[id];else next[id]=edit;
      }
      await persistPresetData(next,state.deletedPresetIds);
      state.presetEdits=next;state.intentPresets=SmoGraph.customIntentCases(next,state.deletedPresetIds);
      closePresetEditor(false);renderPresetOptions(id);
      $('intent-text').value=state.intentPresets.find(p=>p.id===id).text;
      clearIntentReview();intentMessage(reset?'已恢复原始预设。':'预设已保存，点击“分析”后用于新的意图。');
      $('intent-preset-edit').focus();
    }catch(error){presetEditorMessage(error.message,true);$('intent-preset-edit-text').focus();}
    finally{presetSyncBusy=false;syncPresetControls();}
  }
  async function deletePreset() {
    const id=state.editingPreset;if(!id||presetSyncBusy)return;
    presetSyncBusy=true;syncPresetControls();
    try {
      const deleted=[...state.deletedPresetIds.filter(value=>value!==id),id];
      await persistPresetData(state.presetEdits,deleted);
      state.deletedPresetIds=deleted;state.intentPresets=SmoGraph.customIntentCases(state.presetEdits,deleted);
      closePresetEditor(false);renderPresetOptions();clearIntentReview();
      intentMessage('预设已删除；已生效的意图保持不变。');$('intent-preset-undo-delete').focus();
    }catch(error){presetEditorMessage(error.message,true);}
    finally{presetSyncBusy=false;syncPresetControls();}
  }
  async function undoPresetDeletion() {
    if(!state.deletedPresetIds.length||presetSyncBusy)return;
    presetSyncBusy=true;syncPresetControls();
    try {
      const deleted=state.deletedPresetIds.slice(0,-1),selected=$('intent-case').value;
      await persistPresetData(state.presetEdits,deleted);
      state.deletedPresetIds=deleted;state.intentPresets=SmoGraph.customIntentCases(state.presetEdits,deleted);
      renderPresetOptions(selected);intentMessage('已恢复最近删除的预设。');$('intent-case').focus();
    }catch(error){intentMessage(error.message,true);}
    finally{presetSyncBusy=false;syncPresetControls();}
  }
  function setupIntent() {
    try{const saved=JSON.parse(localStorage.getItem(presetStorageKey)||'null');if(saved?.schema==='oran.intent-presets.v1'){
      if(saved.edits&&typeof saved.edits==='object'&&!Array.isArray(saved.edits))state.presetEdits=saved.edits;
      if(Array.isArray(saved.deleted_ids))state.deletedPresetIds=[...new Set(saved.deleted_ids.filter(id=>SmoGraph.intentCases.some(p=>p.id===id)))];
    }}catch{}
    if(location.protocol==='file:'&&!Object.keys(state.presetEdits).length&&window.SMO_BOOTSTRAP?.preset_data){state.presetEdits=window.SMO_BOOTSTRAP.preset_data.edits;state.deletedPresetIds=window.SMO_BOOTSTRAP.preset_data.deleted_ids;}
    state.intentPresets=SmoGraph.customIntentCases(state.presetEdits,state.deletedPresetIds);renderPresetOptions();
    $('intent-preset-edit').addEventListener('click',()=>{
      const preset=state.intentPresets.find(p=>p.id===$('intent-case').value);if(!preset||state.intentBusy)return;
      state.editingPreset=preset.id;$('intent-preset-edit-text').value=preset.text;
      $('intent-preset-edit-tag').value=preset.tag||'';
      $('intent-preset-reset').hidden=!preset.customized;presetEditorMessage('');
      $('intent-preset-editor').hidden=false;syncPresetControls();intentStatus();$('intent-preset-edit-text').focus();
    });
    $('intent-preset-editor').addEventListener('submit',ev=>{ev.preventDefault();savePresetChange();});
    $('intent-preset-cancel').addEventListener('click',()=>closePresetEditor());
    $('intent-preset-reset').addEventListener('click',()=>savePresetChange(true));
    $('intent-preset-delete').addEventListener('click',deletePreset);
    $('intent-preset-undo-delete').addEventListener('click',undoPresetDeletion);
    $('intent-case').addEventListener('change',()=>{
      const preset=state.intentPresets.find(c=>c.id===$('intent-case').value);
      if(preset)$('intent-text').value=preset.text;
      clearIntentReview();syncPresetControls();
    });
    $('intent-text').addEventListener('input',()=>{
      $('intent-case').value=state.intentPresets.find(c=>c.text===$('intent-text').value.trim())?.id||'';
      clearIntentReview();syncPresetControls();
    });
    $('intent-form').addEventListener('submit',async ev=>{
      ev.preventDefault();if(!state.bundle)return intentMessage('请等待图谱服务连接完成。',true);const text=$('intent-text').value.trim();if(!text)return intentMessage('请输入目标 KPI 及期望方向或范围。',true);
      if(state.editingPreset)return;
      if(state.intentPresets.find(c=>c.id===$('intent-case').value)?.needsThreshold){$('intent-text').focus();return intentMessage('分析此意图前，请先填写经过本网络验证的基站功率限值。',true);}
      if(!state.bundle.intent_service?.configured){$('llm-settings').open=true;$('llm-key').focus();return intentMessage('请先连接 DeepSeek，再解析自然语言意图。',true);}
      stop();const version=++state.intentVersion;state.intentBusy=true;state.intent=null;state.graphGroup=null;state.conflictView=null;state.combinationCatalog=null;state.selected=null;state.focus='';$('app-focus').value='';state.intentFrame=structuredClone(frame());
      intentMessage('正在分析…');$('intent-submit').textContent='正在分析…';render();
      try {
        const result=await postIntent('./api/intent',{text,slot:state.intentFrame.slot ?? null,active_intent_ids:state.activeIntents.map(r=>r.id)});
        if(version!==state.intentVersion)return;
        if($('intent-text').value.trim()!==text){intentMessage('输入已修改，请重新分析新的意图。',true);return;}
        applyIntentResult(result);
        fetch('./api/intent/status',{cache:'no-store'}).then(r=>r.ok?r.json():null).then(s=>{if(s){state.bundle.intent_service={...state.bundle.intent_service,...s};intentStatus();}}).catch(()=>{});
        const feedback=SmoGraph.intentFeedback(result,{submitted:true});intentMessage(feedback.text,feedback.problem);
      } catch(e){if(version===state.intentVersion)intentMessage(e.message,true);}
      finally {if(version===state.intentVersion){state.intentBusy=false;$('intent-submit').textContent='分析';render();}}
    });
    $('intent-clear').addEventListener('click',clearAllIntents);
    $('intent-refresh').addEventListener('click',()=>reviewSavedIntents(state.activeIntents.map(r=>r.id)));
    $('preview-run').addEventListener('click',()=>{stop();previewRequestKey=null;render();});
    $('llm-form').addEventListener('submit',async ev=>{
      ev.preventDefault();$('llm-connect').disabled=true;
      try {state.bundle.intent_service={...state.bundle.intent_service,...await postIntent('./api/intent/session-key',{api_key:$('llm-key').value,model:$('llm-model').value.trim()})};$('llm-key').value='';$('llm-settings').open=false;intentMessage('');$('intent-text').focus();}
      catch(e){$('llm-key').value='';intentMessage(e.message,true);}
      finally {$('llm-connect').disabled=false;intentStatus();}
    });
    $('llm-disconnect').addEventListener('click',async()=>{try{state.bundle.intent_service={...state.bundle.intent_service,...await postIntent('./api/intent/clear-key',{})};intentStatus();intentMessage('会话密钥已清除。');}catch(e){intentMessage(e.message,true);}});
  }

  function svg(tag, attrs={}, content) {
    const e = document.createElementNS(NS, tag);
    for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
    if (content !== undefined) e.textContent = content;
    return e;
  }
  function activate(id) {
    state.graphGroup=null;
    state.selected = id;
    const n = find(id);
    if ([n?.id,n?.source,n?.target].some(k => {const target=find(k);return target?.kind === 'kpi' && !target.primary;})) state.all=true;
    if (n?.kind === 'xapp') { state.focus = n.implementation_id; $('app-focus').value = state.focus; }
    else { state.focus = ''; $('app-focus').value = ''; }
    render();
  }
  function connection(id, text, detail='') {
    return `<button class="connection-button" data-select="${esc(id)}">${esc(text)}<span>${esc(detail)}</span></button>`;
  }
  function entityTable(values, unit='') {
    if (!Object.keys(values || {}).length) return '<p>No executed reading is available.</p>';
    return `<table><thead><tr><th>Entity</th><th>Value${unit ? ' / '+esc(unit) : ''}</th></tr></thead><tbody>` + Object.entries(values).map(([k,v]) => `<tr><td>${esc(k.toUpperCase().replace('_',' '))}</td><td>${esc(typeof v === 'number' ? fmt(v,4) : pretty(v))}</td></tr>`).join('') + '</tbody></table>';
  }
  function effectEntityTable(s) {
    return '<table class="effect-table"><thead><tr><th>UE</th><th>Estimate</th><th>95% interval</th></tr></thead><tbody>'+Object.entries(s.entity_estimates).map(([uid,v]) => {
      const ci=s.entity_uncertainty?.[uid]?.confidence_interval;
      return `<tr><td>${esc(uid.replace('ue_',''))}</td><td>${esc(signedEffect(v))}</td><td>${ci ? esc(effectNumber(ci[0])+' … '+effectNumber(ci[1])) : '—'}</td></tr>`;
    }).join('')+'</tbody></table>';
  }
  function populationParameters(n,s) {
    const values=Object.values(s.by_entity || {});
    if (!values.length) return '<p>No executed reading is available.</p>';
    if (n.key === 'ue_drx_configs') {
      const enabled=values.filter(v=>v.enabled), profiles=new Map();
      for (const v of enabled) {const key=`Cycle ${v.cycle_ms} ms · On ${v.on_duration_ms} ms · Inactivity ${v.inactivity_ms} ms`;profiles.set(key,(profiles.get(key)||0)+1);}
      return line('DRX enabled',fmt(enabled.length/values.length*100,1)+'%')+[...profiles].map(([k,count])=>line(k,fmt(count/values.length*100,1)+'%')).join('');
    }
    if (values.every(v=>typeof v === 'number')) {
      const mean=values.reduce((a,b)=>a+b,0)/values.length;
      const show=v=>n.key==='ue_offload_fraction' ? fmt(v*100,1)+'%' : fmt(v,3);
      return line('Mean',show(mean))+line('Range',show(Math.min(...values))+' – '+show(Math.max(...values)));
    }
    const counts=new Map();for(const v of values)counts.set(pretty(v),(counts.get(pretty(v))||0)+1);
    return [...counts].map(([v,count])=>line(v,fmt(count/values.length*100,1)+'%')).join('');
  }
  function rangeCurve(s,kpiUnit) {
    const points=s.grid_points || [],valid=points.filter(p=>typeof p.mean==='number');
    if (!valid.length) return '';
    const [xmin,xmax]=s.sampled_domain,[lo,hi]=s.fit_domain;
    const predict=x=>s.intercept_at_reference+s.estimate*(x-s.reference_parameter);
    const ys=[...valid.flatMap(p=>p.by_seed.filter(v=>typeof v.value==='number').map(v=>v.value)),predict(lo),predict(hi)];
    let ymin=Math.min(...ys),ymax=Math.max(...ys),pad=(ymax-ymin)*.12 || Math.max(1,Math.abs(ymax)*.1);
    ymin=Math.min(...ys)>=0 ? Math.max(0,ymin-pad) : ymin-pad;ymax+=pad;
    const px=x=>46+(x-xmin)/(xmax-xmin)*248,py=y=>139-(y-ymin)/(ymax-ymin)*111;
    const fx=x=>s.x_unit==='fraction' ? fmt(x*100,0)+'%' : fmt(x,2);
    let svg=`<svg class="range-plot" viewBox="0 0 320 179" role="img" aria-label="Measured population KPI and first-order range fit"><line x1="46" y1="28" x2="46" y2="139" class="curve-axis"/><line x1="46" y1="139" x2="294" y2="139" class="curve-axis"/>`;
    for (const y of[ymin,ymax]) svg+=`<text x="40" y="${py(y)+3}" text-anchor="end">${esc(effectNumber(y))}</text>`;
    for (const x of[xmin,(xmin+xmax)/2,xmax])svg+=`<text x="${px(x)}" y="155" text-anchor="middle">${esc(fx(x))}</text>`;
    svg+=`<path d="M${px(lo)},${py(predict(lo))} L${px(hi)},${py(predict(hi))}" class="curve-fit"/>`;
    for(const p of valid){
      for(const r of p.by_seed.filter(v=>typeof v.value==='number'))svg+=`<circle cx="${px(p.x)}" cy="${py(r.value)}" r="2" class="curve-seed"/>`;
      svg+=`<circle cx="${px(p.x)}" cy="${py(p.mean)}" r="3.2" class="curve-point"><title>${esc(fx(p.x)+': '+effectNumber(p.mean)+' '+kpiUnit)}</title></circle>`;
    }
    const axis=s.x_unit==='fraction'?'Offloading probability':s.x_unit==='ms'?'Inactivity (ms)':'Relative priority';
    svg+=`<text x="46" y="15">${esc(kpiUnit+(s.scoring_window_ms?' · '+fmt(s.scoring_window_ms,0)+' ms window':''))}</text><text x="294" y="174" text-anchor="end">${esc(axis)}</text></svg>`;
    return svg+'<p class="curve-caption">Seeds · mean · fit</p>';
  }
  function valueText(n) {
    const s = nodeState(n);
    if (population() && n.kind === 'rcp') {
      const values=Object.values(s.by_entity || {});
      if (n.key === 'ue_drx_configs') {
        const cycles=[...new Set(values.filter(v=>v.enabled).map(v=>v.cycle_ms))];
        return cycles.length === 1 ? `${cycles[0]} ms cycle` : cycles.length ? 'Mixed DRX profiles' : values.length ? 'DRX off' : '—';
      }
      if (values.length && values.every(v=>typeof v === 'number')) {
        const lo=Math.min(...values),hi=Math.max(...values),percent=n.key==='ue_offload_fraction';
        const show=v=>percent ? fmt(v*100,1)+'%' : fmt(v,3);
        return lo === hi ? show(lo) : show(lo)+' – '+show(hi);
      }
      if (s.value == null && values.length) return 'Mixed policies';
    }
    if (n.kind === 'xapp') {
      if (!Object.keys(s).length) return 'No run attached';
      if (!s.enabled) return 'Disabled';
      const p = [...(s.current_proposals || [])].sort((a,b) => (b.slot_id || 0)-(a.slot_id || 0))[0];
      return p ? proposalTitle(p) : 'Enabled · awaiting decision';
    }
    if (s.value !== null && s.value !== undefined) {
      if (typeof s.value === 'number') return metricValue(n.key,s.value,3);
      if (typeof s.value === 'string') return s.value;
      return s.value.enabled === true ? `${s.value.cycle_ms} ms cycle` : s.value.enabled === false ? 'DRX off' : 'Configured';
    }
    const count = Object.keys(s.by_entity || {}).length;
    return count ? `${count} entity values` : '—';
  }
  function proposalTitle(p) {
    if(p.proposed_drx)return 'DRX '+[p.proposed_drx.cycle_ms,p.proposed_drx.on_duration_ms,p.proposed_drx.inactivity_ms].map(v=>fmt(v,1)).join('/')+' ms · offload '+fmt(p.proposed_offload_fraction*100,1)+'%';
    if (typeof p.proposed_offload_fraction === 'number') return `Offload → ${fmt(p.proposed_offload_fraction)}`;
    if (typeof p.p === 'number') return `Offload → ${fmt(p.p)}`;
    if (p.action_name) return p.action_name.replaceAll('_',' ');
    if (p.scheduler_mode) return `${p.scheduler_mode.replaceAll('_',' ')} · w ${p.ue_scheduler_weight}`;
    return p.action !== undefined ? `Proposed action ${p.action}` : 'Decision recorded';
  }

  function intentPaths() {
    if(!state.intent)return {main:new Set(),side:new Set(),apps:new Set()};
    const apps=new Set(viewedGroup()?.app_ids||intentCandidates().map(c=>c.app_id));
    const targets=new Set(state.intent.ranges.map(r=>'kpi:'+r.kpi));
    const controls=graph().edges.filter(e=>e.kind==='controls'&&apps.has(e.source)&&e.deployment!=='native_locked');
    const parameters=new Set(controls.map(e=>e.target));
    const main=new Set([...controls.map(e=>e.id),...graph().edges.filter(e=>e.kind==='influences'&&parameters.has(e.source)&&targets.has(e.target)).map(e=>e.id)]);
    const side=new Set(displayedConflicts().flatMap(c=>c.edge_ids||[]));
    return {main,side,apps};
  }
  function relevant(nodes, edges) {
    if (state.focus) {
      const params = new Set(edges.filter(e => e.source === 'xapp:'+state.focus).map(e => e.target));
      const selectedEdges = edges.filter(e => e.supporting_xapps?.includes(state.focus) || e.sensitivity && params.has(e.source));
      return {edges:new Set(selectedEdges.map(e => e.id)), nodes:new Set(selectedEdges.flatMap(e => [e.source,e.target]))};
    }
    const item = find(state.selected);
    if (!item) {
      if(!state.intent)return null;
      const {main,side,apps}=intentPaths(),selected=edges.filter(e=>main.has(e.id)||side.has(e.id));
      return {edges:new Set(selected.map(e=>e.id)),nodes:new Set([...apps,...selected.flatMap(e=>[e.source,e.target])]),intent:true,main,side};
    }
    if (item.source) return {edges:new Set([item.id]),nodes:new Set([item.source,item.target])};
    const near = edges.filter(e => e.source === item.id || e.target === item.id);
    const ids = new Set([item.id,...near.flatMap(e => [e.source,e.target])]);
    if (item.kind === 'kpi') {
      for (const e of edges) if (ids.has(e.target)) near.push(e);
    }
    return {edges:new Set(near.map(e => e.id)),nodes:new Set([item.id,...near.flatMap(e => [e.source,e.target])])};
  }

  function renderGraph() {
    const canvas = $('graph'); canvas.replaceChildren();
    const requested=new Set(state.intent?.ranges.filter(r=>!invalidatedKpi(r.kpi)).map(r=>'kpi:'+r.kpi) || []);
    const nodes = graph().nodes.filter(n => n.kind !== 'kpi' || state.all || n.primary || requested.has(n.id));
    const ids = new Set(nodes.map(n => n.id));
    const paths=intentPaths(),intentEdges=new Set([...paths.main,...paths.side]);
    const conflicts=displayedConflicts();
    const conflictType=id=>['direct','indirect','implicit'].find(type=>conflicts.some(c=>c.type===type&&(c.edge_ids||[]).includes(id)))||null;
    const invalidGains=new Set(state.bundle.gain_audit?.invalid_edge_ids||[]);
    const structuralOnly=e=>uncalibratedEnvironment()&&e.kind==='influences'
      && find(e.source)?.kind==='rcp'&&find(e.target)?.kind==='kpi'
      && (e.sensitivities==null||Array.isArray(e.sensitivities)&&e.sensitivities.length===0)&&e.sensitivity==null;
    const edges = graph().edges.filter(e => ids.has(e.source) && ids.has(e.target) && (!invalidGains.has(e.id)||structuralOnly(e)) && (!e.intent_only || intentEdges.has(e.id)) && (state.showDashed || !edgePresentation(e).dashed));
    const groups = ['xapp','rcp','kpi'].map(kind => nodes.filter(n => n.kind === kind));
    const previews = new Map(groups[0].map(n=>[n.id,SmoGraph.previewGroups(appPreview(n.id))]));
    const nodeHeight=n=>n.kind==='xapp'?(previews.get(n.id).length?82+previews.get(n.id).length*38:104):76;
    const columnHeight=group=>group.reduce((sum,n)=>sum+nodeHeight(n),0)+Math.max(0,group.length-1)*16;
    const height = Math.max(710,92+Math.max(...groups.map(columnHeight)));
    canvas.setAttribute('viewBox',`0 0 1080 ${height}`);
    const defs = svg('defs');
    const marker = svg('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});
    marker.append(svg('path',{d:'M 1 1 L 9 5 L 1 9',fill:'none',stroke:'context-stroke','stroke-width':1.5})); defs.append(marker); canvas.append(defs);
    const positions = new Map(); const xs = [25,442,824]; const widths = [280,229,229];
    groups.forEach((group, c) => {
      canvas.append(svg('text',{x:xs[c]+1,y:29,class:'column-label'},['xApps','RCPs','KPIs'][c]));
      canvas.append(svg('line',{x1:xs[c],x2:xs[c]+widths[c],y1:42,y2:42,class:'column-rule'}));
      const extra=(height-92-columnHeight(group))/(group.length+1);
      let y=65+extra;
      group.forEach(n=>{const h=nodeHeight(n);positions.set(n.id,{x:xs[c],y,w:widths[c],h});y+=h+16+extra;});
    });
    const focus = relevant(nodes,edges);
    const gainLabels=[];
    for (const e of edges) {
      const {sensitivity,uncalibrated,measured,summary,estimates,dashed}=edgePresentation(e);
      const a = positions.get(e.source), b = positions.get(e.target);
      const sx = a.x+a.w, sy = a.y+a.h/2, ex = b.x, ey = b.y+b.h/2;
      const d = `M${sx},${sy} C${sx+65},${sy} ${ex-65},${ey} ${ex},${ey}`;
      let cls = 'edge';
      const presentation=edgePresentation(e);
      if (e.kind === 'influences') cls += measured ? ' measured'+(presentation.observedZero?' observed-zero':!presentation.confidenceResolved?' variable-gain':'') : presentation.declared?' declared':' candidate';
      if (e.deployment === 'native_locked') cls += ' locked';
      if (focus) cls += focus.edges.has(e.id) ? ' focus' : ' dimmed';
      if(focus?.intent && focus.side.has(e.id) && !focus.main.has(e.id))cls+=' intent-side';
      const conflictKind=conflictType(e.id);
      if(conflictKind&&focus?.edges.has(e.id)){
        cls+=' conflict-'+conflictKind;
        if(e.kind==='influences'&&conflicts.some(c=>c.type===conflictKind&&c.edge_ids.includes(e.id)&&c.dashed))cls+=' conflict-uncertain';
      }
      const path = svg('path',{d,class:cls,'marker-end':'url(#arrow)','data-edge':e.id});
      const hit = svg('path',{d,class:'edge-hit',tabindex:'0',role:'button','aria-label':`${find(e.source).label} to ${find(e.target).label}: ${e.evidence}`});
      const description=uncalibrated?'This action context is not calibrated':measured?`${magnitudeNote(estimates)}\n`+estimates.map(s=>`${contrastLabel(s)}: ${magnitudeText([s])} · ${resolved(s)?'resolved':zeroEffect(s)?'no observed change':'variable gain; interval includes zero'}`).join('\n'):e.kind==='controls'?'Declared writable control':presentation.declared?'Declared xApp objective path; numerical gain not calibrated':'Candidate influence; strength unestimated';
      hit.append(svg('title',{},description));
      hit.addEventListener('click',() => activate(e.id)); hit.addEventListener('keydown',ev => { if (ev.key === 'Enter' || ev.key === ' ') {ev.preventDefault();activate(e.id);} });
      canvas.append(path,hit);
      if (measured&&SmoGraph.influenceMagnitude(estimates))gainLabels.push({id:e.id,x:(sx+ex)/2,y:(sy+ey)/2-5,estimates});
    }
    for(const row of SmoGraph.gainLabelLayout(gainLabels,55,height-20)){
      if(Math.abs(row.labelY-row.y)>3)canvas.append(svg('line',{x1:row.x,y1:row.y+4,x2:row.x,y2:row.labelY+4,class:'gain-label-leader'}));
      const label=svg('text',{x:row.x,y:row.labelY,class:'edge-label','text-anchor':'middle','data-gain-edge':row.id},magnitudeText(row.estimates));
      label.append(svg('title',{},magnitudeNote(row.estimates)+'\n'+row.estimates.map(s=>`${contrastLabel(s)}: ${magnitudeText([s])} · ${resolved(s)?'resolved':zeroEffect(s)?'no observed change':'variable gain; interval includes zero'}`).join('\n')));canvas.append(label);
    }
    for (const n of nodes) {
      const p = positions.get(n.id), s = nodeState(n);
      const el = svg('g',{class:`node${state.selected === n.id ? ' selected' : ''}${focus && !focus.nodes.has(n.id) ? ' dimmed' : ''}`,transform:`translate(${p.x} ${p.y})`,tabindex:'0',role:'button','aria-label':`${n.paper_id || n.kind}: ${n.label}. ${valueText(n)}`});
      const incident=edges.filter(e=>e.source===n.id||e.target===n.id).map(e=>conflictType(e.id));
      const nodeConflict=['direct','indirect','implicit'].find(type=>incident.includes(type));
      if(nodeConflict)el.classList.add('conflict-'+nodeConflict);
      el.append(svg('rect',{width:p.w,height:p.h,rx:10,class:'node-card'}));
      if (n.kind === 'xapp') {
        el.append(svg('text',{x:14,y:20,class:'node-id'},n.paper_id));
        if (s.status) {
          el.append(svg('circle',{cx:p.w-44,cy:16.5,r:2.5,class:`status-circle ${s.enabled ? '' : 'off'}`}));
          el.append(svg('text',{x:p.w-35,y:20,class:`status-text ${s.enabled ? 'status-on' : 'status-off'}`},s.enabled ? 'ON' : 'OFF'));
        }
        el.append(svg('text',{x:14,y:42,class:'node-label'},n.label));
        const preview=appPreview(n.id),rows=previews.get(n.id);
        el.setAttribute('aria-label',`${n.paper_id}: ${n.label}. ${s.enabled?'ON':'OFF'}. ${previewStatus(preview)}. ${rows.map(row=>(previewLabels[row.parameter]||row.parameter)+' '+previewScope(row)+': '+previewValue(row.current,row.parameter)+' → '+previewValue(row.proposed,row.parameter)).join('; ')}`);
        const context=svg('text',{x:14,y:60,class:'preview-context'+(preview?.status==='invalid_proposal'?' preview-warning':'')},previewStatus(preview));
        context.append(svg('title',{},preview?`${previewStatus(preview)}; decision ${time(currentPreview().slot)}; report ${time(preview.report_produced_at_slot)}; not deployed`:currentPreview()?.reason||'No decision preview has been captured for this snapshot.'));
        el.append(context);
        if(rows.length){
          rows.forEach((row,i)=>{
            const y=73+i*38;
            el.append(svg('line',{x1:14,x2:p.w-14,y1:y,y2:y,class:'preview-rule'}));
            el.append(svg('text',{x:14,y:y+13,class:'preview-label'},previewLabels[row.parameter]||row.parameter));
            const scope=svg('text',{x:p.w-14,y:y+13,class:'preview-scope','text-anchor':'end'},previewScope(row));
            scope.append(svg('title',{},row.scopes.map(s=>[s.ue_id,s.cell_id].filter(Boolean).join(' / ')).join(', ')));el.append(scope);
            const value=svg('text',{x:14,y:y+29,class:'preview-value'+(row.changed===true?' changed':'')},previewValue(row.current,row.parameter)+' → '+previewValue(row.proposed,row.parameter));
            value.append(svg('title',{},JSON.stringify({parameter:row.parameter,scopes:row.scopes,current:row.current,proposed:row.proposed},null,2)));el.append(value);
            if(row.changed!==true)el.append(svg('text',{x:p.w-14,y:y+29,class:'preview-unchanged','text-anchor':'end'},row.changed===false?'不变':'待比较'));
          });
        }else el.append(svg('text',{x:14,y:83,class:'preview-context'},preview?.status==='no_action'?'算法本轮未提出新动作':'—'));
      } else if (n.kind === 'rcp') {
        el.append(svg('text',{x:14,y:22,class:'node-label'},n.label));
        const v = valueText(n);
        el.append(svg('text',{x:14,y:45,class:'node-value'},v.length > 21 ? v.slice(0,19)+'…' : v));
        const qualifier = population() ? (n.key === 'ue_drx_configs' ? 'DRX-enabled UEs' : n.key === 'ue_offload_fraction' ? 'MEC workload' : n.scope) : n.key === 'ue_drx_configs' ? 'X-01 native-locked' : `Per ${n.scope === 'UE' ? 'UE' : n.scope.toLowerCase()}`;
        el.append(svg('text',{x:14,y:63,class:'node-meta rcp-id'},qualifier));
      } else {
        el.append(svg('text',{x:14,y:22,class:'node-label'},n.label));
        el.append(svg('text',{x:14,y:46,class:'node-value'},valueText(n)));
        el.append(svg('text',{x:p.w-14,y:46,class:'node-unit','text-anchor':'end'},metricUnit(n.key,n.unit)));
        el.append(svg('text',{x:14,y:64,class:'node-meta kpi-id'},n.scope));
        const goal=kpiGoal(n);
        if(goal){const tag=svg('text',{x:p.w-14,y:64,class:'node-goal','text-anchor':'end'},goal.text);tag.append(svg('title',{},goal.title+'; this is not a measured trend'));el.append(tag);}
        el.append(svg('title',{},n.aggregation));
      }
      el.addEventListener('click',() => activate(n.id));
      el.addEventListener('keydown',ev => { if (ev.key === 'Enter' || ev.key === ' ') {ev.preventDefault();activate(n.id);} });
      canvas.append(el);
    }
  }

  function renderStatisticalFilter(){
    const info=graph().statistical_filter;
    $('pk-filter-panel').hidden=!info?.enabled;
    if(!info?.enabled)return;
    const relationLevel=info.action_context_policy==='retain_valid_measured',audit=state.bundle.primary_gain_audit;
    $('pk-filter-summary').textContent=relationLevel?
      `按 ${Math.round(info.minimum_confidence_level*100)}% 置信门槛确认 P–K 关系；每条关系至少有一个显著标定条件。保留 ${info.active_edge_count}/${info.original_edge_count} 条关系及其有效动作标定，其中 ${info.retained_uncertain_estimate_count} 条条件估计方向尚不确定；置信区间原样保留。${audit?.visible_link_gains_complete?'可见 P–K 连线均有有效数值增益。':''}`:
      `按 ${Math.round(info.minimum_confidence_level*100)}% 置信门槛筛选，区间需不含 0 · ${info.active_edge_count}/${info.original_edge_count} 条 P–K 连接 · 已排除 ${info.excluded_edge_count} 条连接、${info.excluded_estimate_count} 条动作条件估计。被排除部分不参与连接、组合选择或量化。`;
    const reasons={interval_includes_zero:'区间包含 0',observed_zero:'未观测到变化',confidence_interval_unavailable:'缺少有效置信区间',confidence_level_unverified:'置信水平未确认',uncalibrated_relation:'未标定',population_evidence_unavailable:'缺少群体证据',scalar_estimate_unavailable:'缺少标量估计',estimate_outside_interval:'估计与区间不一致'};
    const rows=[...info.excluded_relations.map(r=>({...r,fullyExcluded:true})),...info.partially_filtered_relations];
    $('pk-filter-list').innerHTML='<table><thead><tr><th>参数</th><th>KPI</th><th>排除记录</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${esc(find(r.source)?.label||r.source)}</td><td>${esc(find(r.target)?.label||r.target)}</td><td>${r.fullyExcluded?'整条连接已排除':relationLevel?'无效标定已排除':'保留显著条件'}<small>${esc([...new Set(r.excluded_estimates.map(s=>reasons[s.reason]||s.reason))].join('、')||reasons[r.reason]||r.reason)}</small></td></tr>`).join('')+'</tbody></table>';
    $('show-dashed').title='已被显著性筛选排除的 P–K 关系不会恢复，也不参与组合选择。';
  }

  function renderInspector() {
    const item = find(state.selected);
    $('inspector').hidden=!item&&!viewedGroup();
    if (!item) { $('inspector').innerHTML = viewedGroup()?'<div class="inspector-heading"><h2>组合冲突量化</h2></div>'+renderGroupQuantification(viewedGroup()):'<div class="inspector-heading"><h2>Select a node or link</h2></div>'; return; }
    let html = '';
    if (item.source && item.target) {
      const a = find(item.source), b = find(item.target);
      const view = edgePresentation(item), sensitivity = view.sensitivity;
      let gainDetails='';
      const candidate = item.kind === 'influences' && !view.measured;
      html += `<div class="inspector-heading"><span class="kind-badge evidence">${item.kind === 'controls' ? 'Control' : candidate ? view.declared?'Declared target':'Candidate' : 'Estimated'}</span><h2>${esc(a.label)} → ${esc(b.label)}</h2></div>`;
      if(item.kind==='influences'&&statisticsLabel())html+=section('Evidence source',`<p class="estimate-context">${esc(statisticsLabel())}</p>`);
      if(view.summary&&view.measured) {
        html+=section('Impact magnitude',contextSelect(item.source)+magnitudeTable(view.estimates));
        gainDetails=disclosure('Signed gains',contextTable(view.estimates));
      } else if (sensitivity) {
        const select=contextSelect(item.source);
        const isRange=sensitivity.kind==='range_linear_fit';
        const label=isRange || sensitivity.kind==='local_derivative' ? 'First-order gain' : 'Action effect';
        const ci=sensitivity.confidence_interval;
        let body=select+`<div class="large-value">${esc(signedEffect(sensitivity.estimate))}</div><p>${esc(sensitivity.unit)}</p>`;
        if (ci) body+=line(Number.isFinite(sensitivity.confidence_level)?Math.round(sensitivity.confidence_level*100)+'% interval':'Confidence interval',effectNumber(ci[0])+' … '+effectNumber(ci[1]));
        if (typeof sensitivity.high_minus_low_percent === 'number') body+=line('ΔKPI / reference',signedEffect(sensitivity.high_minus_low_percent)+'%');
        if (sensitivity.entity_estimates) body+=effectEntityTable(sensitivity);
        if(isRange){
          const domain=sensitivity.fit_domain.map(v=>sensitivity.x_unit==='fraction' ? fmt(v*100,0)+'%' : fmt(v,2)).join(' – ');
          body+=line('Range',domain+(sensitivity.x_unit==='ms'?' ms':''));
          const d=sensitivity.fit_diagnostics;
          body+=line('R²',d.within_seed_r2==null?'—':fmt(d.within_seed_r2,3));
          body+=rangeCurve(sensitivity,b.unit);
        }
        const flags=[];
        if(sensitivity.fit_diagnostics?.assessment==='limited_first_order_approximation')flags.push('Nonlinear response');
        if(zeroEffect(sensitivity))flags.push('No observed change');else if(!resolved(sensitivity))flags.push('Uncertain direction');
        if(sensitivity.excluded_levels?.length)flags.push('Missing values excluded');
        if(flags.length)body+=`<p class="gain-status">${esc(flags.join(' · '))}</p>`;
        html+=section('Impact magnitude',select+`<div class="large-value" title="${esc(magnitudeNote([sensitivity]))}">${esc(magnitudeText([sensitivity]))}</div>`);
        gainDetails=disclosure(label,body.replace(select,''));
      } else {
        if (item.kind === 'influences') html+=section('Influence',contextSelect(item.source)+line('Population response',population() ? 'Not identified' : 'Not estimated'));
      }
      if (item.kind === 'controls') html += section('Deployment',item.deployment==='native_locked'?'<p class="gain-status">Native-locked</p>':'<p>Controllable</p>');
      if (item.supporting_xapps?.length) html += section('xApps',`<div class="connections">${item.supporting_xapps.map(id => connection('xapp:'+id,find('xapp:'+id)?.paper_id || id)).join('')}</div>`);
      const measurement=sensitivity||view.estimates[0];
      const measurements=measurement ? line('Seeds',measurement.sample_count)+line('Response horizon',fmt(measurement.horizon_ms,0)+' ms')+(measurement.scoring_window_ms?line('Scoring window',fmt(measurement.scoring_window_ms,0)+' ms'):'')+(measurement.kind==='range_linear_fit'?line('Sampled levels',measurement.fit_point_count+' / '+measurement.grid_point_count)+line('RMSE',effectNumber(measurement.fit_diagnostics.rmse)+' '+b.unit):'')+`<p class="estimate-context">${esc(measurement.scope)}<br>${esc(view.summary?'Contexts listed separately above.':measurement.context)}</p>`:'';
      html += disclosure('Details',gainDetails+measurements+`<p class="estimate-context">${esc(item.note || (item.kind==='controls'?'Declared writable control.':'Conditional parameter-to-KPI evidence.'))}</p>`+(view.estimates.length ? disclosure('Evidence',`<pre>${esc(JSON.stringify(view.estimates.length===1?view.estimates[0]:view.estimates,null,2))}</pre>`) : '')+(item.source_files || []).map(f => `<div class="source-file">${esc(f)}</div>`).join(''));
    } else {
      const s = nodeState(item);
      html += `<div class="inspector-heading"><span class="kind-badge ${item.kind}">${esc(item.paper_id || item.kind.toUpperCase())}</span><h2>${esc(item.label)}</h2><p>${esc(item.kind === 'xapp' ? item.algorithm : item.scope+' · '+metricUnit(item.key,item.unit))}</p></div>`;
      if (item.kind === 'xapp') {
        const effects=currentQuantification()?.apps?.find(a=>a.app_id===item.id)?.effects||[];
        if(effects.length)html+=section('当前动作 · 一阶 KPI 推动量',effects.map(e=>line(chineseKpis[e.kpi]||find('kpi:'+e.kpi)?.label||e.kpi,e.delta==null?'无法估计':(e.delta>0?'+':'')+fmt(e.delta,4)+' '+(find('kpi:'+e.kpi)?.unit||''))).join(''));
        html += section('Runtime state',line('Enabled',s.status || 'No run attached')+line('Active scope',(s.active_cells || []).join(', ') || '—')+line('Current suggestions',(s.current_proposals || []).length)+line('Last suggestion',time(s.proposed_at_slot))+line('Last applied receipt',time(s.last_applied?.slot)));
        html += section(population() ? 'Related KPIs' : 'Target KPIs',`<div class="connections">${Object.entries(item.target_kpis).map(([k,d]) => connection('kpi:'+k,find('kpi:'+k)?.label || k,d.replace('_',' '))).join('')}</div>`);
        if(item.policy_evaluation){
          const ev=item.policy_evaluation;
          html+=disclosure('Matched validation',`<p>${esc(ev.duration_s)} s / arm · ${esc(ev.seeds.length)} paired seeds</p><table class="validation-table"><thead><tr><th>KPI</th><th>Base</th><th>Native DRX</th><th>DRX control</th></tr></thead><tbody>${ev.metrics.map(m=>`<tr><td>${esc(m.label)}<small>${esc(m.unit)}</small></td><td>${esc(fmt(m.base,3))}</td><td>${esc(fmt(m.native_drx,3))}</td><td>${esc(fmt(m.unlocked_drx,3))}</td></tr>`).join('')}</tbody></table><p>Policy-level results; not single-parameter gains.</p>`);
        }
        const current = s.last_proposals || [];
        const grouped=new Map();
        for (const p of current) {const key=proposalTitle(p),old=grouped.get(key);grouped.set(key,{count:(old?.count || 0)+1,slot:Math.max(old?.slot || 0,p.slot_id || 0)});}
        if (population()) {
          html += section('Recorded actions',current.length ? [...grouped].map(([title,p])=>`<div class="proposal"><strong>${esc(title)}</strong>${line('Proposals',p.count)}${line('Latest',time(p.slot))}</div>`).join('') : '<p>No proposal recorded at this point.</p>');
        } else {
        html += section('Latest recorded suggestions',current.length ? (!s.enabled ? '<p>Historical suggestions; this xApp is currently OFF.</p>' : '<p>Suggestions are separate from deployed parameters.</p>')+current.map(p => `<div class="proposal"><div class="proposal-top"><span>${esc(p.ue_id || p.cell_id || 'All cells')}</span><span>${esc(time(p.slot_id))}</span></div><strong>${esc(proposalTitle(p))}</strong><details><summary>Action details</summary><pre>${esc(JSON.stringify(p,null,2))}</pre></details></div>`).join('') : '<p>No proposal recorded at this point.</p>');
        }
        const controls = graph().edges.filter(e => e.source === item.id);
        html += section('Writable controls',`<div class="connections">${controls.map(e => connection(e.target,find(e.target).label,e.deployment === 'native_locked' ? 'native locked' : 'writable')).join('')}</div>`);
        html += disclosure('Model & contract',`<p>${esc(item.evaluation_status.replaceAll('_',' '))}</p><div class="source-file">SHA-256<br>${esc(item.model_sha256)}</div><div class="source-file">${esc(item.source)}</div>`);
      } else {
        html += section(item.kind === 'rcp' ? 'Executed parameter' : 'Observed performance',`<div class="large-value">${esc(valueText(item))}<span class="large-unit">${item.kind === 'kpi' ? esc(metricUnit(item.key,item.unit)) : ''}</span></div>`+line('Sample',time(s.sampled_at_slot))+(item.kind === 'kpi' ? line('Aggregation',s.window || 'Unavailable')+(kpiGoal(item)?line(state.intent?.ranges.some(r=>r.kpi===item.key)?'Intent target':'Reference target',kpiGoal(item).text):'') : '')+(population() ? (item.kind === 'rcp' ? populationParameters(item,s) : '') : entityTable(s.by_entity,item.unit)));
        if (item.kind === 'kpi') html += disclosure('Definition',`<p>${esc(item.aggregation)}</p><div class="source-file">${esc(s.source || 'No source attached')}${s.window_start_slot != null ? '<br>Window: '+esc(time(s.window_start_slot))+' → '+esc(time(s.sampled_at_slot)) : ''}</div>`);
        if(item.kind==='rcp'&&contextSelect(item.id))html+=section('Measured change',contextSelect(item.id));
        const edges = graph().edges.filter(e => e.source === item.id || e.target === item.id);
        html += section(item.kind === 'rcp' ? 'Connected xApps & KPIs' : 'Potential parameter influences',`<div class="connections">${edges.map(e => {
          const other = find(e.source === item.id ? e.target : e.source);
          const v=edgePresentation(e);
          return connection(e.id,other.label,e.kind==='controls'?'control':!v.measured?v.declared?'declared target':'uncalibrated':v.observedZero?'observed zero':v.confidenceResolved?'measured':'variable gain');
        }).join('')}</div>`);
      }
    }
    $('inspector').innerHTML = html;
    $('inspector').querySelectorAll('[data-select]').forEach(b => b.addEventListener('click',() => activate(b.dataset.select)));
    $('estimate-context')?.addEventListener('change',ev => {state.contextViews[ev.target.dataset.parameter]=ev.target.value;render();});
  }

  function renderRuntimeEnvironment() {
    const environment=state.bundle.runtime_environment;
    $('runtime-environment').hidden=!environment;
    const source=statisticsLabel();
    $('graph-statistics-source').hidden=!source;
    $('graph-statistics-source').textContent=source?(source+' · 运行状态来自'+(environment.label||environment.id)):'';
    $('comparison-window-label').textContent=fixedBeforeSlots()==null?'前后窗口':'部署后窗口';
    $('comparison-window-duration').textContent='slots = '+windowDuration(Number($('comparison-slots').value));
    const validated=environment?.validation;
    const pairedValidation=validated?.window_verified===true&&validated.paired_controllers_verified===true&&
      validated.window_basis==='actual_final_paired_report'&&validated.observe_slots===20000&&validated.before_observe_slots===2000&&
      /^[0-9a-f]{64}$/.test(validated.paired_report_sha256||'')&&
      Array.isArray(validated.paired_evaluated_models)&&validated.paired_evaluated_models.length>0&&
      validated.paired_evaluated_models.length<=6&&new Set(validated.paired_evaluated_models).size===validated.paired_evaluated_models.length&&
      validated.paired_evaluated_models.every(id=>['x01','x02','x08','x12_mec','x12_urllc','x13'].includes(id));
    const validatedSlots=(validated?.six_apps_verified===true||pairedValidation)&&Number.isInteger(validated.observe_slots)&&validated.observe_slots>0&&validated.observe_slots<=20000?validated.observe_slots:null;
    $('comparison-validated-window').hidden=validatedSlots==null;
    if(validatedSlots!=null){
      $('comparison-validated-window').dataset.slots=String(validatedSlots);
      $('comparison-validated-window').textContent=windowDuration(validatedSlots).replace(' s','秒')+(validated.cohort_followup_verified?'主窗口':'已验证窗口');
    }
    if(!environment)return;
    $('runtime-environment-label').textContent=environment.label||environment.id;
    const statistics=environment.statistics||{};
    $('runtime-environment-statistics').hidden=!source;
    $('runtime-environment-statistics').innerHTML=esc(source)+
      (statistics.reference_url?.startsWith('/')&&!statistics.reference_url.startsWith('//')?` · <a href="${esc(statistics.reference_url)}">${esc(statistics.reference_label||'历史统计参考')}</a>`:'');
    const c=environment.configuration||{},parts=[];
    if(c.cells!=null)parts.push(c.cells+' cells');
    if(c.ues!=null)parts.push(c.ues+' UEs');
    if(c.carrier_frequency_ghz!=null)parts.push(c.carrier_frequency_ghz+' GHz');
    if(c.bandwidth_mhz!=null)parts.push(c.bandwidth_mhz+' MHz');
    if(c.ue_speed_mps!=null)parts.push('UE '+c.ue_speed_mps+' m/s');
    if(c.rt_max_depth!=null)parts.push('RT depth '+c.rt_max_depth);
    if(c.traffic_load!=null)parts.push('负载 '+c.traffic_load);
    const notes=[parts.join(' · ')];
    if(c.urllc_profile)notes.push('URLLC '+c.urllc_profile);
    if(c.mec_offload_fractions)notes.push('MEC offload '+Object.entries(c.mec_offload_fractions).map(([id,fraction])=>id+' '+fmt(fraction*100,0)+'%').join(' / '));
    if(pairedValidation)notes.push('部署前窗口 '+validated.before_observe_slots+' slots（'+windowDuration(validated.before_observe_slots)+'）');
    if(validatedSlots!=null)notes.push((pairedValidation?'所选 xApp 成对验证（'+validated.paired_evaluated_models.length+' 个'+(validated.cohort_followup_verified?'，同出生队列':'')+'）':'6 个 xApp 已验证')+' · '+validatedSlots+' slots（'+windowDuration(validatedSlots)+'）'+(validated.seeds?.length?' · seeds '+validated.seeds.join(' / '):''));
    $('runtime-environment-details').innerHTML=notes.filter(Boolean).map(note=>'<p>'+esc(note)+'</p>').join('')+
      (validated?.report_url?.startsWith('/')&&!validated.report_url.startsWith('//')?`<p><a href="${esc(validated.report_url)}">${pairedValidation?'查看所选控制器验证报告':'查看环境验证报告'}</a></p>`:'');
  }

  function render() {
    if (!state.bundle) return;
    ensureDecisionPreview();renderDecisionPreview();ensureComparison();ensureConflictQuantification();renderConflictQuantificationStatus();
    const src = state.bundle.source, f = frame();
    $('mode').textContent = src.mode === 'recorded' ? 'Replay' : src.mode === 'live' ? (src.wall_age_s >= 60 ? 'Stale' : 'Live') : src.mode === 'snapshot' ? 'Snapshot' : 'Contracts';
    $('source-name').textContent = src.name;
    $('stage').textContent = src.mode === 'live' ? `· ${fmt(src.wall_age_s,0)} s since publication` : src.mode === 'contracts' ? '' : '· '+src.stage;
    $('playback').hidden = src.mode !== 'recorded';
    $('timeline').max = Math.max(0,src.frames.length-1);
    $('timeline').value = state.index;
    $('clock').textContent = `${fmt((f.time_ms || 0)/1000,3)} / ${fmt((src.total_slots || 0)*(src.slot_duration_ms || .5)/1000,3)} s`;
    $('targets').setAttribute('aria-pressed',String(!state.all)); $('all-kpis').setAttribute('aria-pressed',String(state.all));
    $('window-note').textContent = src.mode === 'contracts' ? '' : `Rate window: ${fmt(f.window_ms,1)} ms`;
    $('download').disabled = false; $('loading').hidden = true;
    $('show-dashed').textContent=state.showDashed?'Hide dashed':'Show dashed';
    $('show-dashed').setAttribute('aria-pressed',String(state.showDashed));
    document.querySelector('.line.locked').parentElement.hidden=!graph().edges.some(e=>e.deployment==='native_locked');
    for(const el of document.querySelectorAll('[data-dashed-legend]'))el.hidden=!state.showDashed;
    renderRuntimeEnvironment();renderStatisticalFilter();renderIntent(); renderDeployment(); renderComparison(); renderGraph(); renderInspector();
  }

  function stop() { state.playing=false; clearInterval(state.timer); state.timer=null; $('play').textContent='▶'; $('play').setAttribute('aria-label','Play recorded simulation'); }
  function initialize(bundle) {
    const initial = !state.bundle;
    const changed=state.bundle?.source.deployment_job_id!==bundle.source.deployment_job_id||state.bundle?.source.run_id!==bundle.source.run_id;
    state.bundle=bundle;deploymentState=bundle.deployment||deploymentState;
    migrateIntentIds();
    const scenario=bundle.scenario_view,scenarioSelect=$('scenario-select');
    $('scenario-control').hidden=!scenario?.choices?.length||location.protocol==='file:';
    if(scenario?.choices?.length){
      scenarioSelect.replaceChildren(...scenario.choices.map(c=>{const o=document.createElement('option');o.value=c.url;o.textContent=c.label;o.disabled=c.available===false;o.selected=c.id===scenario.current;return o;}));
      scenarioSelect.dataset.current=scenarioSelect.value;
    }
    if (initial) {
      for (const n of graph().nodes.filter(n => n.kind === 'xapp')) {
        const o=document.createElement('option'); o.value=n.implementation_id; o.textContent=n.paper_id+' / '+n.label; $('app-focus').append(o);
      }
      const allOn = bundle.source.frames.findIndex(f => Object.values(f.states).filter(s => s.enabled && s.current_proposals?.length).length === 6);
      state.index = allOn >= 0 ? allOn : Math.max(0,bundle.source.frames.length-1);
      state.selected = graph().influence_input ? 'influence:cell_sleep_mode:ap_power_w' : 'xapp:ue_energy_rf';
      state.focus = graph().influence_input ? '' : 'ue_energy_rf'; $('app-focus').value=state.focus;
      if(location.protocol!=='file:'){
        try{const ids=JSON.parse(localStorage.getItem(intentStorageKey())||'[]');if(ids.length)reviewSavedIntents(ids);}catch{}
      }
    } else state.index=changed?Math.max(0,bundle.source.frames.length-1):Math.min(state.index,Math.max(0,bundle.source.frames.length-1));
    $('error').hidden=true; render();
  }
  async function load() {
    try {
      const r=await fetch('./api/graph',{cache:'no-store'}); const data=await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      initialize(data);
    } catch(e) {
      $('error').hidden=false; $('error').querySelector('span').textContent='Could not read network state. '+e.message;
      $('mode').textContent='Source unavailable';
      if (!state.bundle) { $('loading').querySelector('strong').textContent='Network source unavailable'; $('loading').querySelector('span').textContent='Check the selected run or use the contract-only service.'; }
    }
  }

  function main() {
    if(['127.0.0.1','localhost'].includes(location.hostname)&&location.port==='53504'){const target=new URL(location.href);target.port='53505';location.replace(target.href);return;}
    setupIntent();
    $('scenario-select').addEventListener('change',async ev=>{
      const select=ev.target,message=$('scenario-message'),target=new URL(select.value,location.origin);
      select.disabled=true;select.setAttribute('aria-busy','true');message.textContent='Loading…';
      delete select.dataset.state;
      try{
        if(target.origin!==location.origin||!target.pathname.startsWith('/contexts/'))throw new Error('Unsupported scenario URL');
        const response=await fetch(new URL('api/graph',target),{cache:'no-store'});
        if(!response.ok)throw new Error('Scenario unavailable');
        location.assign(target.href);
      }catch{
        select.value=select.dataset.current;select.disabled=false;select.removeAttribute('aria-busy');select.dataset.state='error';
        message.textContent='场景读取失败，请重新选择。';
      }
    });
    syncPresetStore();
    $('catalog-export').addEventListener('click',exportCombinationCatalog);
    $('conflict-method').addEventListener('change',()=>{ensureConflictQuantification(true);render();});
    $('conflict-quantification-refresh').addEventListener('click',()=>{ensureConflictQuantification(true);render();});
    $('quantification-full').addEventListener('toggle',renderQuantificationPanel);
    $('conflict-quantification-export').addEventListener('click',()=>{const data=currentQuantification();if(!data?.method)return;const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='conflict-quantification.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    $('conflict-align-snapshot').addEventListener('click',async()=>{const slot=currentQuantification()?.suggested_slot,index=state.bundle.source.frames.findIndex(f=>f.slot===slot);if(index<0)return;stop();state.index=index;await reviewSavedIntents(state.activeIntents.map(r=>r.id));render();});
    $('comparison-capture').addEventListener('click',()=>{stop();requestComparison(true);});
    $('comparison-refresh').addEventListener('click',()=>requestComparison(false));
    $('comparison-slots').addEventListener('input',()=>{comparisonSlotsDirty=true;render();});
    $('comparison-validated-window').addEventListener('click',()=>{
      $('comparison-slots').value=$('comparison-validated-window').dataset.slots;
      comparisonSlotsDirty=true;render();
    });
    $('evaluation-origin').addEventListener('click',prepareEvaluationOrigin);
    $('deployment-start').addEventListener('click',startDeployment);
    $('deployment-stop').addEventListener('click',stopDeployment);
    $('deployment-resume').addEventListener('click',resumeDeployment);
    $('deployment-reconnect').addEventListener('click',async()=>{try{deploymentState=await postIntent('./api/deployment/reconnect',{});deploymentError='';await refreshDeployment();}catch(e){deploymentError=e.message;renderDeployment();}});
    if(location.protocol!=='file:')setInterval(refreshDeployment,1500);
    $('app-focus').addEventListener('change',ev => {state.graphGroup=null;state.focus=ev.target.value;state.selected=state.focus ? 'xapp:'+state.focus : null;render();});
    $('targets').addEventListener('click',() => {state.all=false;render();});
    $('all-kpis').addEventListener('click',() => {state.all=true;render();});
    $('show-dashed').addEventListener('click',()=>{state.showDashed=!state.showDashed;render();});
    $('reset').addEventListener('click',() => {state.graphGroup=null;state.focus='';state.selected=null;state.all=false;state.contextViews={};$('app-focus').value='';$('graph-scroll').scrollTo(0,0);render();});
    $('timeline').addEventListener('input',ev => {stop();state.index=Number(ev.target.value);render();});
    $('play').addEventListener('click',() => {
      if (state.playing) return stop();
      state.playing=true;$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Pause recorded simulation');
      if (state.index === state.bundle.source.frames.length-1) state.index=0;
      state.timer=setInterval(() => {state.index++;if(state.index>=state.bundle.source.frames.length){state.index--;stop();}render();},180);
    });
    $('download').addEventListener('click',() => {
      const g=structuredClone(graph()); for(const n of g.nodes) n.state=frame().states[n.id] || null;
      const data={...g,source:{...state.bundle.source,frames:undefined},scenario_view:state.bundle.scenario_view,runtime_environment:state.bundle.runtime_environment,snapshot:{slot:frame().slot,time_ms:frame().time_ms,window_ms:frame().window_ms},decision_preview:currentPreview(),intent_comparison:managedComparison()?.comparison||comparisonResult,deployment:deploymentState,intent_review:state.intent,combination_catalog:state.combinationCatalog,conflict_quantification:currentQuantification(),graph_conflict_view:state.conflictView,view_options:{show_dashed:state.showDashed,parameter_contexts:state.contextViews,intent_case:$('intent-case').value||null,viewed_combination:state.graphGroup,analysis_basis:'knowledge_graph',automatic_selection:false,deployment_enabled:false}};
      const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
      const a=document.createElement('a');a.href=u;a.download='oran-knowlege-graph.json';a.click();setTimeout(() => URL.revokeObjectURL(u),1000);
    });
    $('retry').addEventListener('click',load);
    document.addEventListener('keydown',ev => {if(ev.key === 'Escape'){if(state.editingPreset){ev.preventDefault();closePresetEditor();return;}state.graphGroup=null;state.focus='';state.selected=null;$('app-focus').value='';render();}});
    if (window.SMO_BOOTSTRAP) initialize(window.SMO_BOOTSTRAP);
    else {load();state.polling=setInterval(() => {if(!state.bundle || ['live','snapshot'].includes(state.bundle.source.mode)) load();},1500);}
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded',main); else main();
})();
