/* Preregistered intent obstruction in reference-influence coordinates only.
 * No observed deployment labels, fitted score parameters, or KPI forecasts.
 */
var SmoIntentObstruction = (() => {
  'use strict';
  const method='net_intent_obstruction_v1';
  const temporalMethod='temporal_net_intent_obstruction_v1';
  const lookbackSlots=500;
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  const nonempty=x=>typeof x==='string'&&x.length>0;
  const distance=(x,low,high)=>Math.max(low-x,0,x-high);
  const displayScore=x=>finite(x)&&x>=0?100*(x/(1+x)):null;

  function goalProgress(context,row,referenceDrive){
    if(!finite(referenceDrive)||!finite(context.current)||!finite(context.scale)||context.scale<=0)
      return {available:false,reason:'reference_vector_or_context_unavailable'};
    const domain=row.domain||[0,null],low=domain[0],high=domain[1];
    if(finite(low)&&context.current<low||finite(high)&&context.current>high||
      finite(low)&&finite(high)&&low>high)
      return {available:false,reason:'reference_outside_physical_domain'};
    const physicalLow=finite(low)?(low-context.current)/context.scale:-Infinity;
    const physicalHigh=finite(high)?(high-context.current)/context.scale:Infinity;
    // One projection of the whole expected joint vector. Never clamp one
    // controller separately and then add artificial credits from its peers.
    const projected=Math.max(physicalLow,Math.min(physicalHigh,referenceDrive));
    const limits=context.boundaries||[];
    let progress=0,mode='neutral_physical_domain';
    if(limits.length===1){
      progress=limits[0].sign*projected;mode='directional_reference_change';
    }else if(limits.length){
      const a=finite(row.lower)?(row.lower-context.current)/context.scale:-Infinity;
      const b=finite(row.upper)?(row.upper-context.current)/context.scale:Infinity;
      progress=distance(0,a,b)-distance(projected,a,b);mode='interval_distance_reduction';
    }
    return {available:finite(progress),reason:finite(progress)?null:'nonfinite_reference_effect',
      mode,reference_progress:finite(progress)?progress:null,
      expected_normalized_reference_drive:referenceDrive,
      physical_projected_reference_drive:projected,
      physical_saturation_applied:projected!==referenceDrive,
      normalized_physical_domain:[finite(physicalLow)?physicalLow:null,finite(physicalHigh)?physicalHigh:null],
      strict_endpoints_scored_as_distance_to_closure:true};
  }

  function aggregateEffects(progress){
    if(!progress.length||!progress.every(finite))return {reference_effect:null,obstruction:null};
    const effect=progress.some(x=>x<0)?Math.min(...progress):progress.reduce((a,b)=>a+b,0)/progress.length;
    return {reference_effect:effect,obstruction:Math.max(0,-effect)};
  }

  function historySamples(frame,preview,options){
    if(!nonempty(options.sourceIdentity)||!nonempty(options.graphContextSha256))
      return {reason:'history_context_unverified'};
    if(!Array.isArray(options.history)||!options.history.length)return {reason:'history_unavailable'};
    if(!Number.isInteger(frame.slot)||!finite(frame.time_ms))return {reason:'history_reference_time_unavailable'};
    const allowed=new Set(['frame','preview','source_identity','graph_context_sha256']),seen=new Set(),samples=[];
    const models=new Map((preview.apps||[]).map(a=>[a.app_id,a.model_sha256]));
    for(const sample of options.history){
      if(!sample||Object.keys(sample).some(k=>!allowed.has(k)))return {reason:'history_untrusted_fields'};
      const f=sample.frame,p=sample.preview;
      if(!f||!p||!Number.isInteger(f.slot)||f.slot!==p.slot||!finite(f.time_ms))
        return {reason:'history_snapshot_mismatch'};
      if(f.slot>frame.slot||f.time_ms>frame.time_ms)return {reason:'history_future_sample'};
      if(f.slot<=frame.slot-lookbackSlots)return {reason:'history_outside_lookback'};
      if(sample.source_identity!==options.sourceIdentity||sample.graph_context_sha256!==options.graphContextSha256)
        return {reason:'history_context_mismatch'};
      if(seen.has(f.slot))return {reason:'history_duplicate_snapshot'};
      if(p.deployment_enabled===true||finite(p.observation_slot)&&p.observation_slot>p.slot)
        return {reason:'history_invalid_shadow_preview'};
      for(const app of p.apps||[]){
        const expected=models.get(app.app_id);
        if(nonempty(expected)&&app.model_sha256!==expected)return {reason:'history_model_mismatch'};
      }
      seen.add(f.slot);samples.push({frame:f,preview:p});
    }
    samples.sort((a,b)=>a.frame.slot-b.frame.slot);
    return {samples};
  }

  function review(graph,frame,groups,ranges,preview,options={}){
    const selected=options.method||method,temporal=selected===temporalMethod;
    if(![method,temporalMethod].includes(selected))throw Error('Unsupported net-intent-obstruction method');
    const engine=options.projectionEngine||(typeof SmoGoalObstruction==='undefined'?null:SmoGoalObstruction);
    const base={method:selected,schema:'oran.net-intent-obstruction.v1',status:'ready',
      snapshot:{slot:frame?.slot,time_ms:frame?.time_ms},preview_slot:preview?.slot,groups:[],
      score_mapping:'100*D/(1+D)',score_semantics:'worst_negative_normalized_reference_intent_progress',
      reference_effect_semantics:'worst_negative_progress_else_equal_mean_positive_progress',
      reference_effect_unit:'dimensionless_normalized_progress',reference_effect_score_multiplier:100,
      aggregation_order:'mean_joint_reference_vectors_then_physical_domain_projection_then_intent_functional',
      kpi_value_weights:'equal',normalized_drive_unit:'dimensionless',
      normalization:options.normalization||null,
      kpi_forecast_performed:false,observed_after_kpis_used:false,deployment_results_used:false,
      fitted_score_parameters:false,selection_performed:false,deployment_enabled:false,
      arbitration_performed:false,arbitration_evaluated:'exact_fresh_uniform_request_distribution_without_random_draw',
      temporal_scope:temporal?'mixture_of_recorded_pre_intent_joint_shadow_decisions':'one_frozen_joint_shadow_decision',
      history_lookback_slots:temporal?lookbackSlots:null,horizon_scaling_performed:false,
      excluded_relations_are_zero_physical_effects:false};
    const unavailable=reason=>({...base,status:'unavailable',reason});
    if(!engine?.projectOutcomes)return unavailable('reference_projector_unavailable');
    if(!frame||!preview||preview.slot!==frame.slot)return unavailable('matching_decision_preview_unavailable');
    if(preview.deployment_enabled===true||finite(preview.observation_slot)&&preview.observation_slot>preview.slot)
      return unavailable('invalid_shadow_preview');
    let samples=[{frame,preview}];
    if(temporal){
      const history=historySamples(frame,preview,options);
      if(history.reason)return unavailable(history.reason);
      samples=history.samples;
    }
    const project=(f,p)=>engine.projectOutcomes(graph,f,groups,ranges,p,{
      ...options,method:'goal_obstruction_v2',history:undefined,intentFrame:frame});
    // Current intent geometry/scales are frozen even for the historical
    // decisions; those records supply only contemporaneous joint RCP deltas.
    const current=project(frame,preview);
    if(current.status!=='ready')return unavailable(current.reason||'reference_projection_unavailable');
    const contexts=current.intent_contexts,byRange=new Map(ranges.map(r=>[r.kpi,r]));
    const projected=samples.map(s=>s.frame===frame&&s.preview===preview?current:project(s.frame,s.preview));
    if(projected.some(r=>r.status!=='ready'))
      return unavailable(projected.find(r=>r.status!=='ready').reason||'history_projection_unavailable');
    base.intent_contexts=contexts;
    base.history_slots=temporal?samples.map(s=>s.frame.slot):[];
    base.history_sample_count=temporal?samples.length:0;
    base.history_sample_status=temporal?(samples.length===1?'single_sample':'correlated_pre_intent_samples'):null;
    for(const group of groups){
      const id=group.group_id||group.id,records=projected.map(p=>p.groups.find(g=>g.group_id===id));
      const output={group_id:id,app_ids:[...group.app_ids],score:null,raw_obstruction_strength:null,
        reference_effect:null,reference_effect_score:null,score_status:'insufficient_evidence',
        reason:null,goal_components:[],reference_subset_count:0,
        arbitration_outcome_count:records.reduce((n,r)=>n+(r?.actual_outcomes?.length||0),0),
        projection_coverage:records[0]?.projection_coverage||null,
        temporal_projection_coverage:temporal?records.map((r,i)=>({slot:samples[i].frame.slot,coverage:r?.projection_coverage||null})):[],
        model_coverage_complete:records.every(r=>r?.model_coverage_complete===true),
        held_lease_state_not_reconstructed:records.some(r=>r?.held_lease_state_not_reconstructed),
        structural_counts:current.groups.find(g=>g.group_id===id)?.structural_counts||null,
        reference_frame_slots:samples.map(s=>s.frame.slot)};
      if(records.some(r=>!r?.actual_outcomes?.length)){
        output.reason=records.find(r=>!r?.actual_outcomes?.length)?.reason||'proposal_unavailable';
        base.groups.push(output);continue;
      }
      let reason=null,approximation=false;
      for(const context of contexts){
        const row=byRange.get(context.kpi),perSample=[];
        for(const record of records){
          const mass=record.actual_outcomes.reduce((n,r)=>n+r.probability,0);
          if(!finite(mass)||mass<=0){reason='invalid_arbitration_probability_mass';break;}
          let drive=0;
          for(const outcome of record.actual_outcomes){
            const goal=outcome.goals.find(g=>g.kpi===context.kpi);
            if(!finite(outcome.probability)||outcome.probability<0||!finite(goal?.reference_drive)){
              reason='reference_projection_incomplete';break;
            }
            drive+=outcome.probability*goal.reference_drive/context.scale/mass;
            if(goal.excluded_parameters?.length||goal.chains?.some(c=>c.notes?.some(n=>n!=='local_reference_drive_not_kpi_change')))
              approximation=true;
          }
          if(reason)break;
          perSample.push(drive);
        }
        if(reason)break;
        const drive=perSample.reduce((a,b)=>a+b,0)/perSample.length;
        const progress=goalProgress(context,row,drive);
        if(!progress.available){reason=progress.reason;break;}
        output.goal_components.push({kpi:context.kpi,...progress,kpi_scale:context.scale,value_weight:1,
          per_snapshot_expected_normalized_reference_drive:perSample,
          normalized_obstruction:Math.max(0,-progress.reference_progress),
          no_active_push:drive===0});
      }
      if(reason){output.reason=reason;base.groups.push(output);continue;}
      const effect=aggregateEffects(output.goal_components.map(g=>g.reference_progress));
      output.reference_effect=effect.reference_effect;
      output.reference_effect_score=effect.reference_effect===null?null:100*effect.reference_effect;
      output.raw_obstruction_strength=effect.obstruction;
      output.score=displayScore(effect.obstruction);
      output.no_active_push=output.goal_components.every(g=>g.no_active_push);
      output.score_status=!output.model_coverage_complete?'partial_model_score':
        approximation||output.held_lease_state_not_reconstructed||temporal?'reference_approximation':
          output.no_active_push?'no_active_push':'reference_score';
      base.groups.push(output);
    }
    return base;
  }
  return {method,temporalMethod,lookbackSlots,review,goalProgress,aggregateEffects,displayScore};
})();
