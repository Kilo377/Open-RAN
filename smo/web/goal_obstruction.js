/* Frozen-proposal goal obstruction. No deployment, feedback fit or KPI forecast. */
var SmoGoalObstruction = (() => {
  'use strict';
  const method='goal_obstruction_v2';
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  const clone=x=>structuredClone(x);
  const stable=x=>x&&typeof x==='object'?Array.isArray(x)?x.map(stable):Object.fromEntries(Object.keys(x).sort().map(k=>[k,stable(x[k])])):x;
  const encoded=x=>JSON.stringify(stable(x));
  const scheduler=new Set(['scheduler_mode','ue_scheduler_weights']);
  const clip=x=>Math.max(0,Math.min(1,x));

  function goalContext(row,frame,referenceEngine,normalization){
    const original=referenceEngine.intentContext(row,frame),domain=row.domain||[0,null];
    const ratio=row.kpi==='urllc_failure_fraction',unit=row.unit===undefined?'ratio':row.unit;
    const ratioScale=['ratio','fraction'].includes(unit)?1:unit==='%'?100:null;
    const fallbackScale=ratio?ratioScale:finite(original.baseline)&&original.baseline!==0?Math.abs(original.baseline):null;
    const context={...original,scale:normalization?normalization.scale:fallbackScale,
      scale_basis:normalization?.scale_basis||(ratio?(ratioScale===null?'incompatible_ratio_unit':'ratio_percentage_points'):'frozen_reference_magnitude'),weight:1,weight_basis:'equal_kpi_value'};
    const low=row.lower,high=row.upper,point=finite(low)&&finite(high)&&low===high;
    const boundaries=[];
    if(finite(low)&&(point||low!==domain[0]||row.lower_inclusive===false))boundaries.push({side:'lower',bound:low,sign:1});
    if(finite(high)&&(point||domain[1]==null||high!==domain[1]||row.upper_inclusive===false))boundaries.push({side:'upper',bound:high,sign:-1});
    for(const b of boundaries)b.margin=finite(context.current)&&context.scale>0?b.sign*(context.current-b.bound)/context.scale:null;
    const margin=boundaries.length&&finite(context.current)&&context.scale>0?Math.min(...boundaries.map(b=>b.margin)):null;
    // Only machine-rounding tolerance at a band centre; no learned epsilon.
    const tolerance=Number.EPSILON*16*Math.max(1,...boundaries.map(b=>Math.abs(b.margin||0)));
    const active=margin===null?[]:boundaries.filter(b=>Math.abs(b.margin-margin)<=tolerance);
    return {...context,boundaries,active_boundaries:active,normalized_margin:margin,
      neutral_goal:boundaries.length===0,
      intent_margin_factor:boundaries.length===0?0:margin===null?null:1/(1+Math.max(margin,0)),
      physical_domain_bounds_are_objectives:false,exact_point:point};
  }

  function contribution(normalizedDrive,context){
    if(!finite(normalizedDrive)||context.context_missing)return null;
    if(!context.boundaries.length)return 0;
    return Math.min(...context.active_boundaries.map(b=>b.sign*normalizedDrive));
  }

  function loss(potential,drive){
    if(!finite(potential)||!finite(drive))return null;
    if(potential>0)return clip((potential-drive)/potential);
    return drive<0?1:0;
  }
  function referenceLoss(potential,drive){
    return finite(potential)&&finite(drive)?Math.max(potential-drive,0):null;
  }
  function displayScore(strength){
    return finite(strength)&&strength>=0?100*(strength/(1+strength)):null;
  }

  function projectionGraph(graph){
    const sourceFields=['unit','x_unit','kind','parameter','scalar_component','estimate','confidence_interval','fit_domain','sampled_domain'];
    return {...graph,edges:graph.edges.map(edge=>{
      if(edge.kind!=='influences')return edge;
      const errors=[],rows=(edge.sensitivities?.length?edge.sensitivities:edge.sensitivity?[edge.sensitivity]:[]).map(row=>{
        const units=row.projection_units,source=Object.fromEntries(sourceFields.filter(k=>k in row).map(k=>[k,row[k]]));
        if(!units?.valid||encoded(source)!==encoded(units.source)){
          errors.push({calibration:row.contrast_id,reason:units?.reason||(!units?'missing_unit_contract':'stale_unit_contract')});
          return null;
        }
        return {...row,...units.fields};
      });
      return {...edge,sensitivities:errors.length?[]:rows,sensitivity:errors.length?null:rows[0],unit_alignment_errors:errors};
    })};
  }

  function recordedEmptySchedulerBundle(app,cell,mode,decisionSlot){
    const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
    if(!Number.isInteger(decisionSlot)||!['round_robin','latency_aware','qos_deadline'].includes(mode))return null;
    const decisions=(Array.isArray(app.decisions)?app.decisions:[]).map((decision,index)=>({decision,index})).filter(r=>r.decision?.slot_id===decisionSlot);
    const evidence=decisions.filter(r=>object(r.decision.snapshot)&&Object.prototype.hasOwnProperty.call(r.decision.snapshot,cell));
    if(!evidence.length)return null;
    // An explicit same-slot membership or snapshot disagreement is not an
    // empty-cell receipt. Never resolve it by choosing a convenient record.
    if(decisions.some(r=>object(r.decision.membership)&&Object.values(r.decision.membership).includes(cell)))return null;
    let referenceMembership=null;
    for(const {decision}of evidence){
      const config=decision.snapshot[cell],membership=decision.membership;
      if(!object(config)||config.scheduler_mode!==mode||!object(config.ue_scheduler_weights)||
        Object.keys(config.ue_scheduler_weights).length||!object(membership)||
        Object.values(membership).some(c=>typeof c!=='string'||!c)||
        decision.implementation_id&&'xapp:'+decision.implementation_id!==app.app_id)return null;
      if(referenceMembership!==null&&encoded(membership)!==encoded(referenceMembership))return null;
      referenceMembership=membership;
      for(const action of app.actions||[]){
        if(action.parameter==='ue_scheduler_weights'&&(!action.ue_id||membership[action.ue_id]!==action.cell_id))return null;
        const snapshot=decision.snapshot[action.cell_id];
        if(!object(snapshot))continue;
        if(action.parameter==='scheduler_mode'&&'scheduler_mode'in snapshot&&snapshot.scheduler_mode!==action.proposed)return null;
        if(action.parameter==='ue_scheduler_weights'&&object(snapshot.ue_scheduler_weights)&&
          encoded(snapshot.ue_scheduler_weights[action.ue_id])!==encoded(action.proposed))return null;
      }
      for(const [cid,snapshot]of Object.entries(decision.snapshot)){
        if(!object(snapshot)||!Object.prototype.hasOwnProperty.call(snapshot,'ue_scheduler_weights'))continue;
        if(!object(snapshot.ue_scheduler_weights))return null;
        const observed=Object.keys(snapshot.ue_scheduler_weights).sort();
        const expected=Object.keys(membership).filter(uid=>membership[uid]===cid).sort();
        if(encoded(observed)!==encoded(expected))return null;
        for(const uid of observed)if(!(app.actions||[]).some(action=>action.parameter==='ue_scheduler_weights'&&
          action.cell_id===cid&&action.ue_id===uid&&encoded(action.proposed)===encoded(snapshot.ue_scheduler_weights[uid])))return null;
      }
    }
    return {source:'recorded_shadow_decision_snapshot',app_id:app.app_id,cell_id:cell,
      parameter:'ue_scheduler_weights',value:{},decision_slot:decisionSlot,
      decision_indices:evidence.map(r=>r.index),membership:clone(referenceMembership),
      members_in_cell:0,flat_ue_actions_added:0};
  }

  function units(app,decisionSlot){
    const found=new Map();
    for(const action of app.actions||[]){
      const parameter=action.parameter,entity=action.ue_id||action.cell_id;
      const resource=scheduler.has(parameter)?['scheduler_bundle',action.cell_id]:[parameter,entity];
      const key=encoded(resource);
      if(!found.has(key))found.set(key,{resource,app_id:app.app_id,actions:[],value:{}});
      const unit=found.get(key);unit.actions.push(action);
      if(scheduler.has(parameter)){
        if(parameter==='scheduler_mode')unit.value.scheduler_mode=action.proposed;
        else (unit.value.ue_scheduler_weights??={})[action.ue_id]=action.proposed;
      }else unit.value=clone(action.proposed);
    }
    for(const unit of found.values())if(unit.resource[0]==='scheduler_bundle'&&
      'scheduler_mode'in unit.value&&!('ue_scheduler_weights'in unit.value)){
      const receipt=recordedEmptySchedulerBundle(app,unit.resource[1],unit.value.scheduler_mode,decisionSlot);
      if(receipt){unit.value.ue_scheduler_weights={};unit.recovered_empty_scheduler_bundles=[receipt];}
    }
    return [...found.values()];
  }

  function outcomes(ids,apps,limit,decisionSlot){
    const resources=new Map();
    for(const id of ids)for(const unit of units(apps.get(id),decisionSlot)){
      const key=encoded(unit.resource);
      if(!resources.has(key))resources.set(key,[]);
      resources.get(key).push(unit);
    }
    const choices=[],details=[];
    for(const [key,owners]of [...resources].sort(([a],[b])=>a.localeCompare(b))){
      const variants=new Map();
      for(const unit of owners){
        const signature=encoded(unit.value);
        if(!variants.has(signature))variants.set(signature,{actions:unit.actions,probability:0,owners:[]});
        const variant=variants.get(signature);variant.probability+=1/owners.length;variant.owners.push(unit.app_id);
      }
      const type=owners[0].resource[0];
      if(variants.size>1&&type!=='ue_offload_fraction'&&type!=='scheduler_bundle')
        return {error:'unsupported_arbitration_resource',resource:JSON.parse(key)};
      details.push({resource:JSON.parse(key),requesting_apps:owners.map(x=>x.app_id),distinct_values:variants.size,
        recovered_empty_scheduler_bundles:owners.flatMap(x=>x.recovered_empty_scheduler_bundles||[])});
      choices.push([...variants.values()]);
    }
    const count=choices.reduce((n,c)=>n*c.length,1);
    if(count>limit)return {error:'arbitration_outcome_limit',count};
    let rows=[{actions:[],probability:1,winners:[]}];
    for(const options of choices)rows=rows.flatMap(row=>options.map(option=>({
      actions:[...row.actions,...option.actions],probability:row.probability*option.probability,
      winners:[...row.winners,option.owners]})));
    return {rows,resources:details,count};
  }

  function review(graph,frame,groups,ranges,preview,options={}){
    const selectedMethod=options.method??method;
    if(!['goal_obstruction_v1','goal_obstruction_v2'].includes(selectedMethod))throw Error('Unsupported goal-obstruction method');
    const strengthMethod=selectedMethod==='goal_obstruction_v2';
    const reference=options.referenceEngine||(typeof SmoConflictRisk==='undefined'?null:SmoConflictRisk);
    if(!reference)throw Error('Reference projection engine is required');
    const originalGraph=graph;
    const uniqueRanges=[...new Map(ranges.map(r=>[r.kpi,r])).values()];
    const intentFrame=options.projectionOnly&&options.intentFrame?options.intentFrame:frame;
    const contexts=uniqueRanges.map(r=>goalContext(r,intentFrame,reference,options.normalization?.kpis.find(k=>k.kpi===r.kpi)));
    const baseResult={method:selectedMethod,schema:strengthMethod?'oran.goal-obstruction.v2':'oran.goal-obstruction.v1',status:'ready',
      snapshot:{slot:frame.slot,time_ms:frame.time_ms},preview_slot:preview.slot,groups:[],
      intent_snapshot:{slot:intentFrame.slot,time_ms:intentFrame.time_ms},
      score_semantics:strengthMethod?'fixed_monotone_display_of_margin_adjusted_reference_obstruction_strength':'fraction_of_available_positive_reference_control_capability_lost',
      score_mapping:strengthMethod?'100*B/(1+B)':'100*maximum_expected_loss_fraction',
      intent_margin_factor:strengthMethod?'1/(1+max(normalized_current_intent_margin,0))':'not_used',
      kpi_forecast_performed:false,observed_after_kpis_used:false,selection_performed:false,
      deployment_enabled:false,arbitration_performed:false,
      arbitration_evaluated:'exact_fresh_uniform_request_distribution_without_random_draw',
      probability_interpretation:'arbitration_outcomes_only_not_intent_failure_probability',
      temporal_scope:'one_frozen_decision_not_a_closed_loop_window_prediction',
      influence_alignment_version:1,kpi_value_weights:'equal',normalized_drive_unit:'dimensionless',
      normalization:options.normalization||null,intent_contexts:contexts,
      influence_scope:graph.statistical_filter?.enabled?'significant_relations_only':'all_available_relations',
      excluded_relations_are_zero_physical_effects:false,screened_gain_values_used_for_score:false,
      action_context_policy:graph.statistical_filter?.action_context_policy||'available_contexts',
      conditional_gain_uncertainty_preserved:true,projection_only:!!options.projectionOnly};
    if(preview?.slot!==frame?.slot)return {...baseResult,status:'unavailable',reason:'matching_decision_preview_unavailable'};
    if(!contexts.length)return {...baseResult,status:'unavailable',reason:'intent_kpis_unavailable'};
    if(contexts.some(c=>c.empty))return {...baseResult,status:'unavailable',reason:'empty_intent_interval'};
    if(options.arbitration?.mode&&options.arbitration.mode!=='uniform_random')
      return {...baseResult,status:'unavailable',reason:'unsupported_mitigation_mode'};
    if(contexts.some(c=>!finite(c.scale)||c.scale<=0))
      return {...baseResult,status:'unavailable',reason:'kpi_scale_unavailable'};
    if(contexts.some(c=>c.context_missing))
      return {...baseResult,status:'unavailable',reason:'kpi_observation_unavailable'};
    graph=projectionGraph(graph);
    const legacy=reference.review(graph,frame,groups,ranges,preview,{showDashed:!!options.showDashed,normalizationContexts:contexts});
    if(legacy.status!=='ready')return {...baseResult,status:legacy.status,reason:legacy.reason};
    const original=options.projectionOnly?{groups:[]}:
      reference.review(originalGraph,frame,groups,ranges,preview,{showDashed:!!options.showDashed});
    const kpiUnit=k=>graph.nodes?.find(n=>n.id==='kpi:'+k)?.unit;
    const apps=new Map((preview.apps||[]).map(a=>[a.app_id,a]));
    const base={};
    for(const e of graph.edges.filter(e=>e.kind==='controls'))base[e.target.slice(4)]=clone(frame.states?.[e.target]?.by_entity||{});
    for(const app of apps.values())for(const a of app.actions||[]){
      const entity=a.ue_id||a.cell_id;
      if(base[a.parameter]&&base[a.parameter][entity]==null&&a.current!=null)base[a.parameter][entity]=clone(a.current);
    }
    const effectiveFrame={...frame,states:{...frame.states}};
    for(const [p,by_entity]of Object.entries(base))effectiveFrame.states['rcp:'+p]={by_entity};
    const cache=new Map(),limit=options.maxOutcomes??4096,referenceLimit=options.maxReferenceOutcomes??65536;
    const excludedRelations=new Set((graph.statistical_filter?.excluded_relations||[]).map(e=>e.source+'|'+e.target));
    const partialRelations=new Map((graph.statistical_filter?.partially_filtered_relations||[]).map(e=>[e.source+'|'+e.target,e]));
    function project(actions){
      const target=clone(base);
      for(const a of actions)target[a.parameter][a.ue_id||a.cell_id]=clone(a.proposed);
      const signature=encoded(target);
      if(cache.has(signature))return cache.get(signature);
      const changed=Object.keys(base).filter(p=>encoded(base[p])!==encoded(target[p]));
      const goals=contexts.map(context=>{
        let drive=0,known=true;const chains=[],excluded=[],excludedContexts=[];
        for(const p of changed){
          const edge=graph.edges.find(e=>e.kind==='influences'&&e.source==='rcp:'+p&&e.target==='kpi:'+context.kpi);
          if(!edge&&excludedRelations.has('rcp:'+p+'|kpi:'+context.kpi)){
            excluded.push(p);continue; // Omitted from this KG model, not a measured zero.
          }
          if(!edge){known=false;chains.push({parameter:p,drive:null,notes:['reference_direction_unavailable']});continue;}
          const projected=edge.unit_alignment_errors?.length?
            {drive:null,notes:['incompatible_or_missing_gain_units'],unit_alignment_errors:edge.unit_alignment_errors}:
            reference.project(edge,base[p],target[p],effectiveFrame,!!options.showDashed,{target,actions});
          // A screened-out measured action is absent from the active model,
          // just like a wholly screened P–K edge. It is not an unmeasured
          // action and its discarded coefficient must never enter the score.
          const audit=partialRelations.get(edge.source+'|'+edge.target);
          const screened=!finite(projected.drive)&&!edge.unit_alignment_errors?.length&&audit&&reference.screenedProjection?.(audit,edge,base[p],target[p],effectiveFrame);
          if(screened?.excluded){
            excluded.push(p);excludedContexts.push({parameter:p,calibrations:screened.calibrations,reason:'measured_action_screened_out'});
            continue;
          }
          if(finite(projected.drive)&&projected.action_coverage?.complete!==false)drive+=projected.drive;else known=false;
          chains.push({parameter:p,...projected,reference_drive_unit:kpiUnit(context.kpi),
            normalized_drive_unit:'dimensionless',kpi_scale:context.scale,
            unit_contracts:(edge.sensitivities||[]).filter(r=>projected.calibrations?.includes(r.contrast_id)).map(r=>r.projection_units)});
        }
        const normalized=known?drive/context.scale:null;
        return {kpi:context.kpi,reference_drive:known?drive:null,normalized_drive:normalized,
          goal_drive:contribution(normalized,context),chains,excluded_parameters:excluded,excluded_action_contexts:excludedContexts};
      });
      const result={goals};cache.set(signature,result);return result;
    }
    for(const group of groups){
      const prior=legacy.groups.find(g=>g.group_id===(group.group_id||group.id));
      const ids=[...group.app_ids].sort(),id=group.group_id||group.id;
      const missing=prior?.coverage.missing_proposal_apps||ids.filter(a=>!apps.has(a));
      const output={...prior,group_id:id,app_ids:group.app_ids,
        legacy_score:original.groups?.find(g=>g.group_id===id)?.score,score:null,
        chains:(prior?.chains||[]).map(c=>({...c,reference_drive_unit:kpiUnit(c.kpi),
          normalized_drive_unit:'dimensionless',kpi_scale:contexts.find(k=>k.kpi===c.kpi)?.scale,value_weight:1})),
        score_aggregation:strengthMethod?'maximum_over_required_kpis_of_margin_adjusted_expected_reference_loss':'maximum_over_required_kpis_of_expected_reference_capability_loss',
        score_status:'insufficient_evidence',goal_components:[],reference_subset_count:0,
        reference_outcome_count:0,actual_outcomes:[],no_active_push:false};
      if(strengthMethod){
        // Pairwise cancellation attribution belongs to v4, not to the new
        // joint-execution strength. Do not export it as v2 score attribution.
        delete output.contributors;delete output.interactions;
      }
      if(missing.length){output.reason='proposal_unavailable';baseResult.groups.push(output);continue;}
      const invalidBundle=ids.some(a=>units(apps.get(a),frame.slot).some(u=>u.resource[0]==='scheduler_bundle'&&
        (!('scheduler_mode'in u.value)||!('ue_scheduler_weights'in u.value))));
      if(invalidBundle){output.reason='incomplete_scheduler_bundle';baseResult.groups.push(output);continue;}
      if(!options.projectionOnly&&ids.length>16){output.reason='reference_space_limit';baseResult.groups.push(output);continue;}
      const full=outcomes(ids,apps,limit,frame.slot);
      if(full.error){output.reason=full.error;baseResult.groups.push(output);continue;}
      const potentials=Object.fromEntries(contexts.map(c=>[c.kpi,0])),projectionAudit=new Map();let incomplete=false,stopped=false;
      const auditGoal=goal=>{
        for(const chain of goal.chains){
          const status=finite(chain.drive)&&chain.action_coverage?.complete!==false?'supported':'unknown',key=goal.kpi+'|'+chain.parameter+'|'+status;
          projectionAudit.set(key,{kpi:goal.kpi,parameter:chain.parameter,status,notes:chain.notes||[],action_coverage:chain.action_coverage});
        }
        for(const parameter of goal.excluded_parameters)projectionAudit.set(goal.kpi+'|'+parameter+'|excluded',{
          kpi:goal.kpi,parameter,status:'excluded',action_context:goal.excluded_action_contexts.find(c=>c.parameter===parameter)||null});
      };
      // Subsets remove complete xApp policies. Individual policy fields are
      // never cherry-picked to fabricate the positive reference envelope.
      for(let mask=0;!options.projectionOnly&&mask<2**ids.length;mask++){
        const subset=ids.filter((_,i)=>mask&(2**i)),branch=outcomes(subset,apps,limit,frame.slot);
        if(branch.error){incomplete=true;continue;}
        output.reference_subset_count++;
        output.reference_outcome_count+=branch.count;
        if(output.reference_outcome_count>referenceLimit){stopped=true;break;}
        for(const state of branch.rows)for(const goal of project(state.actions).goals){
          auditGoal(goal);
          if(goal.goal_drive==null)incomplete=true;
          else potentials[goal.kpi]=Math.max(potentials[goal.kpi],goal.goal_drive,0);
        }
      }
      if(stopped){output.reason='reference_space_limit';baseResult.groups.push(output);continue;}
      let partial=false;
      for(const state of full.rows){
        const goals=project(state.actions).goals.map(g=>{
          auditGoal(g);
          if(g.goal_drive==null)incomplete=true;
          if(g.chains.some(c=>c.notes?.some(n=>n!=='local_reference_drive_not_kpi_change')))partial=true;
          if(g.excluded_parameters.length)partial=true;
          const positive=finite(g.goal_drive)?Math.max(g.goal_drive,0):null;
          return {...g,potential:potentials[g.kpi],positive_gain:positive,
            obstruction:finite(g.goal_drive)?Math.max(-g.goal_drive,0):null,
            lost_positive_capability:positive===null?null:Math.max(potentials[g.kpi]-positive,0),
            reference_loss_amount:referenceLoss(potentials[g.kpi],g.goal_drive),
            loss_fraction:loss(potentials[g.kpi],g.goal_drive)};
        });
        output.actual_outcomes.push({probability:state.probability,winners:state.winners,goals});
      }
      output.arbitration_resources=full.resources;
      output.recovered_empty_scheduler_bundles=full.resources.flatMap(r=>r.recovered_empty_scheduler_bundles);
      const coverage=[...projectionAudit.values()];
      output.projection_coverage={supported_links:coverage.filter(r=>r.status==='supported'),
        excluded_links:coverage.filter(r=>r.status==='excluded'),unknown_links:coverage.filter(r=>r.status==='unknown'),
        kpis:contexts.map(c=>({kpi:c.kpi,supported:coverage.some(r=>r.kpi===c.kpi&&r.status==='supported'),
          excluded:coverage.some(r=>r.kpi===c.kpi&&r.status==='excluded'),
          unknown:coverage.some(r=>r.kpi===c.kpi&&r.status==='unknown')}))};
      output.projection_coverage.covered_kpis=output.projection_coverage.kpis.filter(r=>r.supported).length;
      output.projection_coverage.required_kpis=contexts.length;
      const probability=full.rows.reduce((s,r)=>s+r.probability,0);
      for(const context of contexts){
        const records=output.actual_outcomes.map(r=>({probability:r.probability,goal:r.goals.find(g=>g.kpi===context.kpi)}));
        const mean=key=>records.every(r=>finite(r.goal[key]))?records.reduce((s,r)=>s+r.probability*r.goal[key],0)/probability:null;
        const expectedLoss=mean('reference_loss_amount');
        output.goal_components.push({kpi:context.kpi,potential:potentials[context.kpi],
          expected_goal_drive:mean('goal_drive'),expected_positive_gain:mean('positive_gain'),
          expected_obstruction:mean('obstruction'),expected_lost_positive_capability:mean('lost_positive_capability'),
          expected_loss_fraction:mean('loss_fraction'),
          expected_reference_loss:expectedLoss,
          intent_margin_factor:context.intent_margin_factor,neutral_goal:context.neutral_goal,value_weight:1,
          obstruction_strength:expectedLoss===null||context.intent_margin_factor===null?null:expectedLoss*context.intent_margin_factor,
          no_active_push:potentials[context.kpi]===0&&records.every(r=>r.goal.goal_drive===0)});
      }
      output.no_active_push=output.goal_components.every(c=>c.no_active_push);
      output.held_lease_state_not_reconstructed=ids.some(a=>frame.states?.[a]?.enabled===true);
      output.fractional_score=incomplete?null:100*Math.max(...output.goal_components.map(c=>c.expected_loss_fraction));
      output.raw_obstruction_strength=incomplete?null:Math.max(...output.goal_components.map(c=>c.obstruction_strength));
      output.score=strengthMethod?displayScore(output.raw_obstruction_strength):output.fractional_score;
      // Completeness is relative to the selected P–K graph. Intentionally
      // absent relations do not invalidate an otherwise covered goal; a
      // missing conditional action or an entirely uncovered goal still does.
      output.model_coverage_complete=!output.projection_coverage.unknown_links.length&&
        !output.projection_coverage.excluded_links.some(r=>r.action_context)&&
        !output.projection_coverage.kpis.some(r=>r.excluded&&!r.supported);
      output.score_status=incomplete?'insufficient_evidence':!output.model_coverage_complete?'partial_model_score':partial||output.held_lease_state_not_reconstructed?'reference_approximation':output.no_active_push?'no_active_push':'reference_score';
      output.reason=incomplete?'reference_projection_incomplete':null;
      if(options.projectionOnly){
        // Research methods can reuse the audited action/gain/arbitration
        // projector without inventing a positive policy-subset envelope.
        // Suppress V2 scores: only the frozen outcome vectors are requested.
        output.score=null;output.raw_obstruction_strength=null;output.fractional_score=null;
        output.score_status='projection_only';
      }
      output.arbitration_probability_mass=probability;
      baseResult.groups.push(output);
    }
    return {...baseResult,intent_contexts:contexts,projected_parameter_states:cache.size};
  }
  function projectOutcomes(graph,frame,groups,ranges,preview,options={}){
    return review(graph,frame,groups,ranges,preview,{...options,method:'goal_obstruction_v2',projectionOnly:true});
  }
  return {method,review,projectOutcomes,goalContext,contribution,loss,referenceLoss,displayScore};
})();
