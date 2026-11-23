/* Additional forward-only reference method. Original methods stay frozen. */
var SmoAggregationAwareObstruction = (() => {
  'use strict';
  const method='aggregation_aware_obstruction_v1';
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
  const displayScore=d=>finite(d)&&d>=0?100*(d/(1+d)):null;
  const clamp=(x,domain)=>Math.max(domain[0]===null?-Infinity:domain[0],Math.min(domain[1]===null?Infinity:domain[1],x));

  function observation(row,frame,normalization){
    const domain=row.domain||[0,null],low=row.lower??null,high=row.upper??null;
    if(!Array.isArray(domain)||domain.length!==2||domain.some(x=>x!==null&&!finite(x))||
       finite(domain[0])&&finite(domain[1])&&domain[0]>domain[1])
      return {reason:'invalid_physical_domain'};
    if(row.empty||low!==null&&!finite(low)||high!==null&&!finite(high)||
       finite(low)&&finite(high)&&(low>high||low===high&&(row.lower_inclusive===false||row.upper_inclusive===false)))
      return {reason:'empty_or_invalid_intent_interval'};
    const current=normalization?.reference_value,scale=normalization?.scale;
    if(!finite(current))return {reason:'kpi_observation_unavailable'};
    if(!finite(scale)||scale<=0)return {reason:'kpi_scale_unavailable'};
    const state=frame.states?.['kpi:'+row.kpi]||{},nested=object(state.measurement)?state.measurement:{};
    if(state.reason||nested.reason)return {reason:'kpi_observation_unavailable'};
    for(const container of [state,nested])if(Object.hasOwn(container,'constraint_uses_extrema')&&typeof container.constraint_uses_extrema!=='boolean')
      return {reason:'invalid_observed_aggregation_flag'};
    const containers=[state,nested].filter(c=>c.constraint_uses_extrema===true);
    let before=[current,current],aggregation='probability_weighted_joint_reference_mean';
    if(containers.length){
      if(containers.some(c=>!finite(c.minimum)||!finite(c.maximum)||c.minimum>c.maximum))
        return {reason:'observed_window_extrema_unavailable'};
      if(containers.some(c=>c.minimum!==containers[0].minimum||c.maximum!==containers[0].maximum))
        return {reason:'contradictory_observed_window_extrema'};
      before=[containers[0].minimum,containers[0].maximum];
      aggregation='baseline_range_by_joint_action_support_envelope';
    }
    if([current,...before].some(x=>finite(domain[0])&&x<domain[0]||finite(domain[1])&&x>domain[1]))
      return {reason:'observed_reference_outside_physical_domain'};
    const point=finite(low)&&finite(high)&&low===high,boundaries=[];
    if(finite(low)&&(point||low!==domain[0]||row.lower_inclusive===false))boundaries.push({side:'lower',bound:low,sign:1});
    if(finite(high)&&(point||high!==domain[1]||row.upper_inclusive===false))boundaries.push({side:'upper',bound:high,sign:-1});
    return {kpi:row.kpi,current,scale,domain,before,aggregation,boundaries,uses_extrema:containers.length>0,
      physical_domain_bounds_are_objectives:false};
  }

  function margin(context,interval){
    return context.boundaries.length?Math.min(...context.boundaries.map(b=>b.side==='lower'?interval[0]-b.bound:b.bound-interval[1])):0;
  }

  function progress(context,after){
    const beforeMargin=margin(context,context.before),afterMargin=margin(context,after);
    const beforeDistance=Math.max(0,-beforeMargin),afterDistance=Math.max(0,-afterMargin);
    const raw=context.boundaries.length===0?0:context.boundaries.length===1?
      afterMargin-beforeMargin:beforeDistance-afterDistance;
    const normalized=raw/context.scale;
    return {reference_progress:finite(normalized)?normalized:null,
      mode:context.boundaries.length===0?'neutral_physical_domain':context.boundaries.length===1?
        'single_boundary_margin_change':'interval_distance_reduction',
      before_reference_margin:beforeMargin,after_reference_margin:afterMargin,
      before_reference_distance:beforeDistance,after_reference_distance:afterDistance,
      strict_endpoints_scored_as_distance_to_closure:true};
  }

  function aggregateEffects(effects){
    if(!effects.length||!effects.every(finite))return {reference_effect:null,obstruction:null};
    const effect=effects.some(x=>x<0)?Math.min(...effects):effects.reduce((s,x)=>s+x,0)/effects.length;
    return {reference_effect:effect,obstruction:Math.max(0,-effect)};
  }

  function review(graph,frame,groups,ranges,preview,options={}){
    if(options.method&&options.method!==method)throw Error('Unsupported aggregation-aware method');
    const engine=options.projectionEngine||(typeof SmoGoalObstruction==='undefined'?null:SmoGoalObstruction);
    const coordinates=options.normalizationEngine||(typeof SmoObservationNormalization==='undefined'?null:SmoObservationNormalization);
    const base={schema:'oran.aggregation-aware-obstruction.v1',method,status:'ready',groups:[],
      snapshot:{slot:frame?.slot,time_ms:frame?.time_ms},preview_slot:preview?.slot,
      score_mapping:'100*D/(1+D)',score_semantics:'worst_negative_aggregation_aware_reference_intent_progress',
      reference_effect_unit:'dimensionless_normalized_progress',reference_effect_score_multiplier:100,
      kpi_value_weights:'equal',normalized_drive_unit:'dimensionless',
      aggregation_order:'complete_joint_reference_then_physical_projection_then_observed_aggregation_then_intent_functional',
      extrema_reference:'conservative_cartesian_observed_range_by_current_joint_action_support',
      finite_window_endpoint_occurrence_guaranteed:false,intent_failure_probability_computed:false,
      kpi_forecast_performed:false,observed_after_kpis_used:false,deployment_results_used:false,
      fitted_score_parameters:false,selection_performed:false,deployment_enabled:false,arbitration_performed:false,
      arbitration_evaluated:'exact_fresh_uniform_request_distribution_without_random_draw',
      temporal_scope:'one_frozen_joint_action_distribution',history_ensemble_used:false,horizon_scaling_performed:false,
      excluded_relations_are_zero_physical_effects:false,conditional_gain_uncertainty_preserved:true};
    const unavailable=reason=>({...base,status:'unavailable',reason});
    if(!engine?.projectOutcomes)return unavailable('reference_projector_unavailable');
    if(!coordinates?.normalization)return unavailable('current_observation_normalization_unavailable');
    if(!frame||!preview||preview.slot!==frame.slot)return unavailable('matching_decision_preview_unavailable');
    if(preview.deployment_enabled===true||finite(preview.observation_slot)&&preview.observation_slot>preview.slot)
      return unavailable('invalid_shadow_preview');
    if(!ranges.length)return unavailable('intent_kpis_unavailable');
    if(new Set(ranges.map(r=>r.kpi)).size!==ranges.length)return unavailable('duplicate_intent_kpis');
    const normalization=coordinates.normalization(ranges,frame);base.normalization=normalization;
    const contexts=ranges.map(r=>observation(r,frame,normalization.kpis.find(k=>k.kpi===r.kpi)));
    base.observation_errors=contexts.map((c,i)=>c.reason?{kpi:ranges[i].kpi,reason:c.reason}:null).filter(Boolean);
    if(base.observation_errors.length)return unavailable(base.observation_errors[0].reason);
    base.intent_contexts=contexts;
    const projected=engine.projectOutcomes(graph,frame,groups,ranges,preview,
      {...options,method:'goal_obstruction_v2',normalization,intentFrame:frame});
    if(projected.status!=='ready')return unavailable(projected.reason||'reference_projection_unavailable');
    for(const group of groups){
      const id=group.group_id||group.id,record=projected.groups?.find(g=>g.group_id===id);
      const output={group_id:id,app_ids:[...group.app_ids],score:null,raw_obstruction_strength:null,
        reference_effect:null,reference_effect_score:null,score_status:'insufficient_evidence',reason:null,
        goal_components:[],reference_subset_count:0,arbitration_outcome_count:record?.actual_outcomes?.length||0,
        projection_coverage:record?.projection_coverage||null,model_coverage_complete:record?.model_coverage_complete===true,
        held_lease_state_not_reconstructed:!!record?.held_lease_state_not_reconstructed,
        structural_counts:record?.structural_counts||null,
        recovered_empty_scheduler_bundles:record?.recovered_empty_scheduler_bundles||[]};
      const fail=reason=>{output.reason=reason;base.groups.push(output);};
      if(!record?.actual_outcomes?.length){fail(record?.reason||'proposal_unavailable');continue;}
      if(!output.model_coverage_complete||record.projection_coverage?.unknown_links?.length||
         record.projection_coverage?.excluded_links?.some(c=>c.action_context)){
        fail('reference_model_or_action_coverage_incomplete');continue;
      }
      const outcomes=record.actual_outcomes;
      if(outcomes.some(o=>!finite(o.probability)||o.probability<0)){
        fail('invalid_arbitration_probability_mass');continue;
      }
      const positive=outcomes.filter(o=>o.probability>0),mass=positive.reduce((s,o)=>s+o.probability,0);
      if(!finite(mass)||mass<=0){fail('invalid_arbitration_probability_mass');continue;}
      let reason=null,approximation=false;
      for(const context of contexts){
        let lower=Infinity,upper=-Infinity,centeredMean=0,clipped=0,zero=true;
        for(const outcome of positive){
          const goals=(outcome.goals||[]).filter(g=>g.kpi===context.kpi),goal=goals[0];
          if(goals.length!==1||!finite(goal.reference_drive)||goal.chains?.some(c=>c.action_coverage?.complete===false||c.supported===false)){
            reason='reference_projection_incomplete';break;
          }
          if(goal.excluded_action_contexts?.length){reason='reference_model_or_action_coverage_incomplete';break;}
          const delta=goal.reference_drive;zero=zero&&delta===0;
          const rawLow=context.before[0]+delta,rawHigh=context.before[1]+delta;
          const lo=clamp(rawLow,context.domain),hi=clamp(rawHigh,context.domain);
          if(!finite(lo)||!finite(hi)||lo>hi){reason='nonfinite_joint_reference_projection';break;}
          clipped+=Number(lo!==rawLow||hi!==rawHigh);
          if(context.uses_extrema){lower=Math.min(lower,lo);upper=Math.max(upper,hi);}
          else centeredMean+=outcome.probability*(lo-context.current)/mass;
          if(goal.excluded_parameters?.length||goal.chains?.some(c=>c.notes?.some(n=>n!=='local_reference_drive_not_kpi_change')))
            approximation=true;
        }
        if(reason)break;
        const after=context.uses_extrema?[lower,upper]:[context.current+centeredMean,context.current+centeredMean];
        if(!after.every(finite)){reason='nonfinite_joint_reference_projection';break;}
        const effect=progress(context,after);
        if(!finite(effect.reference_progress)){reason='nonfinite_reference_effect';break;}
        output.goal_components.push({kpi:context.kpi,...effect,aggregation:context.aggregation,
          kpi_scale:context.scale,value_weight:1,baseline_reference_interval:[...context.before],
          reference_interval:after,positive_probability_outcomes:positive.length,arbitration_probability_mass:mass,
          physically_clipped_joint_outcomes:clipped,physical_saturation_applied:clipped>0,
          normalized_obstruction:Math.max(0,-effect.reference_progress),no_active_push:zero});
      }
      if(reason){fail(reason);continue;}
      const effect=aggregateEffects(output.goal_components.map(c=>c.reference_progress));
      output.reference_effect=effect.reference_effect;output.reference_effect_score=100*effect.reference_effect;
      output.raw_obstruction_strength=effect.obstruction;output.score=displayScore(effect.obstruction);
      output.no_active_push=output.goal_components.every(c=>c.no_active_push);
      output.score_status=approximation||output.held_lease_state_not_reconstructed||contexts.some(c=>c.uses_extrema)?
        'reference_approximation':output.no_active_push?'no_active_push':'reference_score';
      base.groups.push(output);
    }
    return base;
  }
  return {method,review,observation,margin,progress,aggregateEffects,displayScore};
})();
