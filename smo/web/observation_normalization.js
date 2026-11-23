/* Opt-in observation coordinates. No anchor fallback or deployment feedback. */
var SmoObservationNormalization = (() => {
  'use strict';
  const policy='current_observation_v1',policyVersion=1;
  const frameFields=['source_id','source_sha256','run_id','graph_sha256',
    'evaluation_origin_id','window_ms','evaluation_window'];
  const measurementFields=['source','sampled_at_slot','window_start_slot','window_slots',
    'window','aggregation','sample_count','reason','status','unit','canonical_unit','measurement'];
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
  const pick=(value,keys)=>Object.fromEntries(keys.filter(k=>Object.hasOwn(value,k)).map(k=>[k,structuredClone(value[k])]));

  function normalization(ranges,frame){
    const snapshot=Object.fromEntries(['slot','time_ms'].map(k=>[k,finite(frame[k])?frame[k]:null]));
    const states=object(frame.states)?frame.states:{};
    const kpis=ranges.map(row=>{
      const present=object(states['kpi:'+row.kpi]),state=present?states['kpi:'+row.kpi]:{};
      const current=state.value,observed=finite(current);
      let scale=null,basis;
      if(row.kpi==='urllc_failure_fraction'){
        const unit=Object.hasOwn(row,'unit')?row.unit:'ratio';
        scale=['ratio','fraction'].includes(unit)?1:unit==='%'?100:null;
        basis=scale!==null?'ratio_percentage_points':'incompatible_ratio_unit';
      }else if(!observed){basis='current_observation_unavailable';}
      else if(current===0){basis='explicit_scale_required';}
      else {scale=Math.abs(current);basis='current_observation_magnitude';}
      const reason=scale===null?basis:observed?null:'current_observation_unavailable';
      return {kpi:row.kpi,scale,scale_basis:basis,reference_value:observed?current:null,
        weight:1,weight_basis:'equal_kpi_value',status:reason===null?'available':'unavailable',reason,
        scale_available:scale!==null,reference_observation_available:observed,
        observation_provenance:{snapshot:structuredClone(snapshot),state_present:present,metadata:pick(state,measurementFields)}};
    });
    return {schema:'oran.influence-normalization.v1',policy,policy_version:policyVersion,
      value_weights:'equal',reference_basis:'current_observed_frame',reference_selection:'current_frame_only',
      reference_snapshot:snapshot,reference_frame_provenance:pick(frame,frameFields),
      batch_before_reference_pending:false,observed_after_kpis_used:false,deployment_batch_used:false,
      intent_anchors_used_for_scale:false,interval_endpoints_modified:false,kpis};
  }
  return {policy,policyVersion,normalization};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=SmoObservationNormalization;
