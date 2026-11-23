"""Active KG projection from pointwise independent-seed confidence intervals.

Never edits estimates, refits data, uses deployments or treats exclusion as a
proof of zero physical effect. A lower screening confidence may recompute an
interval from its audited standard error; the original evidence is preserved.
"""
from collections import Counter
from copy import deepcopy
import math

METHOD='pointwise_ci_nonzero_v1'
# Two-sided Student critical values, df = independent seed count - 1. These
# match the 95% intervals emitted by identify.paired_summary (2--10 seeds).
_T95=(12.706204736432095,4.302652729696142,3.182446305284263,2.776445105197799,
      2.570581835636305,2.446911851144969,2.3646242510103,2.306004135204166,2.2621571628540993)
_T90=(6.313751514675043,2.919985580355516,2.3533634348018264,2.13184678632665,
      2.0150483733330233,1.9431802805153022,1.894578605061305,1.8595480375228424,1.8331129326536335)


def screening_estimate(row, confidence_level):
    """Recompute only a verifiable seed-Student interval, never slot pseudo-n.

    If provenance is insufficient the original, stricter interval remains in
    force. Missing SE or incompatible intervals cannot gain admission by fiat.
    """
    result=deepcopy(row)
    if confidence_level!=.90 or row.get('confidence_level')!=.95:return result
    n=row.get('sample_count');se=row.get('standard_error');value=row.get('estimate');ci=row.get('confidence_interval')
    finite=lambda x:type(x)in(int,float)and math.isfinite(x)
    if (type(n)is not int or not 2<=n<=10 or row.get('independent_unit')not in('seed_pair','seed_block')
        or not finite(se)or se<0 or not finite(value)or not isinstance(ci,(list,tuple))or len(ci)!=2
        or not all(map(finite,ci))):return result
    expected=[value-_T95[n-2]*se,value+_T95[n-2]*se]
    if not all(math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-10)for a,b in zip(ci,expected)):return result
    result['confidence_recalculation']=dict(method='seed_student_t_from_verified_standard_error',
        source_confidence_level=.95,source_confidence_interval=deepcopy(ci),
        degrees_of_freedom=n-1,independent_unit=row['independent_unit'],standard_error=se,
        critical_value=_T90[n-2],refitted=False)
    result['confidence_level']=.90
    result['confidence_interval']=[value-_T90[n-2]*se,value+_T90[n-2]*se]
    return result


def reason(row, *, population=False, minimum_confidence_level=.95):
    if population and row.get('population_control_eligible') is not True:
        return 'population_evidence_unavailable'
    value=row.get('estimate');ci=row.get('confidence_interval');level=row.get('confidence_level')
    finite=lambda x:type(x)in(int,float)and math.isfinite(x)
    if not finite(value):return 'scalar_estimate_unavailable'
    if not isinstance(ci,(list,tuple))or len(ci)!=2 or not all(map(finite,ci))or ci[0]>ci[1]:
        return 'confidence_interval_unavailable'
    if not finite(level)or level<minimum_confidence_level or level>=1:return 'confidence_level_unverified'
    if row.get('zero_in_both')or row.get('fit_diagnostics',{}).get('assessment')=='constant_response_observed':
        return 'observed_zero'
    if ci[0]<=0<=ci[1]:return 'observed_zero'if value==0 and ci==[0,0]else'interval_includes_zero'
    if not ci[0]<=value<=ci[1]:return 'estimate_outside_interval'
    return None


def filter_graph(graph, *, confidence_level=.95, retain_measured_contexts=False):
    if confidence_level not in(.90,.95):raise ValueError('Screening confidence must be 0.90 or 0.95')
    if type(retain_measured_contexts)is not bool:raise ValueError('retain_measured_contexts must be boolean')
    result=deepcopy(graph);edges=[];excluded=[];partial=[];counts=Counter();total_rows=kept_rows=total_edges=recalculated=0
    significant_rows=uncertain_rows=observed_zero_rows=0
    population=graph.get('metric_scope')=='ue_population.v1'
    kpi_units={n['id']:n.get('unit') for n in graph.get('nodes',[]) if n.get('kind')=='kpi'}
    from smo.influence_units import projection
    for edge in result['edges']:
        if edge['kind']!='influences':edges.append(edge);continue
        total_edges+=1
        rows=edge.get('sensitivities')or([edge['sensitivity']]if edge.get('sensitivity')else[])
        total_rows+=len(rows);kept=[];rejected=[];classified=[]
        for original in rows:
            row=screening_estimate(original,confidence_level)
            recalculated+=int('confidence_recalculation'in row)
            why=reason(row,population=population,minimum_confidence_level=confidence_level)
            units=projection(row,kpi_units.get(edge['target']))
            if retain_measured_contexts and why in (None,'interval_includes_zero','observed_zero'):
                # Establishing a P–K relation and identifying a measured
                # action's conditional gain are different statistical claims.
                # An admitted relation may retain uncertain measured contexts,
                # but neither relation admission nor projection can rely on an
                # invalid unit, contradictory zero flag or malformed interval.
                ci=row['confidence_interval'];value=row['estimate']
                if not units['valid']:why='gain_units_invalid'
                elif not ci[0]<=value<=ci[1]:why='estimate_outside_interval'
                elif why=='observed_zero' and (value!=0 or not ci[0]<=0<=ci[1]):why='observed_zero_inconsistent'
            classified.append((row,why,units))
        admitted=any(why is None for _,why,_ in classified)
        for row,why,units in classified:
            conditional=why in ('interval_includes_zero','observed_zero')
            if why is None or retain_measured_contexts and admitted and conditional:
                if retain_measured_contexts:
                    row['statistical_context']=dict(direction_significant=why is None,
                        confidence_level=row.get('confidence_level'),screening_reason=why,
                        retained_as='significant_reference'if why is None else
                            'observed_zero_reference'if why=='observed_zero'else'uncertain_reference')
                kept.append(row)
                significant_rows+=int(why is None)
                uncertain_rows+=int(why=='interval_includes_zero')
                observed_zero_rows+=int(why=='observed_zero')
                continue
            counts[why]+=1
            # Keep applicability evidence for the omitted action, separately
            # from the active model. It permits distinguishing a screened
            # measurement from an action that was never measured; its fitted
            # coefficient must never be reintroduced into the risk score.
            calibration=deepcopy(row)
            calibration['projection_units']=units
            ci=row.get('confidence_interval');value=row.get('estimate')
            screened_effect_eligible=(why in ('interval_includes_zero','observed_zero') and
                calibration['projection_units']['valid'] and
                isinstance(ci,(list,tuple)) and len(ci)==2 and ci[0]<=value<=ci[1] and ci[0]<=0<=ci[1])
            rejected.append(dict(contrast_id=row.get('contrast_id'),reason=why,
                estimate=deepcopy(row.get('estimate')),confidence_interval=deepcopy(row.get('confidence_interval')),
                confidence_level=row.get('confidence_level'),confidence_recalculation=deepcopy(row.get('confidence_recalculation')),kind=row.get('kind'),
                scalar_component=row.get('scalar_component'),from_value=deepcopy(row.get('from')),to_value=deepcopy(row.get('to')),
                screened_effect_eligible=screened_effect_eligible,calibration=calibration))
        summary=dict(edge_id=edge['id'],source=edge['source'],target=edge['target'],
            excluded_estimates=rejected,source_evidence=edge.get('evidence'))
        if not kept:
            if not rows:counts['uncalibrated_relation']+=1
            summary['reason']='no_significant_estimates'if rows else'uncalibrated_relation'
            excluded.append(summary);continue
        kept_rows+=len(kept)
        edge['sensitivities']=kept;edge['sensitivity']=deepcopy(kept[0])
        edge['sign']='mixed'if any(r['estimate']>0 for r in kept)and any(r['estimate']<0 for r in kept)else'positive'if kept[0]['estimate']>0 else'negative'
        edge['statistical_selection']=dict(method=METHOD,minimum_confidence_level=confidence_level,
            retained_contrasts=[r.get('contrast_id')for r in kept],excluded_estimate_count=len(rejected))
        if retain_measured_contexts:
            edge['statistical_selection'].update(action_context_policy='retain_valid_measured',
                relation_admission='at_least_one_significant_valid_context',
                significant_contrasts=[r.get('contrast_id')for r in kept if r['statistical_context']['direction_significant']],
                uncertain_contrasts=[r.get('contrast_id')for r in kept if r['statistical_context']['retained_as']=='uncertain_reference'],
                observed_zero_contrasts=[r.get('contrast_id')for r in kept if r['statistical_context']['retained_as']=='observed_zero_reference'])
        if rejected:partial.append(summary)
        edges.append(edge)
    result['edges']=edges
    result['statistical_filter']=dict(method=METHOD,enabled=True,minimum_confidence_level=confidence_level,
        interval_method='independent_seed_student_t',recalculated_estimate_count=recalculated,
        unverified_recalculation_policy='retain_original_stricter_interval',
        criterion='pointwise_confidence_interval_excludes_zero',multiple_testing_correction='none_pointwise_ci',
        original_edge_count=total_edges,active_edge_count=total_edges-len(excluded),
        excluded_edge_count=len(excluded),original_estimate_count=total_rows,active_estimate_count=kept_rows,
        excluded_estimate_count=total_rows-kept_rows,reasons=dict(counts),
        excluded_relations=excluded,partially_filtered_relations=partial,
        controls_modified=False,raw_datasets_modified=False,excluded_means_zero_effect=False,
        selection_scope='active_significant_graph_only',deployment_results_used=False)
    if retain_measured_contexts:
        result['statistical_filter'].update(action_context_policy='retain_valid_measured',
            criterion='relation_has_at_least_one_valid_context_with_pointwise_interval_excluding_zero',
            relation_admission='at_least_one_significant_valid_context',
            significant_estimate_count=significant_rows,retained_uncertain_estimate_count=uncertain_rows,
            retained_observed_zero_estimate_count=observed_zero_rows,
            conditional_gain_confidence_distinct_from_relation_admission=True)
    return result
