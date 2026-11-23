"""Exercise real recorded shadow actions with the updated influence matrices.

Intent intervals below are declared audit probes, not language-model outputs or
user intent records. No simulator, model inference or deployment is invoked.
"""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

from smo.__main__ import GraphService,graph_intent_review
from smo.identify import load,sha,write
from smo.intent_language import resolve_ranges
from smo.runtime import records


RUNS={
    'native_1x':('agx2','shadow_replay_native_1x_s20261120_native'),
    'steady_2x':('agx2','shadow_replay_steady_2x_s20261121_native'),
    'burst_2x':('agx1','shadow_replay_burst_2x_s20261120_native'),
}


def interval(kpi,upper=None,lower='minimum',li=True,ui=True):
    return dict(kpi=kpi,lower=lower,upper=upper,lower_inclusive=li,upper_inclusive=ui)


PROBES={
    'ue_power_and_mec_load':[interval('ue_power_w',{'current_multiplier':.98}),interval('mec_expected_gflops',1.8)],
    'ue_power_and_mec_delay':[interval('ue_power_w',{'current_multiplier':.99}),interval('mec_offloaded_latency_ms',15.)],
    'bs_power_and_mec_delay':[interval('ap_power_w',{'current_multiplier':.98}),interval('mec_offloaded_latency_ms',15.)],
    'downlink_and_urllc':[interval('network_goodput_mbps','maximum',lower='current',li=False),interval('urllc_failure_events_per_s','current',ui=False)],
}


NODE=r'''
const fs=require('fs'),vm=require('vm');const ctx={structuredClone};vm.createContext(ctx);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),ctx);
const d=JSON.parse(fs.readFileSync(0,'utf8')),outputs=[],examples=[];
for(const trial of d.trials){
 const r=ctx.SmoConflictRisk.review(d.graph,trial.frame,trial.groups,trial.ranges,trial.preview,{showDashed:trial.show_dashed});
 if(r.kpi_forecast_performed||r.observed_after_kpis_used||r.selection_performed)throw Error('Risk/feedback boundary changed');
 if(examples.length<3){const g=r.groups.find(g=>g.score>0);if(g)examples.push({probe:trial.probe,slot:trial.frame.slot,show_dashed:trial.show_dashed,group:g});}
 outputs.push({probe:trial.probe,slot:trial.frame.slot,show_dashed:trial.show_dashed,status:r.status,reason:r.reason,
  groups:r.groups.map(g=>({group_id:g.group_id,app_ids:g.app_ids,score:g.score,score_status:g.score_status,coverage:g.coverage,structural_counts:g.structural_counts}))});
}
process.stdout.write(JSON.stringify({trials:outputs,positive_examples:examples}));
'''


def run(artifact,analysis,output,node='node',contexts=None):
    artifact=Path(artifact);analysis=Path(analysis);output=Path(output);output.mkdir(parents=True,exist_ok=True);summary=[]
    root=Path(__file__).resolve().parents[1]
    for context,(worker,name) in RUNS.items():
        if contexts and context not in contexts:continue
        study=artifact/'workers'/worker;folder=study/'cases'/name;matrix=analysis/context/'influences.json'
        service=GraphService(run=folder,influences=matrix,max_frames=160)
        previews=list(records(folder,'xapp_decision_previews'));slots={p['slot']for p in previews if p['slot']>=800}
        frames=[f for f in service.source['frames']if f['slot']in slots];anchor=next(f for f in frames if f['slot']==800)
        frozen={key:resolve_ranges(dict(ranges=rows,questions=[]),service.graph,anchor)['ranges']for key,rows in PROBES.items()}
        trials=[];probe_status={}
        for probe,ranges in frozen.items():
            for frame in frames:
                current=resolve_ranges(dict(ranges=[{k:r[k]for k in('kpi','lower','upper','lower_inclusive','upper_inclusive')}for r in ranges],questions=[]),service.graph,frame)['ranges']
                for row,initial in zip(current,ranges):row['anchors']=[dict(intent_id='audit:'+probe,current=initial['current'],snapshot=dict(slot=800,time_ms=400))]
                review=graph_intent_review(service.graph,frame,dict(ranges=current,questions=[]))
                probe_status.setdefault(probe,dict(planning_blocked=review['planning_blocked'],constraint_issues=review['constraint_issues']))
                for dashed in [False,True]:
                    catalog=review['combination_catalog_with_uncertain'if dashed else'combination_catalog']
                    groups=[dict(id=g['id'],app_ids=g['app_ids'],conflict_summaries=[dict(type=c['type'],apps=c['apps'])for c in g.get('conflict_summaries',[])])for g in catalog['groups']]
                    trials.append(dict(probe=probe,frame=frame,preview=service.preview_xapps(frame['slot']),ranges=current,groups=groups,show_dashed=dashed))
        raw=subprocess.check_output([node,'-e',NODE],input=json.dumps(dict(graph=service.graph,trials=trials)).encode(),cwd=root)
        results=json.loads(raw)
        with gzip.open(output/(context+'-recorded-action-risk.json.gz'),'wt')as stream:json.dump(results,stream,separators=(',',':'),allow_nan=False)
        scopes=[]
        for dashed in [False,True]:
            groups=[g for t in results['trials']if t['show_dashed']==dashed for g in t['groups']]
            scopes.append(dict(show_dashed=dashed,group_snapshot_reviews=len(groups),numeric_scores=sum(g['score']is not None for g in groups),
                zero_scores=sum(g['score']==0 for g in groups),positive_scores=sum(g['score']is not None and g['score']>0 for g in groups),
                unsupported_scores=sum(g['score']is None for g in groups),
                mean_known_chain_fraction=sum(g['coverage']['known_chains']/g['coverage']['total_chains']if g['coverage']['total_chains']else 0 for g in groups)/len(groups)if groups else None))
        summary.append(dict(context=context,decision_snapshots=len(frames),reviews=scopes,probe_status=probe_status,
            actual_recorded_shadow_run=str(folder),matrix_sha256=sha(matrix),checkpoint_sha256=sha(study/'checkpoints'/(name+'.json')),
            positive_examples=results['positive_examples']))
    report=dict(schema='oran.context-recorded-action-risk-audit.v1',probe_intents=PROBES,anchor_slot=800,contexts=summary,
        production_engine='intent_control_dynamics_v4',risk_engine_sha256=sha(root/'smo/web/app.js'),audit_code_sha256=sha(Path(__file__)),
        actual_proposals_recorded=True,new_model_inference=False,simulation_invoked=False,post_deployment_outcomes_used=False,
        user_intent_ledger_modified=False,selection_performed=False)
    write(output/'risk-coverage-summary.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--node',default='node');p.add_argument('--contexts',nargs='+',choices=list(RUNS));a=p.parse_args()
    r=run(a.artifact,a.analysis,a.output,a.node,a.contexts)
    print(json.dumps([dict(context=c['context'],reviews=c['reviews'])for c in r['contexts']],indent=2))


if __name__=='__main__':main()
