"""Export traceable scene-separated identification figures and evidence checks."""
import argparse
from html import escape
import json
import math
from pathlib import Path
import textwrap

from smo.identify import audit_case, load, sha, write


TARGETS = ['ue_power_w', 'ap_power_w', 'network_goodput_mbps',
           'mec_offloaded_latency_ms', 'urllc_failure_events_per_s',
           'urllc_packet_latency_ms', 'mec_expected_gflops']
LABELS = ['UE power', 'BS power', 'Downlink goodput', 'Offloaded MEC delay',
          'URLLC failure events', 'URLLC packet delay', 'MEC compute demand']
CONTEXTS = ['native_1x', 'steady_2x', 'burst_2x']
NAMES = ['Native load', 'Double load / steady', 'Double load / burst']


def verify_shadow_bridges(artifact):
    """Repeated shadows are instrumentation checks, never independent seeds."""
    artifact = Path(artifact)
    pairs = [
        ('agx1', 'pilot_native_1x_s20261120_native', 'agx2', 'shadow_replay_native_1x_s20261120_native'),
        ('agx1', 'pilot_burst_2x_s20261120_native', 'agx1', 'shadow_replay_burst_2x_s20261120_native'),
        ('agx2', 'pilot_steady_2x_s20261121_native', 'agx2', 'shadow_replay_steady_2x_s20261121_native'),
    ]
    results = []
    for wa, ca, wb, cb in pairs:
        rows = []; seals = []
        for worker, name in [(wa, ca), (wb, cb)]:
            study = artifact/'workers'/worker; plan = load(study/'plan.json')
            seal = load(study/'checkpoints'/(name+'.json')); case = seal['case']
            protocol = {**plan, **case, 'allowed_refresh_rejection_reasons': ['ue_scheduler_weight_target_outside_cell_scope']}
            rows.append(audit_case(study, case, protocol, population=True))
            seals.append(sha(study/'checkpoints'/(name+'.json')))
        a, b = rows; differences = {}
        for key in a['metrics']:
            x, y = a['metrics'][key], b['metrics'][key]
            same = x is None and y is None or type(x) in (int, float) and type(y) in (int, float) and math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-9)
            if not same: differences[key] = dict(left=x, right=y)
        for key in ['source_sha256', 'exogenous_environment_sha256', 'cache_key',
                    'native_prefix_sha256', 'exogenous_arrivals_sha256', 'task_births_sha256']:
            if a['proof'][key] != b['proof'][key]: differences[key] = 'mismatch'
        if a['task_counts'] != b['task_counts']: differences['task_counts'] = 'mismatch'
        # Check individual packet counters as well as aggregate derived KPIs.
        packets = [load(artifact/'workers'/w/'cases'/c/'final_summary.json')['packet_metrics']
                   for w, c in [(wa, ca), (wb, cb)]]
        if packets[0] != packets[1]: differences['per_ue_packet_metrics'] = 'mismatch'
        metric_differences = {k:v for k,v in differences.items() if k in a['metrics'] or k in ('task_counts', 'per_ue_packet_metrics')}
        results.append(dict(left=dict(worker=wa, case=ca), right=dict(worker=wb, case=cb),
                            checkpoint_sha256s=seals, same_hardware=wa==wb, differences=differences,
                            scored_metrics_and_packet_counters_equal=not metric_differences,
                            strict_exogenous_bridge_passed=not differences))
    report = dict(schema='oran.shadow-hardware-bridge.v1',
                  same_hardware_instrumentation_passed=all(not r['differences'] for r in results if r['same_hardware']),
                  all_scored_metrics_and_packet_counters_equal=all(r['scored_metrics_and_packet_counters_equal'] for r in results),
                  relative_tolerance=1e-8, absolute_tolerance=1e-9, comparisons=results,
                  independent_observations_added=0,
                  interpretation='Same-hardware shadow instrumentation is checked strictly. The cross-hardware pair is an informational scored-metric comparison; different cache/environment/prefix hashes prohibit using it as a strict exogenous bridge or pooling its repeated run as a new replication.')
    write(artifact/'shadow-hardware-bridge.json', report)
    return report


def figures(analysis, output, contexts=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from smo.contracts import KPIS
    analysis = Path(analysis); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    contexts=contexts or CONTEXTS
    matrices = {c: load(analysis/c/'all-effects.json') for c in contexts}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none'})
    contrast_ids = list(dict.fromkeys(r['contrast_id'] for c in contexts for r in matrices[c]['estimates']))
    aliases = dict(offload_population_range='MEC offloading probability', drx_population_range='DRX inactivity timer',
        drx_cycle_population_range='DRX cycle', weight_mec_population_range='MEC scheduling priority',
        weight_urllc_population_range='URLLC scheduling priority', throughput_priority_component_range='Throughput priority component',
        scheduler_latency='RR → latency-aware', scheduler_qos='RR → QoS deadline', ap_micro_sleep='Active → micro sleep',
        ap_light_sleep='Active → light sleep', ap_deep_sleep='Active → deep sleep', drx_profile_80ms_1ms='Joint DRX profile')
    values = []
    for matrix in matrices.values():
        values += [abs(r['importance']['signed_reference_importance']) for r in matrix['estimates'] if r['kpi'] in TARGETS]
    limit = max(1., float(np.percentile(values, 90)))
    fig, axes = plt.subplots(1, len(contexts), figsize=(7*len(contexts), max(5, len(contrast_ids)*.43)), sharey=True)
    axes=np.atleast_1d(axes)
    for ax, context in zip(axes,contexts):
        title=NAMES[CONTEXTS.index(context)]
        lookup = {(r['contrast_id'], r['kpi']):r for r in matrices[context]['estimates']}
        arr = np.full((len(contrast_ids), len(TARGETS)), np.nan)
        for i, contrast in enumerate(contrast_ids):
            for j, key in enumerate(TARGETS):
                row = lookup.get((contrast, key))
                if row: arr[i, j] = row['importance']['signed_reference_importance']
        ax.set_facecolor('#f0f1f0'); im = ax.imshow(arr, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        for i, contrast in enumerate(contrast_ids):
            for j, key in enumerate(TARGETS):
                row = lookup.get((contrast, key))
                if not row:
                    ax.text(j, i, 'n/a', ha='center', va='center', color='#5b625f', fontsize=7); continue
                ci = row.get('confidence_interval'); uncertain = not ci or ci[0] <= 0 <= ci[1]
                if uncertain: ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch='///', edgecolor='#656c6860', linewidth=0))
                fallback = row['importance']['normalization_basis'] != 'absolute_reference_kpi'
                v = arr[i, j]; label = f'{v:+.2g}' if v else '0'
                ax.text(j, i, label+('†' if fallback else ''), ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > .68*limit else '#172623')
        ax.set_title(title, pad=12); ax.set_xticks(range(len(TARGETS)), LABELS, rotation=55, ha='right')
        ax.set_yticks(range(len(contrast_ids)), [aliases.get(x, x) for x in contrast_ids]); ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle('Signed parameter–KPI influence, separated by traffic context', y=.99, fontsize=14)
    footnote='G × tested parameter span / |reference KPI|; positive means the KPI rises, not necessarily an improvement. Hatching: 95% seed interval includes zero. †: zero-reference magnitude fallback. n/a: not estimated.'
    fig.text(.02, .012,textwrap.fill(footnote,width=110*len(contexts)),fontsize=8,va='bottom')
    fig.tight_layout(rect=(0, .11, .98, .95))
    cbar = fig.colorbar(im, ax=axes, location='right', fraction=.015, pad=.02, shrink=.7); cbar.set_label('Signed reference importance (colour clipped)')
    for ext in ['png', 'svg', 'pdf']: fig.savefig(output/('signed-influence-matrix.'+ext), dpi=190, bbox_inches='tight')
    plt.close(fig)
    # Every raw seed point and its own affine fit, including weak/zero outcomes.
    for context in contexts:
        ranges = [r for r in matrices[context]['estimates'] if r['kind']=='range_linear_fit' and r['kpi'] in TARGETS]
        ids = list(dict.fromkeys(r['contrast_id'] for r in ranges))
        if not ids: continue
        fig, axes = plt.subplots(len(ids), len(TARGETS), figsize=(22, 2.3*len(ids)), squeeze=False)
        for i, contrast in enumerate(ids):
            for j, kpi in enumerate(TARGETS):
                ax = axes[i, j]; row = next((r for r in ranges if r['contrast_id']==contrast and r['kpi']==kpi), None)
                if not row: ax.text(.5, .5, 'Unavailable', ha='center', transform=ax.transAxes); continue
                for n, seed_fit in enumerate(row['seed_fits']):
                    seed = seed_fit['seed']; points = row['grid_points']; xs = []; ys = []
                    for p in points:
                        v = next(x['value'] for x in p['by_seed'] if x['seed']==seed)
                        if v is not None: xs.append(p['x']); ys.append(v)
                    colour = plt.get_cmap('tab10')(n)
                    ax.scatter(xs, ys, color=colour, s=12)
                    x = row['fit_domain']; ax.plot(x, [seed_fit['intercept']+seed_fit['beta']*v for v in x], color=colour, lw=.8, alpha=.7)
                ax.set_xlabel(row['x_unit']); ax.set_ylabel(KPIS[kpi][1]); ax.grid(alpha=.15)
                if i==0: ax.set_title(LABELS[j], fontsize=9)
                if j==0: ax.text(-.48, .5, aliases.get(contrast, contrast), rotation=90, va='center', ha='center', transform=ax.transAxes, fontsize=9)
                ax.tick_params(labelsize=7)
        fig.suptitle(NAMES[CONTEXTS.index(context)]+' · raw points and per-seed first-order fits', fontsize=14)
        seeds=sorted({f['seed']for r in ranges for f in r['seed_fits']})
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0],[0],color=plt.get_cmap('tab10')(n),marker='o',lw=.8,label='Seed '+str(seed))for n,seed in enumerate(seeds)],
            loc='upper center',bbox_to_anchor=(.5,.965),ncol=len(seeds),frameon=False)
        fig.text(.04,.008,'Dots: measured values. Lines: per-seed affine fits. Poor fits and crossings of physical bounds remain visible; these lines are descriptive, not KPI forecasts.',fontsize=8,va='bottom')
        fig.tight_layout(rect=(.03, .035, 1, .94))
        for ext in ['png', 'svg']: fig.savefig(output/(context+'-sampled-responses.'+ext), dpi=150, bbox_inches='tight')
        plt.close(fig)
    write(output/'figure-provenance.json', dict(input_sha256s={c:sha(analysis/c/'all-effects.json') for c in contexts},
        target_kpis=TARGETS, unmeasured_values_imputed=False, confidence_unit='independent_seed_block',
        colour_clip=limit, annotations_unclipped=True,renderer_sha256=sha(__file__),matplotlib_version=matplotlib.__version__,numpy_version=np.__version__))
    return str(output/'signed-influence-matrix.png')


def report(artifact,analysis,figures_dir,output):
    """One offline HTML with the actual figure and every fitted coefficient."""
    from smo.contracts import KPIS
    from smo.population import KPI_OVERRIDES,PARAMETER_LABELS
    artifact=Path(artifact);analysis=Path(analysis);output=Path(output)
    matrices={c:load(analysis/c/'all-effects.json')for c in CONTEXTS}
    reports={c:load(analysis/c/'identification-report.json')for c in CONTEXTS}
    def number(x):return 'Unavailable'if x is None else f'{x:.5g}'
    def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+escape(h)+'</th>'for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>'for v in row)+'</tr>'for row in rows)+'</tbody></table>'
    cards=[];summary=[];baseline=[]
    for context,name in zip(CONTEXTS,NAMES):
        matrix=matrices[context];r=reports[context];dataset=load(analysis/context/'measurement-dataset.json')
        summary.append([name,r['observations'],len(r['complete_seed_blocks']),r['estimated_responses'],r['parameter_kpi_links'],r['signed_resolved'],r['unavailable_responses']])
        for case in dataset['cases']:
            if case['arm']!='native':continue
            m=case['metrics'];baseline.append([name,case['seed'],number(m['network_goodput_mbps']),number(m['ue_power_w']),number(m['ap_power_w']),number(m['urllc_failure_events_per_s']),number(m['urllc_failure_fraction']),number(m['mec_offloaded_latency_ms'])])
        rows=[]
        for row in matrix['estimates']:
            ci=row.get('confidence_interval');k=row['kpi'];label=KPI_OVERRIDES.get(k,(KPIS[k][0],))[0]
            rows.append([escape(PARAMETER_LABELS[row['parameter']]),escape(row['contrast_id']),escape(label),number(row['estimate']),escape(row['unit']),
                escape(' … '.join(number(v)for v in ci))if ci else'Unavailable',row['sample_count'],escape(row.get('fit_diagnostics',{}).get('assessment','paired categorical contrast'))])
        missing={}
        for row in matrix.get('unavailable',[]):missing[row['reason']]=missing.get(row['reason'],0)+1
        cards.append('<details><summary>'+escape(name)+' · all fitted responses</summary>'+table(['Parameter','Tested component / transition','KPI','Signed gain','Unit','95% seed interval','Seeds','Fit assessment'],rows)+'<p>Unavailable responses: '+escape(json.dumps(missing))+'</p></details>')
    svg=(Path(figures_dir)/'signed-influence-matrix.svg').read_text();svg=svg[svg.index('<svg'):]
    total=sum(r['observations']for r in reports.values())
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>O-RAN parameter–KPI identification</title><style>
body{font:14px/1.6 system-ui,sans-serif;color:#19332d;background:#f3f6f4;margin:0}main{max-width:1500px;margin:auto;padding:36px 24px}h1{font-size:28px}h2{font-size:20px;margin-top:32px}p{max-width:1100px}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0;background:#fcfdfc}td,th{text-align:left;border-bottom:1px solid #d4ded8;padding:9px;vertical-align:top}th{background:#e4f0eb}svg{width:100%;height:auto}details{margin:16px 0}summary{cursor:pointer;font-weight:600;padding:12px 0}code{overflow-wrap:anywhere}a{color:#17665d}figure{margin:24px 0}small{color:#5a6d65}@media(max-width:700px){table{font-size:10px}td,th{padding:5px}main{padding:20px 10px}}
</style><main><h1>O-RAN parameter–KPI identification</h1>'''
    content+=f'<p>{total} audited primary/supplemental fixed-control observations. Three independent primary seed blocks per traffic context; 6 moving UEs, 3 cells, unchanged accepted fast/system PHY. Each primary run lasts 1.8 s, controls change at 100 ms, counter metrics score 400–1600 ms, and mature task birth cohorts are followed until 1800 ms.</p>'
    if any(len(r['seeds'])>len(r['complete_seed_blocks'])for r in reports.values()):
        content+='<p>The burst context also includes a fixed fourth-seed baseline/QoS pair, admitted only after the required grid and supplemental component runs completed. It extends that paired scheduler contrast; it is not a fourth complete parameter grid. Each coefficient reports its own independent seed count.</p>'
    content+='<p>All five control-parameter families and all 19 KPI definitions are tested. A signed gain is a rough, context-dependent influence weight for conflict analysis. It is not a forecast of an xApp combination’s actual KPI or proof of intent attainment. No xApp was retrained, selected or automatically deployed.</p>'
    content+='<p>Power refers to the existing simulated UE radio/CPU and BS electronics models, not Jetson wall-plug measurements or a calibrated commercial site. The fixed 30 W/cell active reference and configured sleep/transition assumptions are retained in each original run’s configuration snapshot. These finite windows characterize the tested regimes, not a steady-state or carrier-grade URLLC reliability guarantee.</p>'
    content+=table(['Context','Audited runs','Complete primary seeds','Fitted responses','Parameter–KPI links','Sign-resolved responses','Unavailable responses'],summary)
    content+='<figure>'+svg+'</figure><h2>Baseline network conditions</h2><p>Each row is an independent seed, not an averaged-away outlier. Failure events use the scored counter interval. The separate cumulative failure ratio includes the common warm-up and is not the post-control event rate. MEC delay is conditional on completed offloaded tasks from matched mature birth cohorts; completion/success and pending counts remain in the measurement dataset.</p>'
    content+=table(['Context','Seed','Downlink goodput / Mbit/s','UE power / W','BS power / W','URLLC failures / events/s','Cumulative failure ratio','Offloaded MEC delay / ms'],baseline)
    content+='<h2>Measured weights</h2><p>Ranges use at least four distinct numeric levels and an intercept per seed. Confidence intervals use independent seed slopes or paired seed contrasts, never slot-level pseudo-replication. Wide intervals, nonlinearity, observed zero responses and unavailable outcomes are retained. An observed zero in this finite window does not establish a globally absent relationship. Different traffic contexts and observation horizons are never pooled.</p>'+''.join(cards)
    for context,name in zip(CONTEXTS,NAMES):
        plot=Path(figures_dir)/(context+'-sampled-responses.svg')
        if plot.exists():
            graphic=plot.read_text();graphic=graphic[graphic.index('<svg'):]
            content+='<details><summary>'+escape(name)+' · measured points and first-order fits</summary>'+graphic+'</details>'
    risk_path=artifact/'risk-audit/final/risk-coverage-summary.json'
    if risk_path.exists():
        risk=load(risk_path);rows=[];blocked=[]
        for entry in risk['contexts']:
            for scope in entry['reviews']:
                rows.append([escape(entry['context']),'All measured point weights'if scope['show_dashed']else'Default confidence filter',scope['group_snapshot_reviews'],scope['numeric_scores'],scope['positive_scores'],scope['unsupported_scores']])
            blocked.extend(entry['context']+' / '+key for key,value in entry.get('probe_status',{}).items()if value['planning_blocked'])
        content+='<h2>Conflict-module integration check</h2><p>Four declared intent probes use actual recorded, frozen-model proposals at 15 decision snapshots per context. Counts below are combination/snapshot reviews, not independent simulations. The single risk score measures opposing first-order control drives; it does not predict KPI changes or intent attainment. Default filtering excludes uncertain graph evidence, while the expanded view explicitly uses uncertain point weights. No combination is selected or deployed.</p>'
        content+=table(['Context','Evidence view','Reviews','Numerical risk','Positive risk','Unsupported risk'],rows)
        if blocked:content+='<p>Empty target intervals are excluded from enumeration: '+escape('; '.join(blocked))+'. In particular, strictly reducing an already-zero observed failure rate is infeasible at that anchor.</p>'
    content+='<h2>Evidence and retention</h2><p>Original traces, task cohorts, control receipts, replay HTML files, source pins, trained-model checksums and sealed file checkpoints remain under <code>workers/</code>. Full measurements and coefficients are retained under <code>analysis/</code>. The same-hardware shadow bridge verifies that recording trained proposals does not change physical results; repeated shadows are not extra independent observations.</p>'
    for context in CONTEXTS:
        content+='<p>'+escape(context)+': dataset SHA-256 <code>'+reports[context]['measurement_dataset_sha256']+'</code></p>'
    content+='<p><a href="http://127.0.0.1:53505/">Open the local SMO graph</a></p></main></html>'
    output.write_text(content)
    write(output.with_suffix('.provenance.json'),dict(analysis=str(analysis),matrix_sha256s={c:sha(analysis/c/'all-effects.json')for c in CONTEXTS},report_sha256=sha(output),renderer_sha256=sha(__file__),risk_audit_sha256=sha(risk_path)if risk_path.exists()else None))
    return str(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact', type=Path);p.add_argument('--analysis', type=Path);p.add_argument('--output', type=Path);p.add_argument('--figures',type=Path);p.add_argument('--bridges', action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--contexts',nargs='+',choices=CONTEXTS);a=p.parse_args()
    if a.bridges: print(json.dumps(verify_shadow_bridges(a.artifact), indent=2))
    elif a.report:print(report(a.artifact,a.analysis,a.figures,a.output))
    else: print(figures(a.analysis, a.output,a.contexts))


if __name__=='__main__': main()
