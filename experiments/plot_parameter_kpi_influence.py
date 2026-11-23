"""Paper-style unsigned influence panels from audited, scene-specific gains."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from smo.context_report import CONTEXTS, NAMES


PARAMETERS = [
    ('ue_offload_fraction', 'MEC offloading', 'MEC\noffloading'),
    ('ue_drx_configs', 'UE DRX timers', 'UE DRX\ntimers'),
    ('scheduler_mode', 'Downlink scheduling policy', 'Downlink\npolicy'),
    ('ue_scheduler_weights', 'UE scheduling priority', 'UE scheduling\npriority'),
    ('cell_sleep_mode', 'BS sleep mode', 'BS sleep\nmode'),
]
KPI_LABELS = {
    'network_goodput_mbps':'Downlink throughput', 'ue_power_w':'Total UE power',
    'ap_power_w':'Total BS power', 'mec_offloaded_latency_ms':'MEC task delay',
    'urllc_failure_events_per_s':'URLLC failure rate', 'urllc_packet_latency_ms':'URLLC packet delay',
    'mec_expected_gflops':'Expected MEC compute demand', 'ap_available_fraction':'BS availability',
    'hol_delay_ms':'Oldest queued-packet delay', 'mec_realized_gflops':'Realized MEC compute rate',
    'packet_latency_ms':'Downlink packet delay', 'per_ue_goodput_mbps':'Mean UE throughput',
    'queue_bytes':'Queued downlink data', 'queue_overflow_bytes':'Queue-overflow data loss',
    'task_latency_ms':'Task end-to-end delay', 'task_success_ratio':'On-time task completion',
    'ue_active_fraction':'UE active-time fraction', 'ue_energy_j':'UE energy (window)',
    'urllc_failure_fraction':'URLLC cumulative failure ratio',
}
PRIMARY = ['network_goodput_mbps','mec_offloaded_latency_ms','ue_power_w','ap_power_w',
           'urllc_failure_events_per_s','urllc_packet_latency_ms','mec_expected_gflops']
BLUE = '#538cb5'
CAPTION = ('Bars show the largest absolute normalized influence among the tested changes of each parameter family; '
           'dots retain the individual changes. Normalized influence = |gain × tested parameter span / reference KPI|. '
           'Hatching means the selected coefficient’s 95% seed interval includes zero. '
           '† includes a zero-reference fallback; ‡ some tested outcomes are unavailable. '
           'This is a first-order sensitivity summary, not SHAP, an improvement percentage or a combined-control prediction.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(matrix, parameter, kpi):
    rows=[r for r in matrix['estimates']if r['parameter']==parameter and r['kpi']==kpi]
    missing=[r for r in matrix.get('unavailable',[])if r['parameter']==parameter and r['kpi']==kpi]
    if not rows:
        return dict(parameter=parameter,kpi=kpi,value=None,components=[],unavailable=missing)
    values=[abs(r['importance']['signed_reference_importance'])for r in rows]
    if not all(np.isfinite(values)):
        raise ValueError('Nonfinite influence in the audited matrix')
    selected=rows[int(np.argmax(values))]
    ci=selected.get('confidence_interval')
    return dict(parameter=parameter,kpi=kpi,value=max(values),minimum=min(values),
        selected_contrast=selected['contrast_id'],selected_seeds=selected['sample_count'],
        selected_gain=selected['estimate'],selected_gain_ci95=ci,
        uncertain=not ci or ci[0]<=0<=ci[1],
        zero_reference=any(r['importance']['normalization_basis']!='absolute_reference_kpi'for r in rows),
        components=[dict(contrast_id=r['contrast_id'],magnitude=v,
            signed_importance=r['importance']['signed_reference_importance'],gain=r['estimate'],
            unit=r['unit'],seeds=r['sample_count'],confidence_interval=r.get('confidence_interval'),
            normalization_basis=r['importance']['normalization_basis'],parameter_span=r['importance']['parameter_span'],
            kpi_scale=r['importance']['kpi_scale'],reference_kpi_mean=r.get('reference_kpi_mean'),
            reference_scheduler_mode=r.get('reference_scheduler_mode'),members=r.get('members'),
            fit_domain=r.get('fit_domain'),scoring_window_ms=r.get('scoring_window_ms'),
            fit=r.get('fit_diagnostics'))for r,v in zip(rows,values)],
        unavailable=missing)


def number(x):
    if x==0:return '0'
    return f'{x:.2g}'


def draw(ax, rows, labels, title, *, rotation=0, annotate=True):
    highest=max((r['value']or 0 for r in rows),default=0)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#c7c7c7',linewidth=.55,alpha=.7)
    for i,r in enumerate(rows):
        v=r['value']
        if v is None:
            ax.text(i,.03*(highest or 1),'n/a',ha='center',fontsize=8,color='#777777');continue
        ax.bar(i,v,width=.57,color=BLUE,edgecolor='#343e45',linewidth=.65,
               hatch='///'if r['uncertain']and v else None,zorder=3)
        dots=[c['magnitude']for c in r['components']]
        # Jitter separates repeated equal magnitudes without changing values.
        jitter=np.linspace(-.15,.15,len(dots))if len(dots)>1 else[0]
        ax.scatter(np.asarray(jitter)+i,dots,s=12,c='#273c4b',edgecolors='white',linewidths=.35,zorder=4)
        if annotate:
            suffix=('†'if r['zero_reference']else'')+('‡'if r['unavailable']else'')
            ax.annotate(number(v)+suffix,(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7.5)
    ax.set_title(title,fontsize=11,pad=9)
    ax.set_xticks(range(len(rows)),labels,rotation=rotation,ha='right'if rotation else'center',fontsize=8)
    ax.set_xlim(-.6,len(rows)-.4);ax.set_ylim(0,highest*1.28 if highest else 1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3));ax.tick_params(axis='both',length=2,width=.6)
    ax.tick_params(axis='y',labelsize=8)
    for spine in ax.spines.values():spine.set_linewidth(.65);spine.set_color('#4a4a4a')


def save(fig, stem):
    for ext in ['png','svg','pdf']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def panels(matrix, context, kpis, dest, full=False, stem=None):
    columns=4 if full else 2;count=len(kpis);nrows=(count+columns-1)//columns
    fig,axes=plt.subplots(nrows,columns,figsize=(16 if full else 11.2,2.45*nrows+1.2),squeeze=False)
    for ax,kpi in zip(axes.flat,kpis):
        rows=[summarize(matrix,p,kpi)for p,_,_ in PARAMETERS]
        draw(ax,rows,[short for _,_,short in PARAMETERS],KPI_LABELS[kpi])
    for ax in list(axes.flat)[count:]:ax.axis('off')
    fig.suptitle('Parameter–KPI influence · '+NAMES[CONTEXTS.index(context)],fontsize=15,y=.994)
    fig.supylabel('Unsigned normalized influence',x=.006,fontsize=10)
    fig.legend(handles=[Patch(facecolor=BLUE,edgecolor='#343e45',label='Largest tested influence'),
        Line2D([],[],marker='o',color='#273c4b',linestyle='None',markersize=4,label='Individual measured changes'),
        Patch(facecolor='white',edgecolor='#343e45',hatch='///',label='Coefficient interval includes zero')],
        loc='lower center',bbox_to_anchor=(.5,.78/fig.get_figheight()),ncol=3,frameon=False,fontsize=8)
    fig.text(.02,.005,textwrap.fill(CAPTION,width=185 if full else 138),fontsize=7.5,va='bottom')
    fig.tight_layout(rect=(.02,1.2/fig.get_figheight(),1,.97),h_pad=2.1,w_pad=2)
    save(fig,dest/(stem or('all-kpis'if full else'primary-kpis')))


def profile(matrix,context,parameter,label,kpis,dest):
    fig,ax=plt.subplots(figsize=(13.8,6.6))
    rows=[summarize(matrix,parameter,k)for k in kpis]
    draw(ax,rows,[KPI_LABELS[k]for k in kpis],label+' · '+NAMES[CONTEXTS.index(context)],rotation=52)
    ax.set_ylabel('Unsigned normalized influence',fontsize=10)
    fig.text(.015,.005,textwrap.fill(CAPTION,width=170),fontsize=7.5,va='bottom')
    fig.tight_layout(rect=(0,.1,1,1))
    save(fig,dest/('parameter-'+parameter))


def all_parameters(matrix, context, kpis, dest):
    """Keep every graph parameter and KPI visible in one exported figure."""
    fig,axes=plt.subplots(len(PARAMETERS),1,figsize=(16,19.4))
    for ax,(parameter,label,_) in zip(axes,PARAMETERS):
        rows=[summarize(matrix,parameter,k)for k in kpis]
        draw(ax,rows,[KPI_LABELS[k]for k in kpis],label,rotation=43)
    fig.suptitle('All parameter–KPI influences · '+NAMES[CONTEXTS.index(context)],fontsize=16,y=.995)
    fig.supylabel('Unsigned normalized influence',x=.003,fontsize=11)
    fig.legend(handles=[Patch(facecolor=BLUE,edgecolor='#343e45',label='Largest tested influence'),
        Line2D([],[],marker='o',color='#273c4b',linestyle='None',markersize=4,label='Individual measured changes'),
        Patch(facecolor='white',edgecolor='#343e45',hatch='///',label='Coefficient interval includes zero')],
        loc='lower center',bbox_to_anchor=(.5,.039),ncol=3,frameon=False,fontsize=9)
    fig.text(.025,.006,textwrap.fill(CAPTION,width=190),fontsize=8,va='bottom')
    fig.tight_layout(rect=(.025,.075,1,.98),h_pad=2.0)
    save(fig,dest/'all-parameters')


def export_html(output):
    charts={}
    for path in output.glob('*/*.svg'):
        content=path.read_text();charts[str(path.relative_to(output).with_suffix(''))]=content[content.index('<svg'):]
    options=''.join(f'<option value="{c}">{html.escape(n)}</option>'for c,n in zip(CONTEXTS,NAMES))
    params=''.join(f'<option value="parameter-{p}">{html.escape(label)}</option>'for p,label,_ in PARAMETERS)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Parameter–KPI influence</title><style>
body{font:14px/1.6 system-ui,sans-serif;margin:0;background:#f4f5f6;color:#24313a}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 16px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,a{font:inherit}select{padding:8px;border:1px solid #bbc4ca;border-radius:6px;background:white}#chart{background:white;margin-top:20px;padding:14px}svg{width:100%;height:auto}p{max-width:1100px}small{color:#596671}.downloads{display:flex;gap:16px;margin-top:12px}a{color:#32698e}</style><main><h1>Parameter–KPI influence</h1><div class="controls"><label>Scenario <select id="scene">'''+options+'''</select></label><label>View <select id="view"><option value="all-parameters">All 5 parameters × 19 KPIs</option><option value="core-kpis">Four KPI overview</option><option value="primary-kpis">All 7 primary KPI panels</option><option value="all-kpis">All 19 KPI panels</option>'''+params+'''</select></label></div><div id="chart"></div><div class="downloads"><a id="png">PNG</a><a id="pdf">PDF</a><a id="svg">SVG</a></div><p>'''+html.escape(CAPTION)+'''</p><small>Source: completed controlled-perturbation study, 6 moving UEs and 3 cells. Three independent primary seeds per scenario; the burst QoS contrast has four paired seeds. Scenarios remain separate. No new simulation, xApp training, selection or deployment.</small></main><script>const charts='''+json.dumps(charts)+''';function render(){const key=document.getElementById('scene').value+'/'+document.getElementById('view').value;document.getElementById('chart').innerHTML=charts[key];for(const ext of['png','pdf','svg'])document.getElementById(ext).href=key+'.'+ext;}document.querySelectorAll('select').forEach(e=>e.onchange=render);render();</script></html>'''
    (output/'report.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.titleweight':'normal'})
    all_rows=[];hashes={}
    for context in CONTEXTS:
        path=a.analysis/context/'all-effects.json';matrix=json.loads(path.read_text());assert matrix['scenario']['id']==context
        hashes[context]=sha(path);dest=a.output/context;dest.mkdir(exist_ok=True)
        keys=PRIMARY+[k for k in KPI_LABELS if k not in PRIMARY]
        for parameter,_,_ in PARAMETERS:
            for kpi in keys:all_rows.append(dict(context=context,**summarize(matrix,parameter,kpi)))
        all_parameters(matrix,context,keys,dest)
        panels(matrix,context,PRIMARY[:4],dest,stem='core-kpis')
        panels(matrix,context,PRIMARY,dest)
        panels(matrix,context,keys,dest,full=True)
        for parameter,label,_ in PARAMETERS:profile(matrix,context,parameter,label,keys,dest)
    (a.output/'plotted-data.json').write_text(json.dumps(all_rows,indent=2,allow_nan=False))
    with(a.output/'plotted-data.csv').open('w',newline='')as f:
        fields=['context','parameter','kpi','value','minimum','selected_contrast','selected_seeds','uncertain','zero_reference'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_rows)
    export_html(a.output)
    (a.output/'provenance.json').write_text(json.dumps(dict(source_analysis=str(a.analysis.resolve()),source_matrices_sha256=hashes,renderer_sha256=sha(__file__),
        aggregation='max(abs(signed_reference_importance)) across available measured contrasts within each parameter/KPI/context; all components retained',
        missing_values_imputed=False,new_simulations=0,matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
        outputs_sha256={str(f.relative_to(a.output)):sha(f)for f in a.output.rglob('*')if f.is_file() and f.name != 'provenance.json'}),indent=2))
    print(json.dumps(dict(report=str(a.output/'report.html'),figures=27,scenarios=CONTEXTS,parameter_kpi_summaries=len(all_rows))))


if __name__=='__main__':main()
