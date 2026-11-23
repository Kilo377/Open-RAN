"""Presentation-only reading guide for frozen O-RAN study reports."""
import html
from pathlib import Path


INTERPRETATION = {
    'risk': 'Risk R is a dimensionless 0–100 obstruction score; larger means stronger estimated obstruction in the frozen reference model. It is not a probability of failure or a forecast of the deployed KPI.',
    'zero_risk': 'Zero estimated risk does not guarantee target attainment, useful improvement, or absence of natural network drift. Unknown evidence remains unavailable, never zero.',
    'actual_metric': 'Actual Metric uses only KPIs named in the intent. No change is zero. If any requested KPI regresses, the worst signed normalized KPI change determines Metric; otherwise their equally weighted mean is used. Improvements cannot compensate for a regression.',
    'actual_deterioration': 'Observed deterioration is D_obs = max(0, −Metric / 100), a nonnegative dimensionless value. Figures show 100 D_obs in normalized points. Beneficial progress and unchanged outcomes both have D_obs = 0; hard interval attainment is reported separately.',
    'scale': 'Power, throughput, latency and compute changes use the magnitude of their common BEFORE value as scale. URLLC failure-fraction changes use percentage points, not a relative percentage divided by an observed zero. Missing valid scales are not invented.',
    'interval': 'A single active interval boundary scores signed movement in the requested direction. A band or exact point scores reduced distance to that set, without rewarding movement towards an unrequested centre. Strict endpoints remain separate attainment tests: a zero Metric cannot establish that a strict target is satisfied.',
    'window': 'All arms share the certified common origin and equal BEFORE/AFTER windows, (t0 − L, t0] and (t0, t0 + L]. Counter KPIs use interval changes; failure fraction uses interval event counts; MEC latency requires a resolved mature task cohort. Expected MEC demand uses the full observed minimum/maximum for interval constraints, not only its mean.',
    'reference_window': 'Reference scores use predeployment proposals and measured graph gains, not actual AFTER values. The fifth method aligns its reference functional with mean/extrema observation semantics; its action-support envelope is a conservative approximation, not a guarantee that each endpoint occurs in the finite deployment window.',
    'replication': 'There are only two independent random seeds and six scene × seed blocks. Multiple arms, 15 intent reassessments and methods on one physical run are not independent seed replications. Macro comparisons weight intent × scene × seed strata equally; correlations are descriptive, without confidence or generalisation claims.',
    'pools': 'Primary ranking uses the frozen eligible candidate pool and the identical numeric cases across compared methods. OFF, rejected/reserve arms and combined-matrix results are separate diagnostics. Constant scores, ties and unavailable strata stay visible.',
    'versions': 'The original three methods use their earlier archived projector; Net (profile G) and Aggregation-aware (profile G) use the later empty-cell bundle correction. The five-method comparison therefore compares complete frozen versions, not one isolated algorithm change. Only the fourth-versus-fifth pair holds both G and the projector fixed to isolate the aggregation functional.',
}

PARTITIONS = {
    'original_matrix': ('Main study', 'Original 20-arm matrix only, retaining its frozen eligible candidate pool.'),
    'prospective_supplement': ('Two-arm supplement', 'BCE and BCDE only. These arms reuse the same two seeds and six origins; they do not add independent seeds.'),
    'combined_diagnostic': ('Combined diagnostic', 'Original and supplemental arms together for inspection. This does not replace the separately retained main-study primary analysis.'),
}


def interpretation_html(*,expanded=True):
    esc=html.escape
    entries=[('Risk',INTERPRETATION['risk']+' '+INTERPRETATION['zero_risk']),
        ('Actual result',INTERPRETATION['actual_metric']+' '+INTERPRETATION['actual_deterioration']),
        ('Scale and interval',INTERPRETATION['scale']+' '+INTERPRETATION['interval']),
        ('Observation windows',INTERPRETATION['window']+' '+INTERPRETATION['reference_window']),
        ('Comparison limits',INTERPRETATION['replication']+' '+INTERPRETATION['pools']),
        ('Version boundary',INTERPRETATION['versions'])]
    return '<details'+(' open'if expanded else'')+'><summary>How to read the measurements</summary>'+''.join(
        '<p><strong>'+esc(title)+'.</strong> '+esc(text)+'</p>'for title,text in entries)+'</details>'


def write_landing_page(output,*,title,status,pages,forward=False):
    """One concise navigation page; never reads predictions or outcomes."""
    output=Path(output);esc=html.escape;groups=[]
    if forward:
        definitions=[('profileG_pairwise_ablation','Same-G ablation',
            'Compare Net (profile G) and Aggregation-aware (profile G) on their common future numeric pool. Gains, states, ranges and frozen candidate flags are held fixed; only the reference functional changes.'),
            ('joint_five_matched','Five-method comparison',
            'Compare complete frozen versions on the intersection where every method has a numeric score and actual label. The earlier three projectors differ from those of methods four/five; this is not a pure aggregation ablation. This pool can differ from the two-method ablation.')]
    else:definitions=[(None,'Original four methods',
        'Read the main study first. The supplement and combined diagnostic are retained separately, with their own frozen pools.')]
    for comparison,heading,note in definitions:
        cards=[]
        for page in pages:
            if page.get('comparison')!=comparison:continue
            name,description=PARTITIONS[page['partition']];name=page.get('label',name);description=page.get('description',description)
            path=Path(page['html']).relative_to(output)
            cards.append('<article><h3><a href="'+esc(str(path))+'">'+esc(name)+'</a></h3><p>'+esc(description)+'</p></article>')
        groups.append('<section><h2>'+esc(heading)+'</h2><p>'+esc(note)+'</p><div class="cards">'+''.join(cards)+'</div></section>')
    scope=('The fifth method was added after original outcome availability. Its independent validation is restricted to the fixed jobs proven never started in both remote metadata scans bracketing scoring; all previously/during-started jobs are excluded. Original four-method full-matrix reports remain separate.'
        if forward else 'The four-method predictions were frozen before nonempty prospective control arms were admitted. Main, supplement and combined reports must not be treated as three independent experiments.')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title>'
    page+='<style>body{font:15px/1.6 system-ui,sans-serif;color:#203338;background:#f3f6f5;margin:0}main{max-width:1100px;margin:42px auto;padding:0 24px}h1{font-size:30px;margin-bottom:8px}h2{font-size:21px;margin-top:30px}h3{font-size:17px;margin:0}p{color:#485d62}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}article,details{background:white;border:1px solid #dbe4e1;border-radius:8px;padding:18px}a{color:#176b75}summary{font-weight:600;cursor:pointer}</style><main><h1>'+esc(title)+'</h1><p><strong>'+esc(status)+'</strong></p><p>'+esc(scope)+'</p>'
    page+=''.join(groups)+interpretation_html(expanded=False)+'</main></html>'
    (output/'index.html').write_text(page)
