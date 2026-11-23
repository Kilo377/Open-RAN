"""Browser preset edits remain drafts, separate from parsed intent ranges."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def probe(body, data=None):
    script = """
const fs=require('fs'),vm=require('vm'),scope={};
vm.runInNewContext(fs.readFileSync('smo/web/app.js','utf8'),scope);
const graph=scope.SmoGraph,input=JSON.parse(fs.readFileSync(0,'utf8'));
const defaultsBefore=JSON.stringify(graph.intentCases);
const result=(()=>{
""" + body + """
})();
process.stdout.write(JSON.stringify({result,defaultsUnchanged:defaultsBefore===JSON.stringify(graph.intentCases)}));
"""
    completed = subprocess.run(
        ['node', '-e', script], cwd=ROOT, input=json.dumps(data),
        text=True, capture_output=True, check=True,
    )
    output = json.loads(completed.stdout)
    assert output['defaultsUnchanged'], 'Editing a draft changed the built-in presets'
    return output['result']


def test_handwritten_annotation_remains_plain_intent_text():
    text='[直接，间接] 所有 UE 的总平均功率降低至少 2%，同时 MEC 总预期计算需求不超过 1.8 GFLOP/s。'
    result=probe('return graph.validatePresetEdit(input);',dict(text=text))
    assert result==dict(text=text, tag='')


def test_free_memo_tag_roundtrip_stays_separate_from_intent_text():
    result = probe("""
const edits={c04:{text:'  UE 总平均功率降低至少 3%。  ',tag:'  下周复测 / trial B  '}};
const first=graph.customIntentCases(edits);
const restored=graph.customIntentCases(JSON.parse(JSON.stringify(edits)));
return {first:first.find(p=>p.id==='c04'),restored:restored.find(p=>p.id==='c04'),
  original:graph.intentCases.find(p=>p.id==='c04'),
  normalized:graph.validatePresetEdit(edits.c04)};
""")
    assert result['first'] == result['restored']
    assert result['first']['text'] == 'UE 总平均功率降低至少 3%。'
    assert result['normalized'] == dict(text='UE 总平均功率降低至少 3%。', tag='下周复测 / trial B')
    assert result['first']['tag'] == '下周复测 / trial B'
    assert 'trial B' not in result['first']['text']
    assert result['first']['customized'] is True
    assert result['original']['text'] != result['first']['text']


def test_changed_text_clears_inapplicable_hints_and_never_creates_parsed_ranges():
    result = probe("""
const edit={text:'将所有 UE 的总平均功率限制在 6 W 以内。',tag:'[直接，间接]',
  kpis:['ue_power_w'],note:'旧场景提示',needsThreshold:true,
  ranges:[{kpi:'ue_power_w',upper:6,upper_inclusive:true}]};
return graph.customIntentCases({c03:edit,c06:edit}).filter(p=>['c03','c06'].includes(p.id));
""")
    assert {row['id'] for row in result} == {'c03', 'c06'}
    for row in result:
        assert row['kpis'] == []
        assert 'note' not in row and 'needsThreshold' not in row
        assert 'ranges' not in row
        assert row['tag'] == '[直接，间接]'
        assert '解析 KPI 与目标范围' in row['baselineNote']


def test_memo_tag_only_edit_preserves_original_hints_and_does_not_add_ranges():
    result = probe("""
const ids=['c03','c06'];
const edits=Object.fromEntries(ids.map(id=>[id,{text:graph.intentCases.find(p=>p.id===id).text,
  tag:'仍需验证 / arbitrary memo',ranges:[{kpi:'ue_power_w',upper:0}]}]));
return graph.customIntentCases(edits).filter(p=>ids.includes(p.id)).map(row=>({
  row,original:graph.intentCases.find(p=>p.id===row.id)}));
""")
    for item in result:
        row, original = item['row'], item['original']
        assert row['tag'] == '仍需验证 / arbitrary memo'
        assert row['customized'] is True
        assert 'ranges' not in row
        assert row['text'] == original['text']
        assert row['kpis'] == original['kpis']
        assert row['note'] == original['note']
        assert row.get('needsThreshold') == original.get('needsThreshold')
    assert next(item['row'] for item in result if item['row']['id'] == 'c03')['needsThreshold'] is True


def test_memo_tag_can_be_empty_or_omitted_and_cleared_without_changing_text():
    result = probe("""
const original=graph.intentCases.find(p=>p.id==='c03');
const edits={c03:{text:original.text,tag:'待验证'}};
const tagged=graph.customIntentCases(edits).find(p=>p.id==='c03');
edits.c03.tag='  ';
const cleared=graph.customIntentCases(JSON.parse(JSON.stringify(edits))).find(p=>p.id==='c03');
return {tagged,cleared,original,omitted:graph.validatePresetEdit({text:original.text})};
""")
    assert result['tagged']['tag'] == '待验证'
    assert result['cleared'] == result['original']
    assert result['omitted'] == dict(text=result['original']['text'], tag='')


def test_invalid_or_unknown_saved_edit_is_rejected_without_polluting_defaults():
    result = probe("""
const invalid=[{text:' \\n\\t '},{text:'x'.repeat(2001)},{text:123},
  {text:'降低 UE 总功率。',tag:12},{text:'降低 UE 总功率。',tag:{}},
  {text:'降低 UE 总功率。',tag:'备'.repeat(121)}];
return invalid.map(edit=>{
  let rejected=false;try{graph.validatePresetEdit(edit);}catch{rejected=true;}
  const rows=graph.customIntentCases({c04:edit,unknown:{text:'未知预设',tag:'任意备注'}});
  return {rejected,unchanged:JSON.stringify(rows)===JSON.stringify(graph.customIntentCases()),
    hasUnknown:rows.some(p=>p.id==='unknown')};
});
""")
    assert all(row == dict(rejected=True, unchanged=True, hasUnknown=False) for row in result)


def test_text_and_free_tag_limits_accept_the_boundary_after_trimming():
    result = probe("""
return graph.validatePresetEdit({text:'  '+'文'.repeat(2000)+'  ',tag:'  '+'备'.repeat(120)+'  '});
""")
    assert result == dict(text='文' * 2000, tag='备' * 120)


def test_removing_override_restores_the_original_preset():
    result = probe("""
const edits={c03:{text:'基站总平均功率不超过 90 W。',tag:'[直接]'}};
const changed=graph.customIntentCases(edits).find(p=>p.id==='c03');
delete edits.c03;
const restored=graph.customIntentCases(JSON.parse(JSON.stringify(edits))).find(p=>p.id==='c03');
return {changed,restored,original:graph.intentCases.find(p=>p.id==='c03')};
""")
    assert result['changed']['text'] != result['original']['text']
    assert result['restored'] == result['original']
    assert result['restored']['needsThreshold'] is True


def test_deleted_preset_stays_hidden_after_roundtrip_and_undo_keeps_its_edits():
    result=probe("""
const saved={edits:{c04:{text:'将下行吞吐量提高至少 3%。',tag:'我的备注'}},deleted_ids:['c04']};
const restored=JSON.parse(JSON.stringify(saved));
const hidden=graph.customIntentCases(restored.edits,restored.deleted_ids);
restored.deleted_ids.pop();
return {hidden:hidden.some(p=>p.id==='c04'),
  recovered:graph.customIntentCases(restored.edits,restored.deleted_ids).find(p=>p.id==='c04'),
  unrelatedUnchanged:JSON.stringify(hidden.find(p=>p.id==='c03'))===JSON.stringify(graph.intentCases.find(p=>p.id==='c03'))};
""")
    assert result['hidden'] is False
    assert result['recovered']['text']=='将下行吞吐量提高至少 3%。'
    assert result['recovered']['tag']=='我的备注'
    assert result['unrelatedUnchanged'] is True
