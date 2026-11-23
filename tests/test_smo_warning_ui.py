"""Existing requirement issues do not become errors on a new input."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def presentation(result, record_ids=(), submitted=True):
    code = """
const fs=require('fs'),vm=require('vm');
const scope={};
vm.runInNewContext(fs.readFileSync('smo/web/app.js','utf8'),scope);
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const before=JSON.stringify(input.result);
const feedback=scope.SmoGraph.intentFeedback(input.result,{submitted:input.submitted});
const constraintBlocked=scope.SmoGraph.hasBlockingConstraints(input.result);
const records=Object.fromEntries(input.record_ids.map(id=>[id,scope.SmoGraph.constraintIssuesForRecord(input.result.constraint_issues,id)]));
process.stdout.write(JSON.stringify({feedback,records,constraintBlocked,unchanged:before===JSON.stringify(input.result)}));
"""
    output = subprocess.run(
        ['node', '-e', code], cwd=ROOT,
        input=json.dumps(dict(result=result, record_ids=record_ids, submitted=submitted)),
        capture_output=True, text=True, check=True,
    )
    return json.loads(output.stdout)


def test_successful_new_intent_keeps_old_issues_on_their_records():
    legacy = dict(kind='legacy_energy_requires_reparse', kpi='ue_energy_j',
                  intent_ids=['old-energy'], blocking=True,
                  message='旧 UE 累计能量须重新解析，J 基准不变。')
    empty = dict(kind='empty_intersection', kpi='urllc_failure_events_per_s',
                 intent_ids=['old-urllc'], blocking=True,
                 message='URLLC 的共同区间为空。')
    result = dict(questions=[], planning_blocked=True,
                  constraint_issues=[legacy, empty], ranges=[], candidates=[])
    actual = presentation(result, ['old-energy', 'old-urllc', 'new-power'])
    assert actual['feedback']['problem'] is False
    assert actual['constraintBlocked'] is True
    assert '已解析并保存' in actual['feedback']['text']
    assert legacy['message'] not in actual['feedback']['text']
    assert actual['records'] == {
        'old-energy': [legacy], 'old-urllc': [empty], 'new-power': [],
    }
    assert actual['unchanged']


def test_new_input_clarification_remains_an_input_error():
    result = dict(questions=['请选择 KPI 范围。'], constraint_issues=[],
                  planning_blocked=True, ranges=[], candidates=[])
    actual = presentation(result)
    assert actual['feedback'] == dict(text='请选择 KPI 范围。', problem=True)
    assert actual['constraintBlocked'] is False


def test_nonblocking_record_note_does_not_pause_joint_constraints():
    result = dict(questions=[], planning_blocked=False, ranges=[], candidates=[],
                  constraint_issues=[dict(blocking=False, intent_ids=['saved'], message='历史备注')])
    actual = presentation(result, ['saved'])
    assert actual['constraintBlocked'] is False
    assert '共同约束需要处理' not in actual['feedback']['text']
    assert actual['records']['saved'][0]['message'] == '历史备注'


def test_saved_review_does_not_report_an_error_on_the_input_form():
    result = dict(questions=[], planning_blocked=True,
                  constraint_issues=[dict(blocking=True, intent_ids=['old'])])
    actual = presentation(result, submitted=False)['feedback']
    assert actual['problem'] is False
    assert '已更新观测' in actual['text']
    assert '已解析并保存' not in actual['text']
