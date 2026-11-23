"""Categorical transition evidence must not disappear or leak across actions."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node') or '/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node'


def presentation(edge, **options):
    code = "const fs=require('fs'),vm=require('vm'),scope={structuredClone};vm.createContext(scope);vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),scope);const i=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(scope.SmoGraph.influencePresentation(i.edge,i.options)));"
    return json.loads(subprocess.run(
        [NODE, '-e', code], input=json.dumps(dict(edge=edge, options=options)),
        text=True, capture_output=True, cwd=ROOT, check=True,
    ).stdout)


@pytest.fixture
def scheduler_edge():
    return dict(kind='influences', source='rcp:scheduler_mode', sensitivities=[
        dict(contrast_id='scheduler_latency', estimate=-.29, confidence_interval=[-1.12, .55]),
        dict(contrast_id='scheduler_qos', estimate=-6.90, confidence_interval=[-10.4, -3.4]),
    ])


def test_overview_retains_resolved_second_transition_without_a_synthetic_average(scheduler_edge):
    before = deepcopy(scheduler_edge)
    result = presentation(scheduler_edge)
    assert result['dashed'] is False and result['measured'] is True
    assert result['summary'] is True and result['sensitivity'] is None
    assert result['estimates'] == before['sensitivities']
    assert scheduler_edge == before


def test_a_latency_action_cannot_borrow_the_qos_transition_confidence(scheduler_edge):
    result = presentation(scheduler_edge, contrastIds=['scheduler_latency'])
    assert result['dashed'] is False and result['summary'] is False
    assert result['confidenceResolved'] is False
    assert result['sensitivity']['contrast_id'] == 'scheduler_latency'
    qos = presentation(scheduler_edge, contrastIds=['scheduler_qos'])
    assert qos['dashed'] is False and qos['sensitivity']['estimate'] == -6.90


@pytest.mark.parametrize('options', [
    dict(contrastIds=['unmeasured']),
    dict(contrastIds=['scheduler_qos', 'unmeasured']),
    dict(contrastIds=[]),
    dict(contrastIds=['scheduler_qos'], uncalibrated=True),
])
def test_missing_or_incompatible_action_context_stays_uncalibrated(scheduler_edge, options):
    result = presentation(scheduler_edge, **options)
    assert result['dashed'] is True and result['measured'] is False
    assert result['uncalibrated'] is True and result['sensitivity'] is None


def test_conflicting_transition_signs_remain_separate(scheduler_edge):
    scheduler_edge['sensitivities'][0].update(estimate=2, confidence_interval=[1, 3])
    result = presentation(scheduler_edge)
    assert result['dashed'] is False and result['sensitivity'] is None
    assert [s['estimate'] for s in result['estimates']] == [2, -6.90]


def test_zero_only_response_and_control_contract_have_distinct_visibility():
    edge = dict(kind='influences', sensitivity=dict(estimate=0, confidence_interval=[0, 0]))
    assert presentation(edge)['dashed'] is False
    assert presentation(edge)['observedZero'] is True
    control = dict(kind='controls', deployment='native_locked')
    assert presentation(control, contrastIds=['missing'], uncalibrated=True)['dashed'] is False


def test_declared_objective_path_remains_visible_without_inventing_a_gain():
    result=presentation(dict(kind='influences',objectives={'mec_app':'minimize'}))
    assert not result['dashed'] and result['declared']
    assert not result['measured'] and result['sensitivity']is None
