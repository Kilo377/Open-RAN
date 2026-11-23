"""Count whole-intent outcomes, keeping missing evidence and execution separate."""
from pathlib import Path
import json
import shutil
import subprocess
import pytest


def test_batch_counts_require_every_target_and_preserve_missing_and_execution_states():
    node=shutil.which('node')
    if not node:pytest.skip('Node required for UI aggregation')
    app=(Path(__file__).resolve().parents[1]/'smo/web/app.js').read_text()
    fn=app[app.index('  function memberOutcome(member,required=[]){'):app.index('  const deploymentRunning=')]
    def member(*statuses,before='satisfied',phase='complete'):
        return dict(phase=phase,comparison=dict(after=dict(all_satisfied=True),rows=[
            dict(kpi=k,before=dict(status=before),after=dict(status=s))
            for k,s in zip(('power','latency'),statuses)]))
    batch=dict(ranges=[dict(kpi='power'),dict(kpi='latency')],members=[
        member('satisfied','satisfied'),
        member('satisfied','above_range'),
        member('satisfied','below_range'), # strict endpoint remains unmet
        member('satisfied','unavailable'),
        member('satisfied'), # a missing target cannot count as success
        member('satisfied','satisfied',before='unavailable'),
        member('satisfied','empty_range'),
        *[member(phase=p)for p in ('pending','running','failed','cancelled')]])
    script=fn+'\nconsole.log(JSON.stringify(batchOutcomeCounts('+json.dumps(batch)+')));'
    result=json.loads(subprocess.check_output([node,'-e',script],text=True))
    assert result==dict(satisfied=1,unmet=2,insufficient=3,pending=1,running=1,failed=1,cancelled=1,invalid=1)
    assert sum(result.values())==len(batch['members'])
    assert 'data-outcome="'in app and 'catalog-outcome-badge'in app
    assert 'id="deployment-outcomes"'in(Path(__file__).resolve().parents[1]/'smo/web/index.html').read_text()
