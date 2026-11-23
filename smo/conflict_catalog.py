"""Generate the retained legacy action-review catalog from canonical profiles.

python -m smo.conflict_catalog
This is an offline audit tool, not the active graph-only SMO browser path.
The catalog contains full writes, including reset/no-change writes that matter
for old action-space studies. It contains no models, simulator state or credentials.
"""
import json
from pathlib import Path
from smo.contracts import build_graph
from smo.intent_match import profiles

START = '/* ACTION_CATALOG_START */'
END = '/* ACTION_CATALOG_END */'


def action_catalog():
    graph = build_graph()
    return dict(schema='oran.action-catalog.v1', contracts=graph['contract_sha256'],
                profiles={app['id']: profiles(app) for app in graph['nodes'] if app['kind'] == 'xapp'})


def catalog_source():
    return START+'\n  const catalog = '+json.dumps(action_catalog(), separators=(',', ':'))+';\n  '+END


if __name__ == '__main__':
    path = Path(__file__).parent/'web/legacy_action_review.js'
    text = path.read_text()
    start, end = text.index(START), text.index(END)+len(END)
    path.write_text(text[:start]+catalog_source()+text[end:])
