"""Audit control paths without asserting that a declared goal is achieved."""
from smo.graph_combinations import influence_evidence


def audit_graph(graph, *, all_kpis=False):
    nodes={n['id']:n for n in graph['nodes']if n['kind']!='kpi'or all_kpis or n.get('primary')}
    controls=[e for e in graph['edges']if e['kind']=='controls'and e['source']in nodes and e['target']in nodes]
    influences=[e for e in graph['edges']if e['kind']=='influences'and e['source']in nodes and e['target']in nodes]
    visible=[e for e in influences if not influence_evidence(e,graph)['dashed']]
    incoming={key:[]for key in nodes};outgoing={key:[]for key in nodes}
    for e in controls+visible:
        outgoing[e['source']].append(e['target']);incoming[e['target']].append(e['source'])
    missing=[];paths=[]
    for key,node in nodes.items():
        if node['kind']!='xapp':continue
        for kpi in node.get('target_kpis',{}):
            target='kpi:'+kpi
            witnesses=[dict(parameter=c['target'][4:],control_edge=c['id'],influence_edge=e['id'],
                **influence_evidence(e,graph))for c in controls if c['source']==key
                for e in visible if e['source']==c['target']and e['target']==target]
            paths.append(dict(app_id=key,kpi=kpi,paths=witnesses))
            if not witnesses:missing.append(dict(app_id=key,kpi=kpi))
    isolated=[key for key in nodes if not incoming[key]and not outgoing[key]]
    return dict(schema='oran.graph-path-completeness.v1',all_kpis=all_kpis,
        node_counts={kind:sum(n['kind']==kind for n in nodes.values())for kind in('xapp','rcp','kpi')},
        controls=len(controls),visible_influence_links=len(visible),
        measured_influence_links=sum(influence_evidence(e,graph)['measured_evidence']for e in visible),
        declared_only_links=sum(not influence_evidence(e,graph)['measured_evidence']for e in visible),
        observed_zero_links=sum(influence_evidence(e,graph)['observed_zero']for e in visible),
        missing_target_paths=missing,isolated_nodes=isolated,target_paths=paths,
        incoming=incoming,outgoing=outgoing,topology_complete=not missing and not isolated,
        attainment_verified=False,zero_response_is_not_nonzero_influence=True)
