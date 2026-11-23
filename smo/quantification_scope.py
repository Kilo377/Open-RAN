"""Read-only quantification scope, independent of deployment admission.

The structural catalog is enumerated separately from initial selection.
Admission outcomes annotate rows without restricting the scoring input.
"""
from copy import deepcopy


SCOPES = ('deployment_candidates', 'all_covered')


def quantification_catalog(catalog, scope='deployment_candidates', *, structural_catalog=None):
    if scope not in SCOPES:
        raise ValueError('Unknown quantification group scope')
    if scope == 'all_covered' and structural_catalog is None:
        raise ValueError('Independent quantification requires a structural catalog')
    result = deepcopy(structural_catalog if scope == 'all_covered' else catalog)
    groups = []
    seen = set()
    dispositions = {}
    for key, disposition in [('groups', 'candidate'), ('reserve_groups', 'reserve'), ('excluded_groups', 'excluded')]:
        for group in catalog.get(key, []):
            dispositions[group.get('id', group.get('group_id'))] = (disposition, group)
    counts = {'candidate': 0, 'reserve': 0, 'excluded': 0, 'not_selected': 0}
    for group in result.get('groups', []):
        identity = group.get('id', group.get('group_id'))
        if identity in seen:
            continue
        seen.add(identity)
        disposition, admitted = dispositions.get(identity, ('not_selected', {}))
        row = deepcopy(group)
        row['deployment_disposition'] = disposition
        row['deployment_eligible'] = disposition == 'candidate'
        if admitted.get('action_exclusion'):
            row['action_exclusion'] = deepcopy(admitted['action_exclusion'])
        groups.append(row)
        counts[disposition] += 1
    result['groups'] = groups
    result['group_count'] = len(groups)
    result['quantification_scope'] = dict(
        name=scope, counts=counts, group_count=len(groups),
        deployment_admission_changed=False, observed_outcomes_used=False,
        action_filter_used_for_scoring=False, final_selection_performed=False)
    return result
