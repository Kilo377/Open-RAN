"""Ownership facade for the existing xApps; original imports remain compatible."""
from copy import deepcopy


def catalog():
    from ric.near_rt.xapp_library import IMPLEMENTATIONS
    result = {'live_scheduler_test': dict(
        pool='yisong', implementation_id='live_scheduler_test:queue_hol',
        category_id='live_scheduler_test', description='Queue/HOL scheduler test (local rule)',
        commands=[], config=dict(type='live_scheduler_test', name='live_scheduler_test'))}
    for ident, spec in sorted(IMPLEMENTATIONS.items()):
        result[ident] = dict(deepcopy(spec), pool='yisong',
            config=dict(type='xapp_instance', implementation_id=ident, instance_id=ident))
    return result
