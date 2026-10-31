"""Run-local xApp pools. Pool selection never mutates a global app registry."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

POOL_NAMES = ('yisong', 'chongyu')


def validate_pool(name):
    if name not in POOL_NAMES:
        raise ValueError('Unknown xApp pool: ' + str(name))
    return name


def pool_catalog(name):
    validate_pool(name)
    if name == 'yisong':
        from .yisong import catalog
        return catalog()
    path = Path(__file__).parent / name / 'catalog.json'
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:raise ValueError('Duplicate pool catalog key: ' + key)
            result[key] = value
        return result
    entries = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(entries, dict):raise ValueError('Pool catalog must be an object')
    result = {}
    for ident, spec in entries.items():
        if not isinstance(ident, str) or not ident.strip() or not isinstance(spec, dict):
            raise ValueError('Invalid pool catalog entry')
        factory = spec.get('factory', '')
        if not factory.startswith(f'ric.near_rt.pools.{name}.') or factory.count(':') != 1:
            raise ValueError('xApp factory must live inside its selected pool: ' + ident)
        if not isinstance(spec.get('category_id'), str) or not spec['category_id']:
            raise ValueError('Pool xApp requires category_id: ' + ident)
        result[ident] = dict(deepcopy(spec), implementation_id=ident, pool=name,
            description=spec.get('description', ident), commands=list(spec.get('commands', [])),
            config=dict(type='pool_xapp', pool=name, implementation_id=ident, instance_id=ident))
    return result


def registration_configs(pool, cell_ids, slot_ms, *, reversible_only=False):
    return [dict(deepcopy(spec['config']), cell_ids=list(cell_ids), slot_duration_ms=slot_ms,
                 initial_bindings={}) for spec in pool_catalog(pool).values()
            if not reversible_only or not spec['commands']]


def live_configs(pool, cell_ids, slot_ms):
    return registration_configs(pool, cell_ids, slot_ms, reversible_only=True)


def build_pool_xapp(item, pool):
    validate_pool(pool)
    if item.get('pool', pool) != pool:raise ValueError('xApp belongs to a different pool')
    ident = item['implementation_id']
    spec = pool_catalog(pool).get(ident)
    if spec is None or 'factory' not in spec:
        raise ValueError(f'Unregistered implementation in {pool} pool: {ident}')
    module, class_name = spec['factory'].split(':')
    cls = getattr(importlib.import_module(module), class_name)
    app = cls(cell_ids=item['cell_ids'], name=item.get('instance_id', ident),
              slot_duration_ms=item.get('slot_duration_ms', .5), parameters=item.get('parameters'))
    from .base import PoolXApp
    if not isinstance(app, PoolXApp):raise ValueError('Pool xApps must implement PoolXApp')
    if app.implementation_id != ident or app.category_id != spec['category_id']:
        raise ValueError('Pool catalog and xApp identity disagree: ' + ident)
    if set(app.contract.commands) != set(spec['commands']):
        raise ValueError('Pool catalog and command contract disagree: ' + ident)
    app.xapp_pool = pool
    app.initial_bindings = deepcopy(item.get('initial_bindings', {}))
    return app
