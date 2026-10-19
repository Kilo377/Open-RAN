"""Presentation state from admitted bindings and versioned execution receipts."""


def status_rows(data):
    experiment = data.get('xapp_test') or {}
    catalog = experiment.get('catalog') or ([experiment['app']] if experiment.get('app') else [])
    running = data.get('stage') == 'simulating'
    pending = experiment.get('pending_activation') or {}
    latest = {}
    for event in data.get('control', {}).get('history', []):
        for provenance in event.get('provenance', []):
            key = (provenance.get('instance_id'), provenance.get('cell_id'), provenance.get('selection_version'))
            latest[key] = event
    rows = []
    for app in catalog:
        ident = app['instance_id']
        cells = app.get('active_cell_ids', [])
        available = app.get('available', True)
        state = 'ON' if cells else 'OFF' if available else 'UNAVAILABLE'
        switching = pending.get('app') == ident
        if switching:state = 'SWITCHING'
        evidence = []
        for cell in cells:
            token = app.get('selection_versions', {}).get(cell)
            event = latest.get((ident, cell, token)) if token is not None else None
            if event:
                evidence.append(f'{cell}: {event.get("status", "unknown")} at slot {event.get("apply_slot", event.get("slot_id", "--"))}')
            else:
                reasons = [d.get('reason') for d in experiment.get('last_decisions', [])
                    if d.get('cell_id') == cell and d.get('instance_id') == ident
                    and d.get('selection_version') == token]
                evidence.append(f'{cell}: ' + (reasons[-1] if reasons else 'awaiting execution receipt'))
        if not running:
            state = 'HISTORY / ' + state
        rows.append(dict(instance_id=ident, state=state, enabled=bool(cells), running=running and bool(cells),
            switching=switching,
            cells=list(cells), available=available,
            execution='Restoring previous parameters before activation' if switching else
                '; '.join(evidence) if cells else app.get('unavailable_reason') or 'No active binding'))
    return rows


def banner(data):
    rows = status_rows(data)
    active = [r['instance_id'] for r in rows if r['enabled']]
    if data.get('stage') != 'simulating':
        return 'xApps: RUN ENDED / historical selection only' if data.get('stage') in ('completed', 'stopped', 'failed') else 'xApps: INITIALIZING'
    pending = data.get('xapp_test', {}).get('pending_activation')
    if pending:return 'xApps: SWITCHING to ' + pending['app']
    return 'xApps ON: ' + ', '.join(active) if active else 'xApps: ALL OFF'


def button_title(app):
    titles = dict(live_scheduler_test='Queue / HOL demo', throughput_power='Frequency power',
        rf_energy='RF energy', load_balancing='Load balancing', ue_fairness='UE fairness',
        slice_sla='Slice scheduling', latency_guard='Latency guard', interference_coordination='Interference',
        ret_coverage='Coverage / tilt')
    category = app.get('category_id', '')
    if category == 'live_scheduler_test':return titles[category]
    implementation = app.get('implementation_id', app['instance_id'])
    algorithm = implementation.rsplit(':', 1)[-1]
    name = 'Standard rule' if algorithm == 'legacy_v1' else algorithm.replace('_', ' ').capitalize()
    return titles.get(category, category.replace('_', ' ').capitalize()) + ' / ' + name
