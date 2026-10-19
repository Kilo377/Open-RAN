from scripts.orin.live_xapp_status import banner, status_rows


def snapshot(stage='simulating', cells=(), version=2, event_version=None, event_status='applied'):
    app = dict(instance_id='latency', active_cell_ids=list(cells), available=True,
        selection_versions={c: version for c in cells})
    history = [] if event_version is None else [dict(status=event_status, apply_slot=41,
        provenance=[dict(instance_id='latency', cell_id='c0', selection_version=event_version)])]
    return dict(stage=stage, control=dict(history=history), xapp_test=dict(catalog=[app,
        dict(instance_id='power', active_cell_ids=[], available=True),
        dict(instance_id='beam', active_cell_ids=[], available=False, unavailable_reason='No beam reports')]))


def test_off_available_and_unavailable_are_separate_from_selector_or_execution():
    data = snapshot()
    assert [r['state'] for r in status_rows(data)] == ['OFF', 'OFF', 'UNAVAILABLE']
    assert banner(data) == 'xApps: ALL OFF'
    assert status_rows(data)[2]['execution'] == 'No beam reports'


def test_enabled_waits_for_execution_and_old_generation_cannot_confirm_it():
    data = snapshot(cells=['c0'], event_version=1)
    row = status_rows(data)[0]
    assert row['state'] == 'ON' and row['running']
    assert 'awaiting execution' in row['execution']
    assert banner(data) == 'xApps ON: latency'


def test_versioned_receipt_identifies_applied_or_rejected_execution():
    data = snapshot(cells=['c0'], event_version=2)
    assert status_rows(data)[0]['execution'] == 'c0: applied at slot 41'
    data = snapshot(cells=['c0'], event_version=2, event_status='rejected')
    assert status_rows(data)[0]['state'] == 'ON'
    assert 'rejected' in status_rows(data)[0]['execution']


def test_enabled_app_can_hold_without_being_misreported_as_off():
    data = snapshot(cells=['c0'])
    data['xapp_test']['last_decisions'] = [dict(instance_id='latency', cell_id='c0', selection_version=2, reason='no_valid_ue_samples')]
    assert status_rows(data)[0]['state'] == 'ON'
    assert 'no_valid_ue_samples' in status_rows(data)[0]['execution']


def test_completed_run_never_reports_an_app_as_currently_running():
    data = snapshot(stage='completed', cells=['c0'], event_version=2)
    row = status_rows(data)[0]
    assert row['state'] == 'HISTORY / ON' and row['enabled'] and not row['running']
    assert 'RUN ENDED' in banner(data)
    assert 'applied' in row['execution']
