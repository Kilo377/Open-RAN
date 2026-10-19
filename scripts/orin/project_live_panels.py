"""Operations panels and optional registered-xApp test controls."""
import json
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from scripts.orin.live_xapp_status import banner, status_rows

ACCENT = (0.25, 0.85, 0.75, 1.0)
MUTED = (0.65, 0.73, 0.82, 1.0)
WARN = (1.0, 0.72, 0.25, 1.0)
FLAGS = ui.ImGuiWindowFlags_NoMove | ui.ImGuiWindowFlags_NoResize | ui.ImGuiWindowFlags_NoCollapse


def fmt(value, digits=2, suffix=''):
    if value is None:
        return '--'
    if isinstance(value, float):
        return f'{value:,.{digits}f}{suffix}'
    return f'{value}{suffix}'


def table(name, headers, rows):
    if ui.BeginTable(name, len(headers), ui.ImGuiTableFlags_Borders | ui.ImGuiTableFlags_RowBg | ui.ImGuiTableFlags_Resizable):
        for header in headers:
            ui.TableSetupColumn(header)
        ui.TableHeadersRow()
        for row in rows:
            ui.TableNextRow()
            for value in row:
                ui.TableNextColumn()
                ui.TextUnformatted(str(value))
        ui.EndTable()


def tree(label, value):
    if isinstance(value, dict):
        if ui.TreeNode(label):
            for key, child in value.items():
                tree(str(key), child)
            if not value:
                ui.TextColored(MUTED, '(none)')
            ui.TreePop()
    elif isinstance(value, list):
        if ui.TreeNode(f'{label} [{len(value)}]'):
            for i, child in enumerate(value):
                tree(str(i), child)
            ui.TreePop()
    else:
        ui.TextWrapped(f'{label}: {fmt(value)}')


def action_text(action):
    # Describe the actual request, retaining target/validity/source separately.
    omit = {'metadata', 'source_name', 'control_request_id', 'apply_scope', 'target_cell_id',
        'target_ue_id', 'effective_from_slot', 'effective_until_slot', 'valid_from_slot', 'valid_until_slot'}
    return ', '.join(f'{k}={v}' for k, v in action.items() if k not in omit and v is not None and v != {}) or 'Empty control envelope: no parameter change requested.'


class Dashboard:
    def __init__(self, view='integrated', state_path=None):
        self.view = view
        self.ue_index = 0
        self.cell_index = 0
        self.state_path = state_path
        self.test_cell = 0
        self.test_app = None
        self.test_message = ''

    def window(self, name, xy, size):
        ui.SetNextWindowPos(xy, ui.ImGuiCond_Always)
        ui.SetNextWindowSize(size, ui.ImGuiCond_Always)
        ui.SetNextWindowBgAlpha(.97)
        ui.Begin(name, FLAGS)

    def draw(self, data, status, error, focus):
        width, height = ps.get_window_size()
        config = data.get('configuration', {})
        stage = status.get('stage') if status.get('stage') in ('failed', 'stopped') else data.get('stage', status.get('stage', 'starting'))
        age = time.time() - data['updated_unix_s'] if data.get('updated_unix_s') else None
        stale = stage == 'simulating' and age is not None and age > 10
        self.window('OPEN-RAN  /  Live Network Workspace', (10, 10), (width - 20, 112))
        access = banner({**data, 'stage': stage}) if data.get('xapp_test') else 'READ ONLY'
        ui.TextColored(ACCENT, f'{stage.upper()}   |   {data.get("scene", "loading scene")}   |   {config.get("control_mode", "loading mode")} control   |   {access}')
        slot, sim_ms, wall = data.get('slot', 0), data.get('sim_time_ms', 0), data.get('wall_s', 0)
        ui.TextUnformatted(f'Slot {slot}/{data.get("total_slots", "...")}   Simulation {sim_ms:.3f} ms   Wall {wall:.1f} s   State age {fmt(age, 1, " s")}')
        if stale:
            ui.TextColored(WARN, 'Waiting for next actual slot. Display holds the last completed state.')
        else:
            ui.TextColored(MUTED, 'One actual simulation, synchronized views. Charts use simulation time; physics can run slower than wall time.')
        ui.SameLine()
        views = ['Integrated', 'Network only', 'Scene only']
        current = ['integrated', 'telemetry', 'scene'].index(self.view)
        ui.SetNextItemWidth(240.)
        changed, selected = ui.Combo('View', current, views)
        if changed:
            self.view = ['integrated', 'telemetry', 'scene'][selected]
        if error or data.get('error') or status.get('error'):
            ui.TextColored(WARN, str(error or data.get('error') or status.get('error')))
        ui.End()
        if self.view == 'scene':
            return
        panel_width = min(720, max(620, width * .37)) if self.view == 'integrated' else width * .65 - 15
        self.window('Network operations', (10, 132), (panel_width, max(250, height - 142)))
        if ui.BeginTabBar('operations-tabs'):
            tabs = [('Overview', self.overview), ('KPIs', self.kpis), ('UEs', self.ues), ('Cells', self.cells),
                    ('Config', self.config), ('Control', self.control)]
            if data.get('xapp_test'):
                tabs.insert(0, ('xApp test', self.xapp_test))
            for label, draw in tabs:
                active, _ = ui.BeginTabItem(label, True)
                if active:
                    draw({**data, 'stage': stage}, focus)
                    ui.EndTabItem()
            ui.EndTabBar()
        ui.End()
        graph_x = panel_width + 22
        graph_h = min(345, max(265, height * .34))
        graph_y = height - graph_h - 10 if self.view == 'integrated' else 132
        graph_height = graph_h if self.view == 'integrated' else height - 142
        self.window('KPI trends / simulation clock', (graph_x, graph_y), (width - graph_x - 10, graph_height))
        self.trends(data)
        ui.End()

    def overview(self, d, focus):
        ui.TextColored(ACCENT, 'NETWORK HEALTH')
        packet = d.get('packet_totals', {})
        count = len(d.get('ues', []))
        table('overview-metrics', ['Metric', 'Actual value'], [
            ('Connected UEs', f'{d.get("connected_ues", 0)} / {count}'),
            ('Throughput / run average', fmt(d.get('throughput_mbps'), 3, ' Mbps')),
            ('Throughput / latest slot', fmt(d.get('slot_throughput_mbps'), 3, ' Mbps')),
            ('Delivered bytes', f'{d.get("delivered_bytes", 0):,} B'),
            ('MAC queue', f'{d.get("queue_bytes", 0):,} B'),
            ('Packet ledger pending', fmt(packet.get('pending_bytes'), suffix=' B')),
            ('Overflow bytes', fmt(packet.get('overflow_bytes'), suffix=' B')),
            ('Mean completed-packet delay', fmt(packet.get('mean_completion_latency_ms'), 3, ' ms')),
            ('Completed packets', fmt(packet.get('completed_packets'))),
            ('Actual NR decode attempts', fmt(d.get('decode_attempts'))),
            ('NACK / actual decode attempts', f'{d.get("nacks", 0)} / {d.get("decode_attempts", 0)}'),
        ])
        ui.TextWrapped('NACK fraction is an observed cumulative sample ratio, not a converged BLER estimate. Delay excludes unfinished packets.')
        ui.Separator()
        ui.TextColored(ACCENT, 'COMPUTE BACKENDS')
        for key, value in d.get('execution_devices', {}).items():
            ui.TextUnformatted(f'{key}: {value}')
        profile = d.get('configuration', {}).get('platform', {})
        ui.TextUnformatted(f'Platform: {profile.get("requested", "unknown")} / preflight: {profile.get("preflight", "unknown")}')
        if profile.get('fallback_reason'):
            ui.TextWrapped(profile['fallback_reason'])
        ui.Separator()
        ui.TextWrapped('3D uses the current Sionna 2.2 scene. Large points = BS, small points = UE. Lines show serving associations, not multipath rays.')
        if ui.Button('Fit network scene'):
            focus(None)

    def ues(self, d, focus):
        users = d.get('ues', [])
        if not users:
            ui.TextUnformatted('Waiting for the first actual slot...')
            return
        self.ue_index = min(self.ue_index, len(users) - 1)
        changed, self.ue_index = ui.Combo('Selected UE', self.ue_index, [u['ue_id'] for u in users])
        if changed:
            focus(users[self.ue_index]['position_m'])
        ui.SameLine()
        if ui.Button('Focus in 3D'):
            focus(users[self.ue_index]['position_m'])
        table('ue-overview', ['UE', 'Cell', 'SINR dB', 'MCS', 'Queue B', 'ACK'], [
            [u['ue_id'], u['cell_id'], fmt(u.get('sinr_db')), u.get('mcs', '--'), u['queue_bytes'],
             ('ACK' if u['ack'] else 'NACK') if u['scheduled'] else 'idle'] for u in users])
        user = users[self.ue_index]
        ui.Separator()
        ui.TextColored(ACCENT, f'DETAIL / {user["ue_id"]}')
        table('ue-detail', ['Field', 'Actual state'], [
            ('Connection', user.get('connection_state')), ('Slice / traffic', f'{user.get("slice_id")} / {user.get("traffic_type")}'),
            ('PRBs / symbols', f'{user.get("allocated_prbs")} / {user.get("allocated_symbols")}'),
            ('Rank', user.get('rank')), ('Handover count', user.get('handover_count')),
            ('Handover target', user.get('pending_target_cell') or '--'),
            ('Physical position / m', ', '.join(f'{v:.3f}' for v in user['position_m'])),
            ('HOL delay / ms', fmt(user.get('packet', {}).get('hol_delay_ms'), 3)),
            ('Mean completion delay / ms', fmt(user.get('packet', {}).get('mean_completion_latency_ms'), 3)),
            ('Pending HARQ bytes', fmt(user.get('harq', {}).get('pending_retx_bytes'))),
        ])

    def kpis(self, d, focus):
        ui.TextColored(ACCENT, 'LIVE KPI DETAILS / EXPORTED MEASUREMENTS')
        ui.TextWrapped('The trend charts show throughput and queue only. More live KPIs are below. This is the live export, not every internal model metric.')
        packet = d.get('packet_totals', {})
        users = d.get('ues', [])
        attempts = d.get('decode_attempts', 0)
        hol = [u.get('packet', {}).get('hol_delay_ms') for u in users]
        maximum = [u.get('packet', {}).get('max_completion_latency_ms') for u in users]
        table('kpi-network', ['Network metric', 'Current value'], [
            ['Goodput / latest slot Mbps', fmt(d.get('slot_throughput_mbps'), 3)],
            ['Goodput / run average Mbps', fmt(d.get('throughput_mbps'), 3)],
            ['MAC queue bytes', fmt(d.get('queue_bytes'))],
            ['Pending packet bytes', fmt(packet.get('pending_bytes'))],
            ['Overflow bytes', fmt(packet.get('overflow_bytes'))],
            ['Completed packets', fmt(packet.get('completed_packets'))],
            ['Mean completed-packet delay ms', fmt(packet.get('mean_completion_latency_ms'), 3)],
            ['Max observed completed-packet delay ms', fmt(max((v for v in maximum if v is not None), default=None), 3)],
            ['Current max UE HOL delay ms', fmt(max((v for v in hol if v is not None), default=None), 3)],
            ['Late completed fraction', fmt(packet.get('late_completed_fraction'), 4)],
            ['Packets that missed their deadline', fmt(packet.get('deadline_missed_packets'))],
            ['NACK / actual NR decode attempts', f'{d.get("nacks", 0)} / {attempts}'],
            ['Observed cumulative NACK fraction', fmt(d.get('nacks', 0) / attempts if attempts else None, 4)],
        ])
        ui.TextWrapped('Delay covers completed packets; pending work is shown separately. NACK fraction includes the observed decode attempts and is not a converged BLER estimate.')
        table('kpi-cell', ['Cell', 'Mbps', 'Grid %', 'RF W', 'Queue B'], [
            [c['cell_id'], fmt(c.get('slot_throughput_mbps')), fmt(c.get('grid', {}).get('utilization', 0) * 100),
             fmt(c.get('phy_power', {}).get('average_used_w'), 4), fmt(c.get('queue_bytes'))] for c in d.get('cell_metrics', [])])
        tree('All exported network packet counters', packet)
        tree('Per-UE radio, queue, HOL, HARQ and packet counters', users)
        tree('Per-cell resources, MU groups and RF power counters', d.get('cell_metrics', []))
        ui.TextWrapped('Not aggregated in this live export: P95/P99 packet delay, separate first-transmission BLER and UE goodput fairness over a declared window. Extended controller KPI reports remain in experiment artifacts. Whole-device energy needs a device power model or measurement.')

    def cells(self, d, focus):
        cells = d.get('cell_metrics', [])
        table('cell-overview', ['Cell', 'UEs', 'Sched', 'Mbps', 'Queue B'], [
            [c['cell_id'], c['associated_ues'], c['scheduled_ues'], fmt(c['slot_throughput_mbps']), c['queue_bytes']] for c in cells])
        if not cells:
            return
        self.cell_index = min(self.cell_index, len(cells) - 1)
        _, self.cell_index = ui.Combo('Cell details', self.cell_index, [c['cell_id'] for c in cells])
        c = cells[self.cell_index]
        p, g, effective = c.get('phy_power', {}), c.get('grid', {}), c.get('effective', {})
        ui.TextColored(ACCENT, 'ACTUAL POWER & RESOURCES')
        table('cell-resources', ['Field', 'Value'], [
            ('Configured TX / dBm', fmt(effective.get('tx_power_dbm'))),
            ('Power cap / dBm', fmt(effective.get('power_cap_dbm'))),
            ('Emitted average / W', fmt(p.get('average_used_w'), 5)),
            ('PHY power budget / W', fmt(p.get('budget_w'), 5)),
            ('Sleep mode', effective.get('cell_sleep_mode', '--')),
            ('Grid occupancy', fmt(g.get('utilization', 0) * 100, 2, '%')),
            ('PRB budget', fmt(g.get('total_prbs'))),
            ('Scheduled spatial layers', fmt(g.get('scheduled_layer_count_total'))),
            ('MU groups / UEs', f'{g.get("mu_mimo_group_count", "--")} / {g.get("mu_mimo_ue_count", "--")}'),
        ])
        tree('Runtime scheduler parameters', c.get('scheduler', {}))
        tree('Effective cell parameters', effective)

    def config(self, d, focus):
        c = d.get('configuration', {})
        radio = c.get('radio', {})
        ui.TextColored(ACCENT, 'ACTIVE RUN CONFIGURATION')
        table('radio-config', ['Radio parameter', 'Value'], [
            ('Carrier frequency', fmt(radio.get('carrier_frequency_hz', 0) / 1e9, 3, ' GHz')),
            ('Bandwidth', fmt(radio.get('bandwidth_hz', 0) / 1e6, 2, ' MHz')),
            ('Subcarrier spacing', fmt(radio.get('subcarrier_spacing_hz', 0) / 1e3, 1, ' kHz')),
            ('Slot duration', fmt(radio.get('slot_duration_ms'), 3, ' ms')),
            ('FFT size / symbols', f'{radio.get("fft_size", "--")} / {radio.get("num_symbols", "--")}'),
            ('TX array rows x cols', f'{radio.get("tx_rows", "--")} x {radio.get("tx_cols", "--")}'),
            ('UE RX antennas', fmt(radio.get('num_rx_ant'))),
            ('Feedback model', d.get('feedback_mode', '--')),
        ])
        ui.TextWrapped('These values are captured from this run after platform/mode overrides. Current effective per-cell values are in Cells / Control.')
        tree('Propagation / ray solver', c.get('propagation', {}))
        tree('PHY and mobility configuration', c.get('phy', {}))
        tree('Observation / actuation delay settings', c.get('timing', {}))
        tree('Controller configuration', c.get('controllers', {}))
        tree('Platform and startup checks', c.get('platform', {}))

    def control(self, d, focus):
        config = d.get('configuration', {})
        controls = d.get('control', {})
        apps = config.get('controllers', {})
        mode = config.get('control_mode', 'unknown')
        ui.TextColored(ACCENT, f'CONTROL PLANE / {mode.upper()}')
        if mode == 'baseline':
            ui.TextWrapped('Fixed baseline: rApps/xApps are disabled for this run. Effective controls below still show the actual network configuration.')
        ui.TextUnformatted(f'Registered xApps: {len(apps.get("xapps", []))} / rApps: {len(apps.get("rapps", []))}')
        if d.get('xapp_test'):
            ui.TextColored(ACCENT, banner(d))
            ui.TextWrapped('Registration is not activation. Per-instance ON/OFF, cell scope and execution status are in xApp test.')
        ui.TextUnformatted(f'Configured periods: xApp {apps.get("xapp_interval_slots", "--")} slots / rApp {apps.get("rapp_interval_slots", "--")} slots')
        ui.Separator()
        ui.TextColored(ACCENT, 'EFFECTIVE NOW / FROM RUNTIME CELLS')
        table('effective-controls', ['Cell', 'TX dBm', 'Cap dBm', 'Sleep', 'Scheduler'], [
            [c['cell_id'], fmt(c['effective'].get('tx_power_dbm')), fmt(c['effective'].get('power_cap_dbm')),
             c['effective'].get('cell_sleep_mode', '--'), c['scheduler'].get('scheduler_mode', '--')] for c in d.get('cell_metrics', [])])
        for c in d.get('cell_metrics', []):
            tree(f'{c["cell_id"]}: all effective controls', dict(cell=c['effective'], scheduler=c['scheduler']))
        ui.Separator()
        pending = controls.get('pending', [])
        ui.TextColored(WARN if pending else MUTED, f'PENDING ACTUATION / {len(pending)}')
        for i, item in enumerate(pending[:20]):
            action = item.get('action') or {}
            source = action.get('source_name', 'unknown')
            ui.TextWrapped(f'Slot {item["due_slot"]} | {item["kind"]} | {source} | target {action.get("target_cell_id") or action.get("apply_scope", "--")}')
            ui.TextWrapped(action_text(action))
        if not pending:
            ui.TextUnformatted('No pending actuation in the simulator pipeline.')
        ui.Separator()
        totals = controls.get('totals', {})
        ui.TextUnformatted('RUN RECEIPTS: ' + (' | '.join(f'{k}: {v}' for k, v in totals.items()) or 'none yet'))
        ui.TextWrapped('Receipts can include empty baseline envelopes. Applied receipt counts do not mean a parameter changed; effective values are shown above.')
        if ui.CollapsingHeader('Recent applied / rejected / delayed receipts'):
            events = controls.get('history', [])
            if not events:
                ui.TextUnformatted('No execution receipts in this run yet.')
            for item in reversed(events[-30:]):
                ui.TextColored(WARN if item.get('status') == 'rejected' else ACCENT,
                    f'slot {item.get("slot_id")} / {item.get("status", "unknown")} / {item.get("source", item.get("kind", "--"))}')
                ui.TextWrapped(f'target={item.get("target_cell_ids", item.get("target_cell_id", "--"))} | reason={item.get("reason", "--")}')
        tree('Latest requested configuration (not proof of execution)', controls.get('requested_config', []))
        tree('Active policy constraints', controls.get('active_constraints', []))
        tree('Field arbitration / rejected proposals', controls.get('arbitration', {}))
        tree('Current xApp bindings', controls.get('xapp_bindings', {}))
        tree('Controller observation provenance / delay', controls.get('observation', {}))

    def xapp_test(self, d, focus):
        from experiments.live_xapp_test import submit, start_comparison
        x = d['xapp_test']
        catalog = x.get('catalog') or [{**x['app'], 'available': True, 'registered': True}]
        if self.test_app is None:
            self.test_app = next((i for i, a in enumerate(catalog) if a.get('active_cell_ids')), 0)
        self.test_app = min(self.test_app, len(catalog) - 1)
        available = sum(bool(a.get('available')) for a in catalog)
        ui.TextColored(ACCENT, f'xAPP DISCOVERY / {len(catalog)} found / {available} available')
        states = status_rows(d)
        active = [r for r in states if r['enabled']]
        ui.TextColored(ACCENT if d.get('stage') == 'simulating' and active else WARN, banner(d))
        if d.get('stage') != 'simulating':
            ui.TextWrapped('Simulation has ended. ON/OFF and execution values below are the final recorded state; no xApp is running now.')
        labels = [f'[{r["state"]}] {r["instance_id"]}' for r in states]
        _, self.test_app = ui.Combo('Select xApp', self.test_app, labels)
        selected = catalog[self.test_app]
        selected_status = states[self.test_app]
        app = selected['instance_id']
        ui.TextColored(ACCENT if selected_status['running'] else MUTED, f'SELECTED xAPP STATUS: {selected_status["state"]}')
        ui.TextWrapped('Enabled cells: ' + (', '.join(selected_status['cells']) or 'none'))
        ui.TextWrapped('Execution evidence: ' + selected_status['execution'])
        ui.TextWrapped(selected.get('description', app))
        ui.TextWrapped('Writes: ' + ', '.join(selected.get('writable_actions', [])))
        if selected.get('unavailable_reason'):
            ui.TextWrapped('Unavailable: ' + selected['unavailable_reason'])
        ui.TextWrapped('One xApp per test. Disable and wait for parameter restoration before selecting another. Missing or stale measured inputs make the algorithm hold.')
        cells = ['All cells'] + [c['cell_id'] for c in d.get('cells', [])]
        self.test_cell = min(self.test_cell, len(cells) - 1)
        _, self.test_cell = ui.Combo('Test scope', self.test_cell, cells)
        cell = '*' if self.test_cell == 0 else cells[self.test_cell]
        if active:
            ui.TextWrapped(('Running xApp: ' if d.get('stage') == 'simulating' else 'Final selected xApp: ') + ', '.join(r['instance_id'] for r in active))
        ui.TextUnformatted(f'Control period: {x["period_ms"]:g} ms | transition exclusion: {x["settle_ms"]:g} ms')
        if d.get('stage') == 'simulating' and self.state_path:
            for label, operation in [('Enable xApp', 'enable'), ('Disable & restore', 'disable')]:
                if operation == 'enable' and not selected.get('available'):
                    continue
                if ui.Button(label):
                    try:
                        ident = submit(self.state_path, d, operation, cell,
                            app if operation == 'enable' else x['app']['instance_id'])
                        self.test_message = f'Request {ident[:8]} submitted; processing receipt is shown below.'
                    except (OSError, ValueError) as error:
                        self.test_message = str(error)
                if operation == 'enable':ui.SameLine()
        else:
            ui.TextColored(MUTED, 'This run has finished. Hot toggles require a running simulation.')
        if self.test_message:ui.TextWrapped(self.test_message)
        if x.get('receipts'):
            r = x['receipts'][-1]
            ui.TextWrapped(f'Command receipt / slot {r["received_at_slot"]}: {r["operation"]} / {r["status"]}')
            if r['status'] != 'accepted':ui.TextWrapped(r['reason'])
        ui.TextWrapped('ON means admitted for the listed cells. Execution evidence and runtime values confirm execution. OFF withdraws the binding; parameter restoration still waits for the control pipeline.')
        table('test-effective', ['Cell', 'Binding ON/OFF', 'Effective scheduler'], [
            [c['cell_id'], 'ON' if c['cell_id'] in x['selected_cells'] else 'OFF', c['scheduler'].get('scheduler_mode', '--')]
            for c in d.get('cell_metrics', [])])
        tree('Latest measured inputs / xApp decisions', x.get('last_decisions', []))
        if ui.CollapsingHeader('All xApps / ON, OFF and unavailable states'):
            table('all-xapp-states', ['xApp', 'State', 'Cells'], [
                [r['instance_id'], r['state'], ', '.join(r['cells']) or '--'] for r in sorted(states, key=lambda r: not r['enabled'])])
        tree('Effective controls / all runtime fields', [dict(cell=c['cell_id'],
            effective=c['effective'], scheduler=c['scheduler']) for c in d.get('cell_metrics', [])])
        tree('Detected xApp catalog / requirements and availability', catalog)
        ui.Separator()
        ui.TextColored(ACCENT, 'LIVE SEGMENTS / WHOLE NETWORK')
        table('test-segments', ['Phase', 'Sample ms', 'Mbps', 'Queue B', 'Delay ms'], [
            [s['label'], fmt(s['duration_ms'], 1), fmt(s['throughput_mbps']),
             fmt(s['mean_queue_bytes']), fmt(s['mean_completion_delay_ms'])] for s in x['segments']])
        ui.TextWrapped('Segments omit transition warm-up. Live before/after is observational: mobility, traffic, HARQ and queues keep evolving. Delay covers completed packets only.')
        tree('Segment xApp identities', [dict(phase=s['label'], app=s.get('app_id'),
            start_slot=s['start_slot']) for s in x['segments']])
        ui.Separator()
        ui.TextColored(ACCENT, 'MATCHED OFF / ON REPLAY')
        ui.TextWrapped('Starts two fresh runs with the same scene, seed, traffic and initial queues. This can take minutes on AGX and shares compute with the live run.')
        if self.state_path and selected.get('available') and ui.Button('Run matched A/B'):
            try:
                pid = start_comparison(self.state_path, d, cell, app)
                self.test_message = f'Matched comparison started (worker {pid}).'
            except (OSError, ValueError) as error:
                self.test_message = str(error)
        if self.state_path:
            result_path = self.state_path.with_suffix('.comparison.json')
            if result_path.exists():
                try:
                    report = json.loads(result_path.read_text())
                    ui.TextUnformatted('Comparison: ' + report.get('stage', 'unknown'))
                    if report.get('app_id'):ui.TextWrapped('Compared xApp: ' + report['app_id'])
                    if report.get('stage', '').startswith('running_'):
                        from pathlib import Path
                        arm = report['stage'].removeprefix('running_')
                        progress_path = Path(report['output']) / arm / 'live.json'
                        if progress_path.exists():
                            progress = json.loads(progress_path.read_text())
                            ui.TextUnformatted(f'{arm.upper()}: slot {progress.get("slot", 0)} / {progress.get("total_slots", "--")}')
                    if report.get('error'):ui.TextColored(WARN, report['error'])
                    arms = report.get('arms', {})
                    table('paired-results', ['Arm', 'Sample ms', 'Mbps', 'Queue B', 'Delay ms'], [
                        [arm.upper(), fmt(row['metrics']['duration_ms'], 1), fmt(row['metrics']['throughput_mbps']),
                         fmt(row['metrics']['mean_queue_bytes']), fmt(row['metrics']['mean_completion_delay_ms'])]
                        for arm, row in arms.items()])
                    tree('ON minus OFF / absolute differences', report.get('delta_on_minus_off', {}))
                    tree('Executed schedulers / evidence', {arm: row.get('executed_schedulers', {}) for arm, row in arms.items()})
                    ui.TextWrapped(report.get('interpretation', 'One paired seed tests functionality; it does not establish general improvement.'))
                    ui.TextWrapped('Results: ' + report.get('output', '--'))
                except (OSError, ValueError, KeyError) as error:
                    ui.TextWrapped(str(error))

    def trends(self, d):
        history = d.get('history', [])
        if not history:
            ui.TextUnformatted('Waiting for actual completed-slot samples...')
            return
        lo, hi = history[0]['sim_time_ms'], history[-1]['sim_time_ms']
        ui.TextColored(MUTED, f'Simulation interval: {lo:.3f} - {hi:.3f} ms | {len(history)} completed slots (last 300)')
        for label, key, unit in [('Throughput', 'throughput_mbps', 'Mbps'), ('MAC queue', 'queue_bytes', 'bytes')]:
            values = np.array([r[key] for r in history], dtype=np.float32)
            ui.TextUnformatted(f'{label} / {unit}')
            ui.PlotLines('##'+key, values, overlay_text=f'latest {values[-1]:,.3f} {unit}', scale_min=0., graph_size=(-1., 62.))
        attempts = d.get('decode_attempts', 0)
        ui.TextUnformatted(f'Observed NACK ratio: {d.get("nacks", 0) / attempts:.2%} / {attempts} actual decodes' if attempts else 'No actual NR decode samples yet')
        ui.TextWrapped('Gaps in wall-time updates reflect computation time, not missing simulated slots. No replay or synthetic samples.')
