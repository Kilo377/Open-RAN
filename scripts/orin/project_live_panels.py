"""Read-only operations panels for the same snapshot used by the 3D viewer."""
import json
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as ui

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
    def __init__(self, view='integrated'):
        self.view = view
        self.ue_index = 0
        self.cell_index = 0

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
        ui.TextColored(ACCENT, f'{stage.upper()}   |   {data.get("scene", "loading scene")}   |   {config.get("control_mode", "loading mode")} control   |   READ ONLY')
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
            for label, draw in [('Overview', self.overview), ('UEs', self.ues), ('Cells', self.cells),
                                ('Config', self.config), ('Control', self.control)]:
                active, _ = ui.BeginTabItem(label, True)
                if active:
                    draw(data, focus)
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
        ui.TextUnformatted(f'xApps: {len(apps.get("xapps", []))} / rApps: {len(apps.get("rapps", []))}')
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
