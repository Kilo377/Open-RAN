"""GPU viewer of actual project geometry and atomically published RT/SYS slots.

Runs in the GUI environment; performs no independent simulation or ray tracing.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import polyscope as ps
from project_live_panels import Dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--view', choices=['integrated', 'scene', 'telemetry'], default='integrated')
    args = parser.parse_args()
    state_path = args.state.resolve()
    ps.set_program_name('Open-RAN | Live Network Workspace')
    ps.set_use_prefs_file(False)
    ps.set_up_dir('z_up')
    ps.set_front_dir('y_front')
    ps.set_ground_plane_mode('none')
    ps.set_background_color((0.035, 0.055, 0.09))
    ps.set_build_default_gui_panels(False)
    ps.set_user_gui_is_on_right_side(False)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.set_max_fps(30)
    ps.set_ui_scale(1.05)
    ps.init()
    ps.set_window_size(1800, 1000)
    dashboard = Dashboard(args.view)
    palette = np.array([[0.1, 0.7, 1.0], [1.0, 0.6, 0.12], [0.75, 0.3, 1.0]])
    current, structures = {}, {}
    last_poll = 0.0
    last_token = None
    read_error = None

    def focus(position):
        if not current.get('cells'):
            return
        if position is None:
            center = np.array([c['position_m'] for c in current['cells']]).mean(axis=0)
            ps.look_at(center + np.array([350, -450, 500]), center)
        else:
            center = np.asarray(position)
            ps.look_at(center + np.array([65, -90, 110]), center)

    def sync(data):
        nonlocal last_token
        mesh_path = data.get('mesh_path')
        if 'mesh' not in structures and mesh_path and Path(mesh_path).is_file():
            with np.load(mesh_path, allow_pickle=False) as mesh:
                structures['mesh'] = ps.register_surface_mesh('Actual Sionna 2.2 scene', mesh['vertices'], mesh['faces'])
                structures['mesh'].set_color((0.55, 0.60, 0.67))
                structures['mesh'].set_smooth_shade(False)
            cells = data['cells']
            xyz = np.array([c['position_m'] for c in cells])
            structures['bs'] = ps.register_point_cloud('Project base stations', xyz)
            structures['bs'].set_radius(4, relative=False)
            structures['bs'].add_color_quantity('Cell identity', palette[np.arange(len(cells)) % 3], enabled=True)
            center = xyz.mean(axis=0)
            ps.look_at(center + np.array([350, -450, 500]), center)
        token = (data.get('slot'), data.get('updated_unix_s'))
        if token == last_token or not data.get('ues'):
            return
        last_token = token
        cells, users = data['cells'], data['ues']
        positions = np.array([u['position_m'] for u in users])
        indices = {c['cell_id']: i for i, c in enumerate(cells)}
        colors = np.array([palette[indices[u['cell_id']] % 3] if u['cell_id'] in indices else [.5, .5, .5] for u in users])
        if 'ues' not in structures:
            structures['ues'] = ps.register_point_cloud('Actual project UEs (cell colors)', positions)
            structures['ues'].set_radius(2.8, relative=False)
        else:
            structures['ues'].update_point_positions(positions)
        structures['ues'].add_color_quantity('Serving cell', colors, enabled=True)
        nodes = np.concatenate([np.array([c['position_m'] for c in cells]), positions])
        edges = np.array([[indices[u['cell_id']], len(cells) + i] for i, u in enumerate(users) if u['cell_id'] in indices], dtype=np.int32).reshape(-1, 2)
        if 'links' not in structures:
            structures['links'] = ps.register_curve_network('Serving associations - NOT ray paths', nodes, edges)
            structures['links'].set_radius(0.12, relative=False)
            structures['links'].set_color((0.3, 0.85, 0.65))
        else:
            structures['links'].update_node_positions(nodes)
            # Re-register when handover changes topology.
            if not np.array_equal(edges, structures['edges']):
                structures['links'] = ps.register_curve_network('Serving associations - NOT ray paths', nodes, edges)
                structures['links'].set_radius(0.12, relative=False)
        structures['edges'] = edges.copy()

    def tick():
        nonlocal last_poll, read_error
        now = time.monotonic()
        if now - last_poll >= 0.15:
            last_poll = now
            try:
                incoming = json.loads(state_path.read_text())
                sync(incoming)
                current.clear()
                current.update(incoming)
                read_error = None
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as exc:
                read_error = str(exc)
        status_path = state_path.parent / 'run_status.json'
        status = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except (ValueError, OSError):
                pass
        dashboard.draw(current, status, read_error, focus)

    ps.set_user_callback(tick)
    ps.show()


if __name__ == '__main__':
    main()
