"""Read-only, atomic telemetry for the actual RT/SYS execution loop."""
import json
import os
from pathlib import Path
import time
from collections import Counter, deque

import numpy as np

from phy.rt_sys.channel import position_at
from experiments.rt_live_state import configuration, cell_rows, control_snapshot, plain


class LivePublisher:
    def __init__(self, path, manifest, provider, devices, feedback_mode, *, configs=None,
                 control_mode='unavailable', platform_profile=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.devices = devices
        self.feedback_mode = feedback_mode
        self.started = time.monotonic()
        self.delivered = self.attempts = self.nacks = 0
        self.latest = {}
        self.configuration = configuration(manifest, configs, control_mode, platform_profile)
        self.history = deque(maxlen=300)
        self.control_history = deque(maxlen=60)
        self.control_counts = Counter()
        self.last_slot = None
        vertices, faces = [], []
        offset = 0
        for obj in provider.backend.scene.objects.values():
            params = provider.backend.mi.traverse(obj.mi_mesh)
            v = np.asarray(params['vertex_positions'], dtype=np.float32).reshape(-1, 3)
            f = np.asarray(params['faces'], dtype=np.int32).reshape(-1, 3)
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        mesh_path = self.path.with_suffix('.mesh.npz')
        tmp = mesh_path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, vertices=np.concatenate(vertices), faces=np.concatenate(faces))
        os.replace(tmp, mesh_path)
        self.mesh_path = str(mesh_path)
        self.finish('initializing')

    def _write(self):
        self.latest['updated_unix_s'] = time.time()
        self.latest['wall_s'] = time.monotonic() - self.started
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.latest, allow_nan=False))
        os.replace(tmp, self.path)

    def publish(self, simulator, adapter):
        state = simulator.state
        if state.slot_id == self.last_slot:
            return
        rows = {r['ue_id']: r for r in adapter.trace['ues']}
        self.delivered += sum(r['delivered_bytes'] for r in rows.values())
        decoded = [r for r in rows.values() if r.get('actual_decode')]
        self.attempts += len(decoded)
        self.nacks += sum(not r['ack'] for r in decoded)
        users = []
        for spec, user in zip(self.manifest['ues'], state.ues):
            assert spec['ue_id'] == user.ue_id
            row = rows.get(user.ue_id, {})
            pos, _ = position_at(spec['trajectory'], state.sim_time_ms / 1000)
            connection = state.connections[user.ue_id]
            ledger = getattr(user, 'packet_ledger', None)
            packet = plain(ledger.summary()) if ledger is not None else {}
            users.append(dict(ue_id=user.ue_id, position_m=pos.tolist(),
                cell_id=connection.serving_cell_id, connection_state=connection.connection_state,
                queue_bytes=int(user.queue_bytes), scheduled=bool(row.get('scheduled', False)),
                sinr_db=float(row['sinr_db']) if row.get('scheduled') else None,
                mcs=int(row.get('mcs', 0)), ack=bool(row['ack']) if row.get('scheduled') else None,
                delivered_bytes=int(row.get('delivered_bytes', 0)),
                throughput_mbps=float(row.get('throughput_mbps', 0)),
                packet=packet, slice_id=getattr(user, 'slice_id', None),
                traffic_type=getattr(user, 'traffic_type', None),
                allocated_prbs=getattr(user, 'allocated_prbs', None),
                allocated_symbols=getattr(user, 'allocated_symbols', None),
                rank=getattr(user, 'rank', None),
                harq=plain(dict(pending_retx_bytes=getattr(getattr(user, 'harq', None), 'pending_retx_bytes', None),
                    is_retx=bool(row.get('is_retx', False)))),
                handover_count=getattr(connection, 'handover_count', None),
                pending_target_cell=getattr(connection, 'handover_pending_target_cell_id', None)))
        cells = cell_rows(state, adapter.trace, users)
        control = control_snapshot(simulator)
        for event in control['current_outcomes']:
            self.control_history.append({**event, 'slot_id': int(state.slot_id)})
            self.control_counts[event.get('status', 'unknown')] += 1
        control.update(history=list(self.control_history), totals=dict(self.control_counts))
        packet_keys = ('offered_bytes', 'delivered_bytes', 'pending_bytes', 'overflow_bytes',
            'completed_packets', 'late_completed_packets', 'deadline_missed_packets', 'completion_latency_ms_sum')
        packet_totals = {key: sum(u['packet'].get(key, 0) for u in users) for key in packet_keys}
        completed = packet_totals['completed_packets']
        packet_totals['mean_completion_latency_ms'] = packet_totals['completion_latency_ms_sum'] / completed if completed else None
        packet_totals['late_completed_fraction'] = packet_totals['late_completed_packets'] / completed if completed else None
        self.last_slot = state.slot_id
        self.latest.update(stage='simulating', slot=int(state.slot_id),
            sim_time_ms=float(state.sim_time_ms), ues=users,
            delivered_bytes=self.delivered, decode_attempts=self.attempts, nacks=self.nacks,
            throughput_mbps=self.delivered * 8 / max(state.sim_time_ms * 1000, 1),
            slot_throughput_mbps=sum(u['delivered_bytes'] for u in users) * 8 /
                (self.manifest['slot_duration_ms'] * 1000),
            queue_bytes=sum(u['queue_bytes'] for u in users),
            cell_metrics=cells, control=control, packet_totals=packet_totals,
            configuration=self.configuration,
            connected_ues=sum(u['connection_state'] == 'connected' for u in users))
        self.history.append(dict(slot=int(state.slot_id), sim_time_ms=float(state.sim_time_ms),
            throughput_mbps=self.latest['slot_throughput_mbps'], queue_bytes=self.latest['queue_bytes'],
            nack_fraction=self.nacks / self.attempts if self.attempts else None,
            mean_completion_latency_ms=packet_totals['mean_completion_latency_ms']))
        self.latest['history'] = list(self.history)
        self._write()

    def finish(self, stage, error=None):
        self.latest.update(stage=stage, scene=self.manifest['scene'], mesh_path=self.mesh_path,
            total_slots=self.manifest['total_slots'], cells=self.manifest['cells'],
            execution_devices=self.devices, feedback_mode=self.feedback_mode)
        self.latest['configuration'] = self.configuration
        if error:
            self.latest['error'] = error
        self._write()
