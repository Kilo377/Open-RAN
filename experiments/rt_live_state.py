"""Read-only network/operations views of the simulator's own state objects."""
from dataclasses import asdict, is_dataclass
import math

import numpy as np


CELL_FIELDS = ('tx_power_dbm', 'nominal_tx_power_dbm', 'power_cap_dbm', 'cell_sleep_mode',
    'current_codebook_id', 'current_beam_profile_id', 'beam_candidate_preference',
    'traffic_steering_preference', 'mobility_bias_db', 'handover_margin_db',
    'handover_delay_slots', 'handover_cost_db', 'handover_interruption_slots',
    'steering_hysteresis_db', 'measurement_event_trigger', 'measurement_event_offset_db',
    'prb_power_w', 'ue_prb_limits', 'neighbor_cell_ids', 'cio_offsets_db',
    'ret_enabled', 'ret_tilt_deg', 'ret_codebook_id')
SCHEDULER_FIELDS = ('scheduler_mode', 'scheduler_policy_hint', 'packing_strategy',
    'latency_bias', 'slice_weight_adjustment', 'enable_spatial_multiplexing',
    'max_mu_users_per_group', 'max_total_spatial_layers', 'min_mu_beam_separation_deg')
GRID_FIELDS = ('total_prbs', 'total_symbols', 'total_resource_elements',
    'allocated_resource_elements', 'utilization', 'scheduled_layer_count_total',
    'mu_mimo_group_count', 'mu_mimo_ue_count', 'fft_size')


def plain(value):
    """Return owned JSON values; unavailable/non-finite measurements stay null."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def fields(obj, names):
    if isinstance(obj, dict):
        return {key: plain(obj[key]) for key in names if key in obj}
    return {key: plain(getattr(obj, key)) for key in names if hasattr(obj, key)}


def configuration(manifest, configs, control_mode, profile):
    configs = configs or {}
    ric = configs.get('ric', {})
    return dict(control_mode=control_mode, platform=plain(profile or {}),
        radio=fields(manifest, ('carrier_frequency_hz', 'bandwidth_hz', 'subcarrier_spacing_hz',
            'slot_duration_ms', 'num_symbols', 'fft_size', 'tx_rows', 'tx_cols', 'num_rx_ant')),
        propagation=fields(manifest, ('path_update_ms', 'max_geometry_displacement_m',
            'ray_solver', 'cache_format', 'rng_seed')),
        timing=fields(configs.get('sim', {}), ('obs_delay_slots', 'config_actuation_delay_slots',
            'command_actuation_delay_slots', 'measurement_queue_delay_slots', 'measurement_sinr_noise_std_db')),
        phy=fields(configs.get('phy', {}).get('rt_sys', {}), ('feedback_mode', 'bler_target',
            'noise_figure_db', 'rzf_relative_regularization', 'csi_error_nmse_db', 'harq_combining',
            'mobility', 'nr_execution', 'recording_mode')),
        controllers=dict(**fields(ric, ('xapp_interval_slots', 'rapp_interval_slots', 'freeze_learning')),
            xapps=[fields(app, ('name', 'type', 'priority', 'cell_ids')) for app in ric.get('xapps', [])],
            rapps=[fields(app, ('name', 'type', 'priority')) for app in ric.get('rapps', [])]))


def cell_rows(state, trace, users):
    rows = []
    for cid, runtime in getattr(state, 'managed_cells', {}).items():
        served = [u for u in users if u['cell_id'] == cid]
        radio = getattr(runtime, 'radio_profile', {})
        rows.append(dict(cell_id=cid, effective=fields(runtime.cell, CELL_FIELDS),
            scheduler=fields(runtime.scheduler, SCHEDULER_FIELDS),
            grid=fields(runtime.resource_grid, GRID_FIELDS),
            radio=fields(radio, ('carrier_frequency_hz', 'bandwidth_hz', 'subcarrier_spacing_hz')),
            phy_power=plain(trace.get('cells', {}).get(cid, {})),
            associated_ues=len(served), scheduled_ues=sum(u['scheduled'] for u in served),
            queue_bytes=sum(u['queue_bytes'] for u in served),
            slot_throughput_mbps=sum(u['throughput_mbps'] for u in served)))
    return rows


def control_snapshot(simulator):
    state = simulator.state
    control = getattr(state, 'control', None)
    pending = []
    for kind, attr in (('config', '_pending_config_actuations'), ('command', '_pending_command_actuations')):
        for due, sequence, action in getattr(simulator, attr, []):
            pending.append(dict(kind=kind, due_slot=int(due), sequence=int(sequence), action=plain(action)))
    manager = getattr(simulator, 'controller_manager', None)
    selection = getattr(manager, 'instance_selection', None)
    active = [plain(a) for a in getattr(control, 'active_config_actions', [])]
    return dict(pending=sorted(pending, key=lambda v: (v['due_slot'], v['sequence'])),
        current_outcomes=plain(getattr(control, 'actuation_outcomes', [])),
        requested_config=active,
        active_constraints=plain(getattr(control, 'active_policy_constraints', [])),
        arbitration=plain(getattr(control, 'arbitration_trace', None)),
        observation=plain(getattr(control, 'input_observation_trace', {})),
        xapp_bindings=plain(getattr(selection, 'bindings', {})))
