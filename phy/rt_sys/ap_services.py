"""Opt-in, slot-clock AP power and real sleep/wake availability.

These are explicit system-level scenario assumptions, not hardware readings.
Awake RF power is unchanged. Sleep and wake intervals transmit neither PDSCH
nor the reference-signal hypothesis; packet queues and HARQ are not reset.
The power ledger includes static, load, PA, sleep, waking and wake-transition
energy separately. Numerical profiles must be supplied by the scenario.
"""
from collections import Counter
from copy import deepcopy
import math
from numbers import Real


MODES = ('active', 'micro_sleep', 'light_sleep', 'deep_sleep')
PERIODIC = ('micro_sleep', 'light_sleep')
SLEEP_MODES = MODES[1:]
POWER_KEYS = {'static_w', 'load_w', 'pa_efficiency', 'waking_w', 'sleep_w', 'wake_transition_j'}


def number(value, name, minimum=0., maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('invalid_ap_' + name)
    return float(value)


def radio_available(cell):
    """The disabled path deliberately keeps the historical deep-sleep rule."""
    if getattr(cell, 'ap_service_enabled', False):
        return bool(cell.ap_available)
    return cell.cell_sleep_mode != 'deep_sleep'


class ApServices:
    def __init__(self, state, config, slot_duration_ms):
        if not isinstance(config, dict) or set(config) != {'enabled', 'power', 'profiles'} or config.get('enabled') is not True:
            raise ValueError('AP services require explicit enabled, power and profiles')
        self.state = state
        self.config = deepcopy(config)
        self.dt_ms = number(slot_duration_ms, 'slot_duration_ms', 1e-9)
        self.dt = self.dt_ms / 1000
        self.power = deepcopy(config['power'])
        if not isinstance(self.power, dict) or set(self.power) != POWER_KEYS:
            raise ValueError('invalid_ap_power_fields')
        for key in ('static_w', 'load_w', 'waking_w'):
            number(self.power[key], key)
        number(self.power['pa_efficiency'], 'pa_efficiency', 1e-9, 1.)
        for key in ('sleep_w', 'wake_transition_j'):
            if not isinstance(self.power[key], dict) or set(self.power[key]) != set(SLEEP_MODES):
                raise ValueError('invalid_ap_' + key)
            for mode, value in self.power[key].items():
                number(value, key + '_' + mode)
        self.profiles = deepcopy(config['profiles'])
        if not isinstance(self.profiles, dict) or set(self.profiles) != set(SLEEP_MODES):
            raise ValueError('invalid_ap_profile_modes')
        self.profile_slots = {}
        for mode, profile in self.profiles.items():
            keys = {'wakeup_ms'} | ({'period_ms', 'sleep_ms'} if mode in PERIODIC else set())
            if not isinstance(profile, dict) or set(profile) != keys:
                raise ValueError('invalid_ap_profile_fields')
            slots = {}
            for key, value in profile.items():
                n = number(value, mode + '_' + key) / self.dt_ms
                if not math.isclose(n, round(n), abs_tol=1e-8, rel_tol=1e-10):
                    raise ValueError('ap_timing_must_align_to_slots')
                slots[key.replace('_ms', '_slots')] = round(n)
            if mode in PERIODIC and not 0 < slots['sleep_slots'] < slots['sleep_slots'] + slots['wakeup_slots'] + 1 <= slots['period_slots']:
                raise ValueError('invalid_ap_periodic_profile')
            self.profile_slots[mode] = slots
        # The public measurement cycle spans complete periods of either arm.
        self.period_slots = math.lcm(*(self.profile_slots[m]['period_slots'] for m in PERIODIC))
        self.slot = 0
        self.accounted_slot = 0
        self.cells = {}
        self.reports = {}
        for cid, runtime in state.managed_cells.items():
            cell = runtime.cell
            if cell.cell_sleep_mode not in MODES:
                raise ValueError('invalid_initial_ap_policy')
            self.cells[cid] = dict(policy=cell.cell_sleep_mode, owner=None,
                applied_slot=None, request_id=None, generation=0, runtime='active',
                sleep_mode=None, wake_until_slot=0, pending_transition_j=0.,
                energy=Counter(), slots=Counter(), wake_transitions=0,
                last_power=0., last_load=0., delivered_bits=0)
            cell.ap_service_enabled = True
            cell.ap_available = True
            cell.ap_runtime_state = 'active'
        self.begin_slot(0)
        self._publish()

    def validate_policy(self, mode):
        if not isinstance(mode, str) or mode.strip().lower() not in MODES:
            raise ValueError('invalid_ap_sleep_mode')
        return mode.strip().lower()

    def set_policy(self, cid, mode, slot, *, owner=None, request_id=None):
        """Called only by the real executor, including same-valued ownership ACKs."""
        mode = self.validate_policy(mode)
        if type(slot) is not int or slot < self.slot or cid not in self.cells:
            raise ValueError('invalid_ap_policy_time_or_cell')
        if owner is not None and not isinstance(owner, str):
            raise ValueError('invalid_ap_policy_owner')
        record = self.cells[cid]
        if mode != record['policy'] or owner != record['owner'] or record['applied_slot'] is None:
            record.update(policy=mode, owner=owner, applied_slot=slot,
                request_id=request_id, generation=record['generation'] + 1)
        self.state.managed_cells[cid].cell.cell_sleep_mode = mode
        self.begin_slot(slot)

    def _desired(self, record, phase):
        policy = record['policy']
        if policy == 'active':
            return 'active'
        if policy == 'deep_sleep':
            return 'sleeping'
        profile = self.profile_slots[policy]
        phase %= profile['period_slots']
        if phase < profile['sleep_slots']:
            return 'sleeping'
        if phase < profile['sleep_slots'] + profile['wakeup_slots']:
            return 'waking'
        return 'active'

    def begin_slot(self, slot):
        if type(slot) is not int or slot < self.slot:
            raise ValueError('invalid_ap_slot')
        self.slot = slot
        phase = max(0, slot - 1) % self.period_slots
        for cid, record in self.cells.items():
            desired = self._desired(record, phase)
            previous = record['runtime']
            if desired == 'sleeping':
                record.update(runtime='sleeping', sleep_mode=record['policy'], wake_until_slot=0)
            elif previous == 'sleeping':
                source = record['sleep_mode']
                delay = self.profile_slots[source]['wakeup_slots']
                record['wake_transitions'] += 1
                record['pending_transition_j'] += self.power['wake_transition_j'][source]
                record.update(runtime='waking' if delay else 'active', wake_until_slot=slot + delay)
            elif previous == 'waking':
                if slot >= record['wake_until_slot']:
                    record['runtime'] = 'active'
            # An already-awake AP does not need a fabricated wake-up if a new
            # periodic policy is first requested in its scheduled wake phase.
            cell = self.state.managed_cells[cid].cell
            cell.ap_available = record['runtime'] == 'active'
            cell.ap_runtime_state = record['runtime']

    def before_scheduling(self):
        self.begin_slot(self.state.slot_id)

    def _public(self, cid):
        record = self.cells[cid]
        slots = record['slots']
        total = sum(slots.values())
        index = max(0, self.slot - 1) // self.period_slots
        return dict(ap_energy_total_j=sum(record['energy'].values()), ap_power_w=record['last_power'],
            ap_energy_components_j=dict(record['energy']), ap_load_fraction=record['last_load'],
            ap_available_fraction=slots['active'] / total if total else 1.,
            ap_available_slots_total=slots['active'], ap_sleep_slots_total=slots['sleeping'],
            ap_waking_slots_total=slots['waking'], ap_wake_transitions_total=record['wake_transitions'],
            ap_state=record['runtime'], ap_available=record['runtime'] == 'active', ap_policy=record['policy'],
            ap_policy_owner=record['owner'], ap_policy_applied_slot=record['applied_slot'],
            ap_policy_request_id=record['request_id'], ap_policy_generation=record['generation'],
            ap_delivered_bits_total=record['delivered_bits'],
            ap_cycle_period_slots=self.period_slots, ap_cycle_index=index,
            ap_cycle_start_slot=index * self.period_slots + 1,
            ap_cycle_phase_slot=max(0, self.slot - 1) % self.period_slots)

    def _publish(self):
        self.reports = {cid: self._public(cid) for cid in self.cells}

    def policy_receipt(self, cid):
        """Current actual consumer state, without advancing an energy counter."""
        return deepcopy(self._public(cid))

    def after_phy(self, trace):
        if self.slot != self.state.slot_id or self.slot != self.accounted_slot + 1:
            raise ValueError('ap_energy_requires_each_slot_exactly_once')
        if set(trace.get('cells', {})) != set(self.cells):
            raise ValueError('ap_energy_requires_all_cell_rf_counters')
        output = {}
        for cid, record in self.cells.items():
            radio = trace['cells'][cid]
            rf = number(radio['average_used_w'], 'emitted_rf_power_w')
            occupied = number(radio.get('occupied_re', 0), 'occupied_re')
            grid = number(radio.get('active_grid_re', radio.get('grid_re', 0)), 'grid_re', 1.)
            load = number(occupied / grid, 'load_fraction', 0., 1.)
            if record['runtime'] != 'active' and (rf or occupied):
                raise ValueError('unavailable_ap_emitted_radio_resources')
            components = dict.fromkeys(('static', 'load', 'pa', 'sleep', 'waking', 'transition'), 0.)
            if record['runtime'] == 'active':
                components.update(static=self.power['static_w'] * self.dt,
                    load=self.power['load_w'] * load * self.dt,
                    pa=rf / self.power['pa_efficiency'] * self.dt)
            elif record['runtime'] == 'sleeping':
                components['sleep'] = self.power['sleep_w'][record['sleep_mode']] * self.dt
            else:
                components['waking'] = self.power['waking_w'] * self.dt
            components['transition'] = record['pending_transition_j']
            record['pending_transition_j'] = 0.
            record['energy'].update(components)
            record['slots'][record['runtime']] += 1
            record['last_power'] = sum(components.values()) / self.dt
            record['last_load'] = load
            record['delivered_bits'] += sum(number(u.get('delivered_bytes', 0), 'delivered_bytes') * 8
                for u in trace.get('ues', []) if u['cell_id'] == cid)
            output[cid] = dict(self._public(cid), slot_energy_j=sum(components.values()),
                slot_energy_components_j=components, emitted_rf_power_w=rf)
        self.accounted_slot = self.slot
        self._publish()
        trace['ap_services'] = dict(cells=output,
            scope='modeled AP static + load + PA + sleep + waking + wake-transition energy; not host-device power')

    def summary(self):
        return dict(enabled=True, config=deepcopy(self.config), accounted_slots=self.accounted_slot,
            cells=deepcopy(self.reports), network_ap_energy_total_j=sum(r['ap_energy_total_j'] for r in self.reports.values()),
            scope='modeled AP static + load + PA + sleep + waking + wake-transition energy; not host-device power')
