"""Arrival-only URLLC timing factories over the fixed C3 integer means.

No controller, deadline, channel or network state is changed here. Mixed
patterns use the production phased-arrival clock: global slot1 starts the
40-slot high phase; the following low sampler starts at local slot0.
"""
from copy import deepcopy

PATTERNS = ('burst4', 'mixed4', 'mixed5', 'rare6')
MEAN_PACKET_BYTES = {'ue_2': 131, 'ue_6': 103}
PHASED = {'mixed4': (400, 4, 2, 1), 'mixed5': (400, 5, 5, 4),
          'rare6': (2000, 6, 44, 5)}


def arrivals(parent_profile, pattern='burst4'):
    if pattern not in PATTERNS:
        raise ValueError('Unknown URLLC timing pattern')
    expected = {}
    for uid, mean in MEAN_PACKET_BYTES.items():
        expected[uid] = dict(arrival_model='on_off', arrival_rate_bps=mean*4*16000,
            packet_size_bytes=mean, on_duration_slots=40, off_duration_slots=120,
            burst_size_bytes=0, start_offset_slots=139)
    if parent_profile.get('ues') != expected:
        raise ValueError('UR timing refinements require the exact unchanged C3 parent streams')
    if pattern == 'burst4':
        return deepcopy(expected)
    period, peak, low_on, low_off = PHASED[pattern]
    result = {}
    for uid, mean in MEAN_PACKET_BYTES.items():
        result[uid] = dict(arrival_model='phased', packet_size_bytes=mean, burst_size_bytes=0,
            arrival_phases=[
                dict(duration_slots=40, arrival_model='cbr', arrival_rate_bps=peak*mean*16000),
                dict(duration_slots=period-40, arrival_model='on_off', arrival_rate_bps=mean*16000,
                     on_duration_slots=low_on, off_duration_slots=low_off,
                     burst_size_bytes=0, start_offset_slots=0)])
    return result


def metadata(pattern='burst4'):
    if pattern not in PATTERNS:
        raise ValueError('Unknown URLLC timing pattern')
    if pattern == 'burst4':
        return dict(pattern=pattern, parent_trace_unchanged=True,
            description='Exact original C3 20ms ON/60ms OFF, fourfold peak, analytically fixed slot21 first ON')
    period, peak, low_on, low_off = PHASED[pattern]
    low_duration = period-40
    high_packets, low_packets = 40*peak, low_duration*low_on//(low_on+low_off)
    if high_packets+low_packets != period or low_duration % (low_on+low_off):
        raise ValueError('Mixed timing design does not preserve native packet count')
    return dict(pattern=pattern, parent_trace_unchanged=False, arrival_only=True,
        cycle_slots=period, cycle_ms=period//2, first_high_global_slot=1,
        high_phase=dict(duration_slots=40, duration_ms=20, arrival_model='cbr', peak_multiplier=peak,
                        packets_per_cycle=high_packets, burst_size_bytes=0),
        low_phase=dict(duration_slots=low_duration, arrival_model='on_off', on_slots=low_on, off_slots=low_off,
                       one_native_packet_each_on_slot=True, packets_per_cycle=low_packets,
                       burst_size_bytes=0, local_start_slot=0),
        mean_packets_per_slot=1, packet_bytes=deepcopy(MEAN_PACKET_BYTES),
        invariant_windows=dict(before=[41,2040], pilot_after=[2041,4040], validation_after=[2041,6040]),
        description=('Different temporal shape; '+('rare6 has20ms sixfold high and980ms low(44 on/5 off slots); ' if pattern == 'rare6' else '')
            +'identical C3 mean bytes, fixed packet sizes and packet totals in every registered1s/2s observation window. Whole run startup/tail need not contain full cycles.'))
