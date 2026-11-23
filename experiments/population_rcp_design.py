"""Matched group-wide RCP interventions; aggregate outcomes across all UEs."""
from pathlib import Path
from types import SimpleNamespace

from experiments.rcp_perturbation import POPULATION_SCHEMA,population_memberships,validate


def design(root):
    from phy.rt_sys.scenario import build_case
    from experiments.run_rt_sys import configure_ue_services
    m,c,_=build_case(scene='munich',num_cells=3,num_ues=6,total_slots=1600,seed=20261116,
        config_path=Path(root)/'configs/rt_sys/agx_munich_6ue_depth2_fast.json',xapp_pool='chongyu')
    configure_ue_services(c,m,SimpleNamespace(ue_services='drx-mec',ue_services_config=Path(root)/'configs/rt_sys/ue_services.json'))
    groups,drx,offload=population_memberships(c,m)
    if set(offload.values())!={.5} or {drx[u]['inactivity_ms']for u in groups['drx']}!={8}:
        raise ValueError('The matched population pilot requires the released native operating point')
    reference=dict(schema=POPULATION_SCHEMA,arm='native',switch_ms=100,scheduler_mode='round_robin',
        cell_sleep_mode='active',groups=groups,native_drx=drx,native_offload=offload,
        offload_probability=.5,drx_inactivity_ms=8.,mec_weight=1.,urllc_weight=1.)
    arms=dict(native={},offload_low=dict(offload_probability=.4),offload_high=dict(offload_probability=.6),
        drx_short=dict(drx_inactivity_ms=4.),ap_micro=dict(cell_sleep_mode='micro_sleep'),
        latency_native=dict(scheduler_mode='latency_aware'),qos_native=dict(scheduler_mode='qos_deadline'),
        weight_mec_low=dict(scheduler_mode='latency_aware',mec_weight=.75),
        weight_mec_high=dict(scheduler_mode='latency_aware',mec_weight=1.25),
        weight_urllc_low=dict(scheduler_mode='qos_deadline',urllc_weight=.75),
        weight_urllc_high=dict(scheduler_mode='qos_deadline',urllc_weight=1.25))
    contrasts=[
        dict(id='offload_population',parameter='ue_offload_fraction',low='offload_low',high='offload_high',reference='native',step=.2,
            kind='local_derivative',parameter_component='mec_workload',level_field='offload_probability',
            scope='All MEC task sources; one common offloading probability',context='round_robin; offload probability .5±.1; native DRX; active APs'),
        dict(id='drx_population',parameter='ue_drx_configs',low='native',high='drx_short',reference='native',step=1.,
            kind='finite_contrast',parameter_component='drx_enabled_population',level_field='drx_inactivity_ms',
            scope='All DRX-enabled UEs; common inactivity timer 8→4 ms; native DRX-disabled traffic unchanged',
            context='round_robin; native MEC offload; active APs; experimental timer unlock; other timers held fixed',
            **{'from':{'inactivity_ms':8.},'to':{'inactivity_ms':4.}}),
        dict(id='ap_sleep',parameter='cell_sleep_mode',low='native',high='ap_micro',reference='native',step=1.,
            kind='finite_contrast',parameter_component='all_cells',level_field='cell_sleep_mode',scope='All cells: active→micro_sleep',
            context='round_robin; native DRX and MEC offload; micro profile 1 ms per 20 ms',**{'from':'active','to':'micro_sleep'}),
        dict(id='scheduler_latency',parameter='scheduler_mode',low='native',high='latency_native',reference='native',step=1.,
            kind='finite_contrast',parameter_component='all_cells',level_field='scheduler_mode',scope='All cells; all UE weights held at 1',
            context='round_robin→latency_aware; native DRX and MEC offload; active APs',**{'from':'round_robin','to':'latency_aware'}),
        dict(id='scheduler_qos',parameter='scheduler_mode',low='native',high='qos_native',reference='native',step=1.,
            kind='finite_contrast',parameter_component='all_cells',level_field='scheduler_mode',scope='All cells; all UE weights held at 1',
            context='round_robin→qos_deadline; native DRX and MEC offload; active APs',**{'from':'round_robin','to':'qos_deadline'}),
        dict(id='weight_mec_population',parameter='ue_scheduler_weights',low='weight_mec_low',high='weight_mec_high',reference='latency_native',step=.5,
            kind='local_derivative',parameter_component='mec_priority',level_field='mec_weight',
            scope='Entire MEC-workload group; common relative priority; other traffic weights fixed at 1',
            context='latency_aware held fixed; MEC priority 1±.25; native DRX and offload; active APs'),
        dict(id='weight_urllc_population',parameter='ue_scheduler_weights',low='weight_urllc_low',high='weight_urllc_high',reference='qos_native',step=.5,
            kind='local_derivative',parameter_component='urllc_priority',level_field='urllc_weight',
            scope='Entire URLLC traffic group; common relative priority; other traffic weights fixed at 1',
            context='qos_deadline held fixed; URLLC priority 1±.25; native DRX and offload; active APs'),
    ]
    for c in contrasts:
        c['population_control_eligible']=True
    return reference,arms,contrasts
