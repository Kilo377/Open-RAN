"""UE-map leases restore original UID policies without stale source-cell writes."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from models.action import ConfigAction
from ric.near_rt.rule_based_v1.leases import RuleLeases


DRX = dict(enabled=True, cycle_ms=40., on_duration_ms=4., inactivity_ms=8., wakeup_ms=.5, offset_ms=0.)
FIELDS = ['ue_drx_configs', 'ue_offload_fraction']


def manager(names=('energy',), allowed=('cell_0', 'cell_1'), priorities=None):
    apps = [NS(name=name, rule_based_v1=True, library_instance=False,
               allowed_cells=allowed, controlled_fields=FIELDS,
               priority=(priorities or {}).get(name,20)) for name in names]
    enabled = set(names)
    result = NS(e2_interfaces=[NS(xapp=app) for app in apps],
                current_action_by_scope={}, enabled=enabled,
                instance_selection=NS(mode='exclusive'))
    result._is_xapp_enabled=lambda name:name in result.enabled
    return result


def observation(association=None, drx=None, fractions=None, include_baseline=True):
    association = association or {'ue_0':'cell_0'}
    drx = drx or {uid:dict(DRX) for uid in association}
    fractions = fractions or {uid:.5 for uid in association}
    cells={cid:{'executed_config':{field:{} for field in FIELDS}}
           for cid in set(association.values()) | {'cell_0','cell_1'}}
    if include_baseline:
        for uid,cid in association.items():
            cells[cid]['executed_config']['ue_drx_configs'][uid]=deepcopy(drx[uid])
            cells[cid]['executed_config']['ue_offload_fraction'][uid]=fractions[uid]
    return NS(slot_id=1,
              ue_observations=[NS(ue_id=uid,serving_cell_id=cid) for uid,cid in association.items()],
              paper_report={'values':{'cell':cells,'ue':{
                  uid:{'association':{'cell_id':cid}} for uid,cid in association.items()}}})


def request(name='energy', cid='cell_0', uid='ue_0', cycle=80, fraction=1.):
    return ConfigAction(source_name=name,target_cell_id=cid,
                        ue_drx_configs={uid:{'cycle_ms':cycle}},
                        ue_offload_fraction={uid:fraction})


def reconcile(leases, manager, obs, actions=(), extra=None):
    winners=deepcopy(extra or {})
    scopes={key:('cell',key.split(':',1)[1]) for key in winners}
    apps={i.xapp.name:i.xapp for i in manager.e2_interfaces}
    for action in actions:
        scope='cell:'+action.target_cell_id;scopes[scope]=('cell',action.target_cell_id)
        for field in FIELDS:
            value=getattr(action,field)
            if value is not None:
                winner=(apps[action.source_name].priority,deepcopy(value),action.source_name)
                old=winners.setdefault(scope,{}).get(field)
                if old is None or winner[0]>old[0]:winners[scope][field]=winner
    dropped=[]
    leases.reconcile(manager,obs,winners,scopes,list(actions),dropped)
    return winners,dropped


class TestUeEnergyLeases(unittest.TestCase):
    def test_off_restores_canonical_policies_and_waits_for_confirmation(self):
        leases=RuleLeases();m=manager()
        winners,_=reconcile(leases,m,observation(),[request()])
        self.assertEqual(winners['cell:cell_0']['ue_offload_fraction'][1],{'ue_0':1.})
        active=observation(drx={'ue_0':dict(DRX,cycle_ms=80)},fractions={'ue_0':1.})
        m.enabled.clear()
        restored,_=reconcile(leases,m,active)
        self.assertEqual(restored['cell:cell_0']['ue_drx_configs'][1],{'ue_0':DRX})
        self.assertEqual(restored['cell:cell_0']['ue_offload_fraction'][1],{'ue_0':.5})
        self.assertTrue(leases.ue_service_owned)
        again,_=reconcile(leases,m,active)
        self.assertEqual(again,restored)
        confirmed,_=reconcile(leases,m,observation())
        self.assertFalse(leases.ue_service_owned)
        self.assertFalse(any(fields for fields in confirmed.values()))

    def test_handover_prunes_old_requests_and_restores_at_new_allowed_cell(self):
        leases=RuleLeases();m=manager()
        action=request();reconcile(leases,m,observation(),[action])
        m.current_action_by_scope['cell:cell_0']=deepcopy(action)
        moved=observation({'ue_0':'cell_1'},drx={'ue_0':dict(DRX,cycle_ms=80)},fractions={'ue_0':1.})
        held,dropped=reconcile(leases,m,moved)
        self.assertFalse(any(fields for fields in held.values()))
        self.assertIsNone(m.current_action_by_scope['cell:cell_0'].ue_drx_configs)
        self.assertIsNone(m.current_action_by_scope['cell:cell_0'].ue_offload_fraction)
        self.assertTrue(any(d.reason=='ue_service_request_moved_cell' for d in dropped))
        m.enabled.clear()
        restored,_=reconcile(leases,m,moved)
        self.assertEqual(restored['cell:cell_1']['ue_drx_configs'][1],{'ue_0':DRX})
        self.assertEqual(restored['cell:cell_1']['ue_offload_fraction'][1],{'ue_0':.5})
        self.assertNotIn('cell:cell_0',restored)

    def test_one_moved_uid_does_not_block_remaining_map(self):
        leases=RuleLeases();m=manager()
        initial=observation({'ue_0':'cell_0','ue_4':'cell_0'})
        action=request()
        action.ue_drx_configs['ue_4']={'cycle_ms':80}
        action.ue_offload_fraction['ue_4']=1.
        reconcile(leases,m,initial,[action])
        m.current_action_by_scope['cell:cell_0']=deepcopy(action)
        moved=observation({'ue_0':'cell_1','ue_4':'cell_0'})
        held,_=reconcile(leases,m,moved)
        self.assertEqual(held['cell:cell_0']['ue_offload_fraction'][1],{'ue_4':1.})
        self.assertEqual(m.current_action_by_scope['cell:cell_0'].ue_offload_fraction,{'ue_4':1.})

    def test_outside_allowed_cells_is_deferred_until_return(self):
        leases=RuleLeases();m=manager(allowed=('cell_0',))
        reconcile(leases,m,observation(),[request()]);m.enabled.clear()
        moved=observation({'ue_0':'cell_1'},drx={'ue_0':dict(DRX,cycle_ms=80)},fractions={'ue_0':1.})
        restored,dropped=reconcile(leases,m,moved)
        self.assertFalse(any(fields for fields in restored.values()))
        self.assertTrue(leases.ue_service_owned)
        self.assertTrue(any(d.reason=='ue_service_restore_deferred_outside_allowed_cells' for d in dropped))
        returned=observation(drx={'ue_0':dict(DRX,cycle_ms=80)},fractions={'ue_0':1.})
        restored,_=reconcile(leases,m,returned)
        self.assertEqual(restored['cell:cell_0']['ue_offload_fraction'][1],{'ue_0':.5})

    def test_missing_baseline_blocks_new_map(self):
        leases=RuleLeases();m=manager()
        winners,dropped=reconcile(leases,m,observation(include_baseline=False),[request()])
        self.assertFalse(any(fields for fields in winners.values()))
        self.assertTrue(any(d.reason=='ue_service_baseline_unavailable' for d in dropped))

    def test_whole_map_priority_is_unchanged(self):
        leases=RuleLeases();m=manager(('low','high'),priorities={'low':20,'high':30})
        a=request('low',fraction=0.);b=request('high',fraction=1.)
        winners,_=reconcile(leases,m,observation(),[a,b])
        self.assertEqual(winners['cell:cell_0']['ue_offload_fraction'],(30,{'ue_0':1.},'high'))
        m.enabled.remove('high')
        winners,_=reconcile(leases,m,observation(),[])
        self.assertEqual(winners['cell:cell_0']['ue_offload_fraction'],(20,{'ue_0':0.},'low'))

    def test_off_does_not_merge_restore_with_another_live_whole_map(self):
        leases=RuleLeases();m=manager(('old','other'),priorities={'other':30})
        initial=observation({'ue_0':'cell_0','ue_4':'cell_0'})
        reconcile(leases,m,initial,[request('old')])
        active=observation({'ue_0':'cell_0','ue_4':'cell_0'},
                           drx={'ue_0':dict(DRX,cycle_ms=80),'ue_4':dict(DRX,cycle_ms=80)},
                           fractions={'ue_0':1.,'ue_4':1.})
        m.enabled.remove('old')
        winners,dropped=reconcile(leases,m,active,[request('other',uid='ue_4')])
        self.assertEqual(winners['cell:cell_0']['ue_offload_fraction'],(30,{'ue_4':1.},'other'))
        self.assertIn(('ue_offload_fraction','ue_0'),leases.ue_service_owned)
        self.assertTrue(any(d.reason=='ue_service_restore_deferred_field_owned' for d in dropped))
        m.enabled.clear()
        winners,_=reconcile(leases,m,active)
        self.assertEqual(winners['cell:cell_0']['ue_offload_fraction'][1],{'ue_0':.5,'ue_4':.5})

    def test_stale_queued_action_is_not_resurrected_or_mutated(self):
        leases=RuleLeases();m=manager();action=request()
        action._library_provenance={'instance_id':'energy','selection_version':1}
        original=deepcopy(vars(action))
        reconcile(leases,m,observation(),[action]);m.enabled.clear()
        winners,_=reconcile(leases,m,observation(),[action])
        self.assertFalse(any(fields for fields in winners.values()))
        self.assertEqual(vars(action),original)
        self.assertFalse(leases.requests)

    def test_other_control_fields_are_unchanged(self):
        leases=RuleLeases();m=manager()
        raw={'cell:cell_0':{'tx_power_dbm':(50,25.,'native')}}
        winners,_=reconcile(leases,m,observation(),extra=raw)
        self.assertEqual(winners,raw)


if __name__ == '__main__':
    unittest.main()
