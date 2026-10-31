# Chongyu xApp pool

Develop new xApps in this directory. `catalog.json` is intentionally empty:
selecting this pool currently runs the native network with no xApp. It never
falls back to a Yisong implementation. Pools are selected at initialization;
restart the simulation after changing the catalog or switching pools.

An xApp should subclass `ric.near_rt.pools.base.PoolXApp`. The shared base supplies
cell-scoped selection, identity, contract and descriptor plumbing. Implement the
algorithm's `run(observation, policy_snapshot)` and validate its parameters.
Consume the normal delivered observation, not simulator state or PHY truth.
Return `ControlDecision` with `ConfigAction` entries; each action must use
`source_name=self.name`, an allowed `target_cell_id`, and
`action._library_provenance = self.identity(cell_id)`. The normal arbitration,
delayed actuation and withdrawal path remains responsible for execution.

For example, create `my_xapp.py` with this interface (fill in the algorithm
before registering it):

```python
from ric.near_rt.pools.base import PoolXApp

class MyXApp(PoolXApp):
    implementation_id = 'my_xapp'
    category_id = 'my_scheduler'
    description = 'Describe the implemented controller'
    required_kpis = ('queue_bytes',)
    writable_actions = ('scheduler_mode',)

    def run(self, observation, policy_snapshot):
        raise NotImplementedError('Implement the control algorithm here')
```

After implementation, add its explicit entry to `catalog.json`:

```json
{
  "my_xapp": {
    "factory": "ric.near_rt.pools.chongyu.my_xapp:MyXApp",
    "category_id": "my_scheduler",
    "description": "Describe the implemented controller",
    "commands": []
  }
}
```

The class's identity/category and command declaration must match the catalog.
Factories must live in the selected pool's namespace. Unknown pools or IDs fail
explicitly. Registration is not a capability certificate: the existing runtime
KPI/action checks still determine whether the UI can enable an xApp.

```bash
# Start the native baseline and discover only this pool's registered xApps.
python experiments/run_rt_sys.py --platform dgx --scene munich --cells 3 --ues 12 \
  --slots 400 --xapp-pool chongyu --live-xapp-test \
  --output outputs/chongyu_live --live-state outputs/chongyu_live/live.json

# After registering an implementation, optionally start with it ON.
# Add: --xapp-test-app my_xapp --xapp-test-start on
```

On AGX, use its activation environment and `--platform agx`. The launcher
`bash scripts/orin/live.sh --xapp-pool chongyu` passes the same selection through.
The normal default is OFF. An empty pool can run OFF but cannot start ON or run
a matched ON/OFF comparison until a compatible implementation is registered.

For programmatic initialization, use `ric_config['xapp_pool'] = 'chongyu'` with
`type: pool_xapp` entries containing `implementation_id`, `instance_id`, and
`cell_ids`. No process-global registry mutation is needed.
