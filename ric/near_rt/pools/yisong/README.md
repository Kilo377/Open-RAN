# Yisong xApp pool

All xApps that existed before the pool split belong to `yisong`, including the
live queue/HOL test. This directory is their ownership and discovery entry point.
Their existing implementation paths remain compatible:

- `ric/near_rt/xapp_library/`: implementation catalog, instances and algorithms.
- `ric/near_rt/rule_based_v1/`: rule implementations and shared control machinery.
- `ric/near_rt/*_xapp.py`: original xApp classes, variants and the live test.

Run the existing pool with `--xapp-pool yisong`. Registration does not certify
that an implementation satisfies a particular backend's KPI/action contract.
