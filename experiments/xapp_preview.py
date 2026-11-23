"""Opt-in shadow inference, sampled at actual RIC decision boundaries."""
import importlib.util
import json
from pathlib import Path

from smo.xapp_preview import capture

ROOT = Path(__file__).resolve().parents[1]


def model_source(configs,root=ROOT):
    """A complete hot model set must preview its actual installed release."""
    apps=configs.get('ric',{}).get('xapps',[])
    expected={'ue_energy_rf','ap_energy_rf','throughput_scheduler_mlp',
        'latency_scheduler_rf','latency_urllc_rf','mec_offload_budget_qgbdt'}
    if len(apps)==6 and {a.get('implementation_id')for a in apps}==expected:
        if any(a.get('parameters',{}).get('mode')!='frozen'for a in apps):
            raise ValueError('A complete hot preview requires six frozen instances')
        paths=[Path(a['parameters']['model_path']).resolve()for a in apps]
        bundles={p.parent.parent for p in paths}
        if len(bundles)!=1:raise ValueError('Hot previews cannot mix model installations')
        bundle=bundles.pop()
        if not bundle.is_relative_to(Path(root).resolve()/'outputs')or not(bundle/'release.json').is_file():
            raise ValueError('Hot preview release is not a complete owned installation')
        return bundle
    return Path(root)/'artifacts/xapps/chongyu/20261117'


class PreviewRecorder:
    def __init__(self, simulator, output, manifest, configs, *, before_slots=()):
        from ric.near_rt.pools import build_pool_xapp
        spec = importlib.util.spec_from_file_location('preview_model_installer', ROOT/'scripts/orin/install-trained-xapps.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        source=model_source(configs,ROOT)
        target = Path(output)/'preview_models'
        installer.install(source, target, ROOT)
        release = installer.load_release(target)
        configured={a.get('implementation_id'):a for a in configs.get('ric',{}).get('xapps',[])}
        if source!=ROOT/'artifacts/xapps/chongyu/20261117':
            for entry in release['models'].values():
                actual=Path(configured[entry['implementation_id']]['parameters']['model_path']).resolve()
                if actual!=source/entry['path']:
                    raise ValueError('Hot preview release differs from the actual frozen model path')
        self.apps = []
        for name, entry in release['models'].items():
            params = dict(entry['runtime_parameters'], mode='frozen', model_path=str(target/entry['path']))
            if name == 'x12_mec':
                params['complete_scheduler_bundle'] = True
            if name == 'x12_urllc':
                params['balanced_start_windows'] = 0
            item=dict(implementation_id=entry['implementation_id'],
                cell_ids=[c['cell_id'] for c in manifest['cells']],
                slot_duration_ms=manifest['slot_duration_ms'], parameters=params)
            if name=='x01':
                item['implementation_variant']=entry.get('implementation_variant','legacy9')
            self.apps.append(build_pool_xapp(item, 'chongyu'))
        self.manager = simulator.controller_manager
        self.period = max(1, round(100/manifest['slot_duration_ms']))
        self.path = Path(output)/'xapp_decision_previews.jsonl'
        if self.path.exists():
            raise ValueError('Use a fresh decision-preview output')
        self.latest = None
        self.before_slots=set(before_slots)
        self.manager.decision_preview_sink = self

    def before_selection(self,manager,observation,slot):
        if slot in self.before_slots:
            self._capture(manager,observation,slot,'before_selection')

    def __call__(self, manager, observation, slot):
        if slot % self.period or self.latest and self.latest.get('slot')==slot:
            return
        self._capture(manager,observation,slot,'before_arbitration')

    def _capture(self,manager,observation,slot,phase):
        registered = manager.instance_selection.apps()
        apps = [registered.get(a.name, a) for a in self.apps]
        try:
            self.latest = capture(manager, observation, slot, apps)
            self.latest['capture_phase']=phase
            with self.path.open('a') as stream:
                stream.write(json.dumps(self.latest, allow_nan=False)+'\n')
        except Exception as error:
            # A shadow monitor must never stop or change the RAN control loop.
            self.latest = dict(schema='oran.xapp-decision-preview.v1', slot=slot,
                status='unavailable', reason='preview_capture_failed',
                error=type(error).__name__, apps=[], deployment_enabled=False)
