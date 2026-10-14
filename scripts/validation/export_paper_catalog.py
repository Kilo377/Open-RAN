"""Export interface facts without instantiating or enabling any xApp policy."""
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.action import ConfigAction, CommandAction
from phy.rt_sys.control import PHY, SCHEDULER, ASSOCIATION, ADMIN
from phy.rt_sys.paper_controls import FIELDS, COMMANDS
from ric.xapp_contract import KPI_REGISTRY, capability_record


def main():
    path = ROOT / 'docs/rt_sys_kpi_and_xapp_catalog.json'
    catalog = json.loads(path.read_text())
    catalog['kpis'] = {name: asdict(spec) for name, spec in KPI_REGISTRY.items()}
    for record in catalog['xapps']:
        actions = record['writable_actions'] + record['unavailable_actions']
        record.update(capability_record(record['xapp_name'], actions))
    catalog['notes'] = [
        'Registry membership does not imply a valid sample; inspect per-record validity and age.',
        'Paper controls and reports require paper_interfaces.enabled; connection rates additionally require connection_model.enabled.',
        'Existing xApp algorithms and required KPI declarations are not changed by this export.',
        'whole_device_* models are unavailable; reported CQI is RT reference SINR quantization, not calibrated NR CSI.',
    ]
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    categories = {'phy': PHY, 'scheduler': SCHEDULER, 'association': ASSOCIATION, 'administrative': ADMIN}
    controls = {
        f.name: dict(type=str(f.type), consumers=[name for name, members in categories.items() if f.name in members],
                     feature_gate='paper_interfaces.enabled' if f.name in FIELDS else None)
        for f in fields(ConfigAction)
    }
    detail = dict(
        config_actions=controls,
        commands={name: dict(feature_gate='paper_interfaces.enabled', consumer='PaperEnvironment.consume_commands',
                             fields={f.name: str(f.type) for f in fields(CommandAction)}) for name in sorted(COMMANDS)},
        kpis=catalog['kpis'], notes=catalog['notes'],
        semantics_document='docs/PAPER_INTERFACES.md', paper_traceability='docs/PAPER_FIELD_TRACEABILITY.md',
    )
    (ROOT / 'docs/paper_interface_catalog.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
