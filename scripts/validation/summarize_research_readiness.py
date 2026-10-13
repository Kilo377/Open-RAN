"""Finalize evidence only after every declared GPU validation has completed."""
import csv, json, sys, tarfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def cell_diagnostics(folder):
    from phy.nr_reference import lookup_nr_max_prbs
    config = json.loads((folder / 'config_snapshot.json').read_text())
    steady = json.loads((folder / 'steady_state.json').read_text())
    manifest = config['manifest']
    nprb, _ = lookup_nr_max_prbs(**{k: manifest[k] for k in
                                    ('bandwidth_hz', 'subcarrier_spacing_hz', 'carrier_frequency_hz')})
    cells = {c['cell_id']: Counter() for c in manifest['cells']}
    previous, latest = {}, {}
    for row in csv.DictReader((folder / 'ue_metrics.csv').open()):
        uid, cell = row['ue_id'], row['cell_id']
        offered = int(row['packet_offered_bytes'])
        if int(row['slot_id']) * manifest['slot_duration_ms'] > steady['warmup_ms']:
            cells[cell]['offered_bytes'] += offered - previous.get(uid, 0)
            cells[cell]['delivered_bytes'] += int(row['delivered_bytes'])
            cells[cell]['scheduled'] += row['scheduled'] == 'True'
            cells[cell]['capacity_outage'] += row['ack_source'] == 'capacity_outage_no_tb'
            cells[cell]['no_path'] += row['ack_source'] == 'no_path'
            cells[cell]['amc_unmet'] += row['amc_selection_target_met'] == 'False'
        previous[uid] = offered
        latest[uid] = row
    for row in latest.values():
        cells[row['cell_id']]['final_ues'] += 1
        cells[row['cell_id']]['pending_bytes'] += int(row['packet_pending_bytes'])
    for row in csv.DictReader((folder / 'cell_metrics.csv').open()):
        if int(row['slot_id']) * manifest['slot_duration_ms'] > steady['warmup_ms']:
            cells[row['cell_id']]['occupied_re_sum'] += int(row['occupied_re'])
            cells[row['cell_id']]['evaluated_slots'] += 1
    result = {}
    for cell, counts in cells.items():
        result[cell] = dict(counts, offered_mbps=counts['offered_bytes']*.008/steady['evaluated_ms'],
                            delivered_mbps=counts['delivered_bytes']*.008/steady['evaluated_ms'],
                            active_grid_occupancy=counts['occupied_re_sum']/
                            (max(1, counts['evaluated_slots'])*nprb*12*manifest['num_symbols']))
    assert sum(c['offered_bytes'] for c in cells.values()) == sum(u['offered_bytes'] for u in steady['per_ue'].values())
    assert sum(c['delivered_bytes'] for c in cells.values()) == sum(u['delivered_bytes'] for u in steady['per_ue'].values())
    return result


def main():
    out = ROOT / 'outputs/rt_research_readiness_20261013/final_ran'
    source = source_fingerprint(ROOT)
    files = dict(links='link_validation_optimized.json',
                 controls='control_lifecycle_verified/validation.json',
                 controls_long='control_lifecycle_long/validation.json',
                 initial_fairness='initial_fairness/validation.json',
                 finite_capacity='long_baselines_verified/validation.json')
    results = {name: json.loads((out / path).read_text()) for name, path in files.items()}
    assert all(r['passed'] and r['source_sha256'] == source for r in results.values())
    assert '252 passed' in (out / 'regression_local.log').read_text()
    assert '253 passed' in (out / 'regression_spark.log').read_text()
    from scripts.validation.validate_environment_repairs import audit
    payload_audits = [audit(out / 'long_baselines_verified' / case['case'])
                      for case in results['finite_capacity']['cases']]
    archive_path = out / f'executed_sources_{source[:12]}.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        folders={'core','phy','ran','ric','state','models','traffic','experiments','configs','scripts','tests','docs'}
        folders.update(p.name for p in ROOT.iterdir() if p.is_dir() and (p/'__init__.py').exists())
        for folder in sorted(folders):
            for p in (ROOT / folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.md'):
                    archive.add(p, arcname=str(p.relative_to(ROOT)))
        for p in [ROOT / 'README.md', *ROOT.glob('requirements*.txt')]:
            if p.is_file(): archive.add(p, arcname=p.name)
    report = dict(passed=True, source_sha256=source, validation_files=files,
                  regression=dict(local_passed=252, local_skipped=1, spark_passed=253, subtests_passed=2),
                  clocks_unchanged=dict(slot_ms=.5, xapp_slots=5, rapp_slots=10), results=results,
                  resource_payload_audits=payload_audits,
                  cell_diagnostics={case['case']: cell_diagnostics(out / 'long_baselines_verified' / case['case'])
                                    for case in results['finite_capacity']['cases']},
                  limitations=['Finite stationary capacity check, not asymptotic stability or moving generalization',
                               'Local BLER bin domain; unknown profiles require cold NR calibration',
                               'Statistical feedback is not actual CRC feedback or waveform soft combining',
                               'RV0 Chase implemented; IR and strict URLLC tail validation remain incomplete',
                               'Ideal CSI / Gaussian post-LMMSE resources; real A1/E2 and full OFDM deferred'],
                  preserved_partial_runs=['../long_baselines', '../long_baselines_final', '../long_baselines_verified', '../scoped_mobility_diagnostics'],
                  historical_versions='Older link/control reports and source archives are retained, not merged into final acceptance',
                  regression_collection_diagnostic='An unrestricted pytest scan collected archived candidate tests; corrected final command is pytest -q tests')
    (out / 'validation_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=True, source_sha256=source,
                          cases=results['finite_capacity']['cases']), indent=2))


if __name__ == '__main__':
    main()
