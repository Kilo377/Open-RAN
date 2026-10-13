"""Finalize evidence only after every declared GPU validation has completed."""
import hashlib, json, sys, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def main():
    out = ROOT / 'outputs/rt_research_readiness_20261013'
    source = source_fingerprint(ROOT)
    files = dict(links='link_validation_optimized.json',
                 controls='control_lifecycle_verified/validation.json',
                 initial_fairness='initial_fairness/validation.json',
                 finite_capacity='long_baselines_verified/validation.json')
    results = {name: json.loads((out / path).read_text()) for name, path in files.items()}
    assert all(r['passed'] and r['source_sha256'] == source for r in results.values())
    assert '246 passed' in (out / 'regression_optimized_local.log').read_text()
    assert '247 passed' in (out / 'regression_optimized_tests.log').read_text()
    archive_path = out / f'executed_sources_{source[:12]}.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for folder in ('core', 'phy', 'ran', 'ric', 'state', 'models', 'traffic',
                       'experiments', 'configs', 'scripts', 'tests', 'docs'):
            for p in (ROOT / folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.md'):
                    archive.add(p, arcname=str(p.relative_to(ROOT)))
        for p in [ROOT / 'README.md', *ROOT.glob('requirements*.txt')]:
            if p.is_file(): archive.add(p, arcname=p.name)
    report = dict(passed=True, source_sha256=source, validation_files=files,
                  regression=dict(local_passed=246, local_skipped=1, spark_passed=247, subtests_passed=2),
                  clocks_unchanged=dict(slot_ms=.5, xapp_slots=5, rapp_slots=10), results=results,
                  limitations=['Finite stationary capacity check, not asymptotic stability or moving generalization',
                               'Local BLER bin domain; unknown profiles require cold NR calibration',
                               'Statistical feedback is not actual CRC feedback or waveform soft combining',
                               'RV0 Chase implemented; IR and strict URLLC tail validation remain incomplete',
                               'Ideal CSI / Gaussian post-LMMSE resources; real A1/E2 and full OFDM deferred'],
                  preserved_partial_runs=['long_baselines', 'long_baselines_final'],
                  historical_versions='Older link/control reports and source archives are retained, not merged into final acceptance',
                  regression_collection_diagnostic='An unrestricted pytest scan collected archived candidate tests; corrected final command is pytest -q tests')
    (out / 'validation_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=True, source_sha256=source,
                          cases=results['finite_capacity']['cases']), indent=2))


if __name__ == '__main__':
    main()
