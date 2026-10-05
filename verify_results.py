"""Verify recorded decisions and statistics without audio or manuscript files."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(results_dir, summary_file=None):
    results_dir = Path(results_dir)
    c = pd.read_csv(results_dir / 'candidate_metrics.csv')
    p = pd.read_csv(results_dir / 'policy_metrics.csv')
    manifest = json.loads((results_dir / 'source_manifest.json').read_text(encoding='utf8'))
    summary_path = Path(summary_file) if summary_file else results_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf8')) if summary_path.exists() else None
    checks = []

    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    n = len(manifest)
    check(n > 0 and c.source_id.nunique() == n, 'manifest and measured source counts')
    check(set(c.source_id) == {r['source_id'] for r in manifest}, 'source identifiers agree')
    check(len(p) == n * 6, 'six declared policy outcomes per source')
    check(not c.duplicated(['source_id', 'config']).any(), 'unique candidate identities')
    check('error' not in c or c.error.isna().all(), 'all candidate measurements valid')
    source_hashes = candidate_hashes = 0
    for source in manifest:
        sid = source['source_id']
        rows = c[c.source_id == sid]
        expected = 27 if source['rate'] == 16000 else 49
        check(len(rows) == expected, sid + ' candidate count')
        for policy, threshold in [('Global-only', None), ('Tail-constrained', 15)]:
            feasible = rows[rows.snr_db >= 20]
            if threshold is not None:
                feasible = feasible[feasible.tail_snr_db >= threshold]
            check(not feasible.empty, sid + ' ' + policy + ' feasible')
            best = feasible.sort_values(['bytes', 'tail_snr_db', 'config'], ascending=[True, False, True]).iloc[0]
            selected = p[(p.source_id == sid) & (p.policy == policy)]
            check(len(selected) == 1, sid + ' ' + policy + ' single decision')
            chosen = selected.iloc[0]
            check(best.config == chosen.config and best.bytes == chosen.bytes, sid + ' ' + policy + ' finite minimum')
        relative = Path(source['reference_path'].replace(chr(92), '/'))
        reference = (ROOT / 'data' if source['group'].startswith('supplied') else ROOT) / relative
        if reference.is_file():
            check(digest(reference) == source['sha256'], sid + ' prepared reference hash')
            source_hashes += 1
        for item in rows.itertuples():
            bitstream = ROOT / 'data' / 'encoded' / sid / (item.config + item.extension)
            if bitstream.is_file():
                check(bitstream.stat().st_size == item.bytes and digest(bitstream) == item.encoded_sha256,
                      sid + ' ' + item.config + ' bitstream hash and bytes')
                candidate_hashes += 1

    for group in ['public_speech', 'public_music']:
        sources = [r for r in manifest if r['group'] == group]
        if not sources:
            continue
        check(len({r['independent_unit'] for r in sources}) == len(sources), group + ' distinct source units')
        chosen = p[(p.group == group) & (p.policy == 'Tail-constrained')]
        check(len(chosen) == len(sources) and chosen.passes_both.all(), group + ' constraint compliance')
        pivot = p[p.group == group].pivot(index='source_id', columns='policy', values='bytes')
        if summary is not None:
            for baseline in ['MP3-128', 'AAC-128', 'Opus-64', 'PCM-16', 'Global-only']:
                values = (100 * (1 - pivot['Tail-constrained'] / pivot[baseline])).to_numpy()
                indices = np.random.default_rng(20261004).integers(0, len(values), (10000, len(values)))
                estimates = values[indices].mean(axis=1)
                actual = [values.mean(), *np.quantile(estimates, [.025, .975])]
                record = summary['public'][group]['savings_vs_' + baseline]
                check(np.allclose(actual, [record['mean'], record['lo'], record['hi']], atol=1e-10),
                      group + ' paired bootstrap versus ' + baseline)

    grid = pd.read_csv(results_dir / 'threshold_sensitivity.csv')
    check(len(grid) == n * 12 and grid.bytes.notna().all(), 'complete feasible threshold grid')
    for (_, _), rows in grid.groupby(['source_id', 'tail_db']):
        check(np.all(np.diff(rows.sort_values('global_db').bytes) >= 0), 'global threshold monotonicity')
    for (_, _), rows in grid.groupby(['source_id', 'global_db']):
        check(np.all(np.diff(rows.sort_values('tail_db').bytes) >= 0), 'tail threshold monotonicity')
    if results_dir.resolve() == (ROOT / 'results' / 'recorded').resolve():
        check(n == 92 and len(c) == 3628 and len(p) == 552, 'published record counts')
        for group, expected in [('public_speech', 36), ('public_music', 22)]:
            global_only = p[(p.group == group) & (p.policy == 'Global-only')]
            check(global_only.passes_global.all() and int((~global_only.passes_tail).sum()) == expected,
                  group + ' published global-only tail violations')
        auxiliary = pd.read_csv(ROOT / 'results' / 'supplementary' / 'synthetic_denoising_validation.csv')
        check(len(auxiliary) == 90 and np.allclose(auxiliary.input_snr_db, auxiliary.measured_input_snr_db, atol=1e-6),
              '90 auxiliary noise conditions with measured input SNR')
    return dict(status='PASS', sources=n, candidate_files=len(c), policy_rows=len(p), check_count=len(checks),
                summary_verified=summary is not None, local_reference_hashes_verified=source_hashes,
                local_candidate_hashes_verified=candidate_hashes, checks=checks,
                scope='Checks recorded decisions and statistics; hashes additionally checked only for locally present audio.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results' / 'recorded')
    parser.add_argument('--summary-file', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'verification.json')
    args = parser.parse_args()
    report = verify(args.results_dir, args.summary_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
