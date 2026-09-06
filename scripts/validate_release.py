"""Check curated evidence, frozen metrics and local documentation links offline."""
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT / 'results/eofm_benchmark_v1'
    manifest = json.loads((evidence / 'artifact_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    protocol = json.loads((evidence / 'protocol_freeze.json').read_text())
    config_path = ROOT / 'configs/eofm_benchmark.yaml'
    config = yaml.safe_load(config_path.read_text())
    assert protocol['config'] == config
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == protocol['release_config_sha256']
    sample = ROOT / config['sample_manifest']
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == config['sample_manifest_sha256']
    for name, count in [('fewshot_runs.csv', 2400), ('fewshot_summary.csv', 80),
                        ('paired_f1_intervals.csv', 126), ('ood_summary.csv', 20),
                        ('shared_draw_summary.csv', 300)]:
        frame = pd.read_csv(evidence / name)
        assert len(frame) == count and not frame.duplicated().any(), name
    runs = pd.read_csv(evidence / 'fewshot_runs.csv')
    assert runs.key.nunique() == 2400
    assert runs.groupby(['representation', 'regime', 'method', 'budget']).seed.nunique().eq(30).all()
    source = pd.read_csv(evidence / 'source_zero_shot_runs.csv')
    assert len(source[['representation', 'seed']].drop_duplicates()) == 12
    zero = pd.read_csv(evidence / 'source_zero_shot_summary.csv').set_index('representation')
    few = pd.read_csv(evidence / 'fewshot_summary.csv')
    few = few[(few.budget == 500) & (few.regime == 'distributed') & (few.method == 'target_only')].set_index('representation')
    ood = pd.read_csv(evidence / 'ood_summary.csv')
    ood = ood[ood.score == 'knn10_cosine'].set_index('representation')
    domain = pd.read_csv(evidence / 'domain_audit.csv').set_index('representation')
    expected = {'temporal_reconstructed': (.815, .859, .456, 1.),
                'alphaearth': (.836, .898, .798, 1.),
                'presto_primary': (.826, .884, .438, .999),
                'galileo': (.822, .862, .490, .993)}
    for rep, values in expected.items():
        actual = (zero.loc[rep, 'target_val'], few.loc[rep, 'f1_mean'],
                  ood.loc[rep, 'error_auroc'], domain.loc[rep, 'true_domain_auroc'])
        assert tuple(round(v, 3) for v in actual) == values, rep
    paired = pd.read_csv(evidence / 'paired_f1_intervals.csv')
    paired = paired[paired.regime == 'zero_shot']
    assert len(paired) == 6 and (paired.ci95_low <= 0).all() and (paired.ci95_high >= 0).all()
    for document in list(ROOT.glob('*.md')) + list((ROOT / 'docs').glob('*.md')) + list((ROOT / 'results').rglob('*.md')):
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if ':' not in link and not link.startswith('#'):
                assert (document.parent / link.split('#')[0]).exists(), (document, link)
    print('PASS: 17 curated hashes, config/sample identity, inventory, headline metrics, zero-shot intervals and Markdown links')


if __name__ == '__main__':
    main()
