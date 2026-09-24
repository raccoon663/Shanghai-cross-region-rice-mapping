"""Validate saved GeoAI metrics, inventories, frozen inputs and curve endpoints."""
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.common import sha256
from src.experiments.run_geoai_rqs import classification
from src.experiments.run_eofm_benchmark import write_json


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-private-predictions',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/geoai_rqs.yaml').read_text())
    out=ROOT/cfg['result_dir']
    private=ROOT/cfg['output_dir']/'private'
    published=json.loads((out/'artifact_manifest.json').read_text())
    for name,digest in published['files'].items():
        assert sha256(ROOT/name)==digest,name
    legacy=json.loads((ROOT/'results/eofm_benchmark_v1/artifact_manifest.json').read_text())
    for name,digest in legacy['files'].items():
        assert sha256(ROOT/name)==digest,name
    manifest=pd.read_csv(ROOT/'data_metadata/alphaearth_sample_manifest.csv')
    target=np.flatnonzero(manifest.split.eq('target_val'))
    y=manifest.loc[target,'class_id'].to_numpy()
    first=pd.read_csv(out/'rq1_runs.csv')
    second=pd.read_csv(out/'rq2_runs.csv')
    third=pd.read_csv(out/'rq3_scores.csv')
    curve_path=out/'rq3_risk_coverage.csv'
    curve=pd.read_csv(curve_path if curve_path.exists() else out/'rq3_risk_coverage.csv.gz')
    assert len(first)==2504
    assert len(second)==100
    assert len(third)==44
    assert len(curve)==44*len(target)
    if args.with_private_predictions:
        for record in second.itertuples():
            key=f'adapt_{record.representation}_{record.method}_{record.seed}'
            artifact=np.load(private/f'{key}.npz')
            meta=json.loads((private/f'{key}.json').read_text())
            assert sha256(private/f'{key}.npz')==meta['sha256']
            np.testing.assert_array_equal(artifact['rows'],target)
            computed=classification(y,artifact['probabilities'],cfg['threshold'])
            for metric in ['precision','recall','f1','auroc','auprc','error_rate']:
                np.testing.assert_allclose(computed[metric],getattr(record,metric),atol=1e-12)
    for record in third.itertuples():
        group=curve[(curve.representation==record.representation)&(curve.score==record.score)]
        np.testing.assert_allclose(group.risk.mean(),record.aurc,atol=1e-12)
        np.testing.assert_allclose(group.iloc[-1].risk,record.error_prevalence,atol=1e-12)
        np.testing.assert_allclose(group.coverage,np.arange(1,len(target)+1)/len(target),atol=1e-12)
    record=dict(status='passed',legacy_artifacts_unchanged=len(legacy['files']),
        rq1_rows=len(first),rq2_rows=len(second),
        private_predictions_rechecked=bool(args.with_private_predictions),
        rq3_score_combinations=len(third),risk_curve_rows=len(curve),
        risk_curve_endpoints_and_aurc_verified=True)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
