import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.experiments.run_eofm_benchmark import fit_medians, impute
import pandas as pd
f=pd.read_csv(root/'data_metadata/alphaearth_sample_manifest.csv')
x=np.load(root/'outputs/eofm_benchmark_v1/private/canonical.npz')['alphaearth']
source=np.flatnonzero(f.split.eq('source_train'))
target=np.flatnonzero(f.split.eq('target_val'))
xs,xt=x[source],x[target]
saved=joblib.load(root/'outputs/eofm_benchmark_v1/private/source_alphaearth_42.joblib')['model']
weighted=np.load(root/'outputs/geoai_rqs_v1/private/adapt_alphaearth_importance_42.npz')['probabilities']
baseline=saved.predict_proba(xt)[:,1]
results={}
with threadpool_limits(limits=1):
    for name,w in [('none',None),('unit',np.ones(len(source))),('normalized_clipping_floor',np.full(len(source),.1)/np.full(len(source),.1).mean())]:
        model=RandomForestClassifier(**saved.get_params()).fit(xs,f.loc[source,'class_id'],sample_weight=w)
        p=model.predict_proba(xt)[:,1]
        results[name]={'constant_weight':float(w[0]) if w is not None else None,'max_probability_delta_to_source':float(abs(p-baseline).max()),
            'max_probability_delta_to_importance':float(abs(p-weighted).max())}
print(json.dumps(results,indent=2))
(root/'results/geoai_rqs_v1/constant_weight_diagnostic.json').write_text(json.dumps(results,indent=2)+'\n')
