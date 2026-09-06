"""Build a local review report from the complete frozen benchmark tables."""
from pathlib import Path
import json
import hashlib
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/eofm_benchmark_v1'
LABELS = {'temporal_reconstructed':'Temporal-92D (rebuilt)', 'alphaearth':'AlphaEarth',
          'presto_primary':'Presto (monthly)', 'galileo':'Galileo'}
COLORS = dict(zip(LABELS, ['#B76A36','#246A9C','#528B5E','#8A62A2']))


def table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.3f}'
        return str(value)
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        experiment_source_sha256=hashlib.sha256((ROOT/'src/experiments/run_eofm_benchmark.py').read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((ROOT/'configs/eofm_benchmark.yaml').read_bytes()).hexdigest())
    (OUT/'execution_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    runs = pd.read_csv(OUT/'source_zero_shot_runs.csv')
    few = pd.read_csv(OUT/'fewshot_summary.csv')
    raw_few = pd.read_csv(OUT/'fewshot_runs.csv')
    paired = pd.read_csv(OUT/'paired_f1_intervals.csv')
    ood = pd.read_csv(OUT/'ood_summary.csv')
    risk = pd.read_csv(OUT/'risk_coverage.csv')
    if len(raw_few) != 2400 or few.n_seeds.ne(30).any() or len(paired) != 126:
        raise ValueError('Do not report an incomplete experiment')
    means = runs.groupby(['representation','split']).f1.agg(['mean','std'])
    comparison = means['mean'].unstack().reindex(LABELS)
    comparison['source_to_target_drop'] = comparison.source_test - comparison.target_val
    comparison.to_csv(OUT/'source_zero_shot_summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    fig, ax = plt.subplots(figsize=(9,5.1),layout='constrained')
    for i, rep in enumerate(LABELS):
        vals = [comparison.loc[rep,'source_test'],comparison.loc[rep,'target_val']]
        ax.plot([0,1],vals,'o-',lw=2,ms=7,color=COLORS[rep],label=LABELS[rep])
        label_y = {'temporal_reconstructed': .803, 'alphaearth': .844,
                   'presto_primary': .831, 'galileo': .817}[rep]
        ax.annotate(f'{vals[1]:.3f}', (1, vals[1]), xytext=(1.07, label_y),
                    color=COLORS[rep], fontsize=9, va='center',
                    arrowprops=dict(arrowstyle='-', color=COLORS[rep], lw=.7))
    ax.set(xticks=[0,1],xticklabels=['Jiangxi source test','Shanghai weak reference'],
           ylabel='F1 (mean over 3 fixed seeds)',ylim=(.78,1.0),xlim=(-.1,1.25),
           title='Cross-region transfer under a matched RF protocol')
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper right',frameon=False)
    fig.text(.01,-.045,'Shanghai scores measure product agreement, not independent field accuracy. Temporal is a reconstructed data version.',fontsize=8,color='#555555')
    save(fig,'transfer_comparison.png')

    fig, axs = plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True,layout='constrained')
    for row, regime in enumerate(['distributed','clustered']):
        for col, method in enumerate(['target_only','joint']):
            ax=axs[row,col]
            for rep in LABELS:
                part=few[(few.regime==regime)&(few.method==method)&(few.representation==rep)].sort_values('budget')
                ax.plot(part.budget,part.f1_mean,'o-',ms=4,color=COLORS[rep],label=LABELS[rep])
                ax.fill_between(part.budget,(part.f1_mean-part.f1_sd).clip(0,1),(part.f1_mean+part.f1_sd).clip(0,1),color=COLORS[rep],alpha=.12)
            ax.set_title(f'{regime.title()} labels · {method.replace("_"," ")}')
            ax.grid(alpha=.18)
            ax.set_xticks([20,100,200,500])
            if row==1: ax.set_xlabel('Target weak-label budget')
            if col==0: ax.set_ylabel('Shanghai weak-reference F1')
    axs[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Few-shot transfer: 30 shared draws per setting',fontsize=15)
    fig.text(.01,-.025,'Bands show ±1 seed standard deviation; they are not population confidence intervals.',fontsize=9,color='#555555')
    save(fig,'fewshot_comparison.png')

    scores=['nn_standardized','nn_cosine','knn10_cosine','mahalanobis','domain_probability']
    values=ood.pivot(index='representation',columns='score',values='error_auroc').reindex(index=LABELS,columns=scores)
    fig,ax=plt.subplots(figsize=(10,4.1),layout='constrained')
    mat=ax.imshow(values, vmin=.3,vmax=.85,cmap='RdYlBu')
    ax.set(xticks=np.arange(5),xticklabels=['1-NN\nstandardized','1-NN\ncosine','10-NN\ncosine','Mahalanobis','Domain\nprobability'],
           yticks=np.arange(4),yticklabels=list(LABELS.values()),title='Can source-distance scores rank target errors?')
    for i in range(4):
        for j in range(5): ax.text(j,i,f'{values.iloc[i,j]:.3f}',ha='center',va='center',fontsize=12)
    fig.colorbar(mat,ax=ax,label='Error AUROC (0.5 = chance)',shrink=.9)
    fig.text(.01,-.04,'Each row uses that classifier’s own errors; AUROC is not a cross-model accuracy score.',fontsize=9,color='#555555')
    save(fig,'ood_comparison.png')

    risk_summary=risk.groupby(['representation','score','target_coverage']).agg(
        actual_coverage=('actual_coverage','mean'),error_rate=('error_rate','mean'),
        held_full_error=('held_block_full_error','mean')).reset_index()
    risk_summary.to_csv(OUT/'risk_coverage_summary.csv',index=False)
    display=comparison[['source_test','target_val','source_to_target_drop']].reset_index().rename(
        columns={'representation':'Representation','source_test':'Source test F1','target_val':'Shanghai F1','source_to_target_drop':'Transfer drop'})
    display['Representation']=display['Representation'].map(LABELS)
    adaptation=few[few.budget.eq(500)][['regime','method','representation','f1_mean','f1_sd']].copy()
    adaptation['representation']=adaptation.representation.map(LABELS)
    pair_display=paired[paired.budget.eq(0)][['representation_a','representation_b','delta_f1','ci95_low','ci95_high']].copy()
    for column in ['representation_a','representation_b']: pair_display[column]=pair_display[column].map(LABELS)
    knn=ood[ood.score.eq('knn10_cosine')][['representation','error_auroc','error_auroc_ci_low','error_auroc_ci_high','error_auprc','error_prevalence']].copy()
    knn['representation']=knn.representation.map(LABELS)
    report=f'''# Four-representation rice transfer experiment — local review

The full first benchmark is complete: 12 source RF fits, 2,400 adaptation fits,
300 shared balanced draws, five OOD scores per representation, and 126 paired
spatial-block comparisons. All work in this stage remains local.

## Source and zero-shot transfer

{table(display)}

![Source and zero-shot comparison](transfer_comparison.png)

The historical portfolio numbers remain frozen. The temporal row above uses the
new reconstructed 92D data, not the missing historical source table. Shanghai
numbers are agreement with a product-derived weak reference, not field accuracy.

## Zero-shot paired uncertainty

Differences below are B minus A. Intervals resample spatial blocks while retaining
the same three source seeds. These are descriptive, unadjusted intervals.

{table(pair_display)}

## Few-shot adaptation

![Few-shot comparison](fewshot_comparison.png)

At budget 500 (30 shared seeds per row):

{table(adaptation)}

Distributed and clustered labels use the existing frozen samplers. Target-only
and joint methods use the same 300-tree RF budget across representations.
Seed standard deviations measure draw/training variability; paired spatial-block
intervals for every budget and method are in `paired_f1_intervals.csv`.

## OOD risk

![OOD comparison](ood_comparison.png)

The prespecified 10-NN cosine score:

{table(knn)}

Scores are not flipped or selected after examining errors. Negative associations
are retained. Error AUROC uses each classifier's own error outcome and does not
measure its classification accuracy. `domain_audit.csv` includes permuted-label
controls; `risk_coverage_summary.csv` reports held-block coverage and disagreement
for confidence and 10-NN cosine thresholds chosen on separate tuning blocks.

## Protocol and limitations

- Exact shared population: 1,429 Jiangxi and 12,000 Shanghai samples; zero exclusions.
- Training/validation/test blocks remain disjoint. Only source-training data fits
  source RFs, imputation, scaling and distance reference distributions.
- All 12 source prediction artifacts were frozen before Shanghai metrics.
- No target-performance tuning of representations, RF settings or thresholds.
- Representation systems have different upstream inputs and resolutions; this is
  not an architecture-only comparison. Presto uses coordinates only in its frozen
  encoder; downstream features never include them.
- The same Shanghai reference has appeared in earlier project analysis, so it is
  not a newly untouched holdout. These results support a controlled extension,
  not independent external validation.
- Main README, legacy results, parcel products and GitHub were not changed by
  this local experiment stage.

Reproducibility: `protocol_freeze.json`, `source_prediction_freeze.json`,
`shared_draw_summary.csv`, the experiment config and runner. Private per-sample
features, predictions, draws and model files remain in ignored runtime storage.
'''
    (OUT/'benchmark_report.md').write_text(report,encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}
    hashes.update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.png')})
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete_local_review',source_fits=12,
        adaptation_fits=2400,shared_draws=300,paired_comparisons=126,publication='not_pushed',files=hashes),indent=2)+'\n')
    print('Complete local review report:',OUT/'benchmark_report.md')


if __name__=='__main__':
    main()
