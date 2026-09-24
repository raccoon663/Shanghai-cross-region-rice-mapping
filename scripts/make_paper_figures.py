"""Rebuild GeoAI tables, research results and all figures from saved CSVs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

NAMES = {'temporal_reconstructed':'Temporal (rebuilt)', 'alphaearth':'AlphaEarth',
         'presto_primary':'Presto (monthly)', 'galileo':'Galileo'}
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7']
SCORES = {'euclidean_nn':'Euclidean 1-NN','cosine_nn':'Cosine 1-NN','cosine_knn':'Cosine 10-NN',
          'mahalanobis':'Mahalanobis','centroid':'Class centroid','entropy':'Entropy',
          'confidence':'1 - confidence','margin':'1 - margin','vote_uncertainty':'Vote uncertainty',
          'tree_variance':'Tree variance','domain_probability':'Domain probability'}


def markdown(frame):
    def fmt(value):
        if pd.isna(value):
            return 'NA'
        if isinstance(value,(float,np.floating)):
            return f'{value:.4f}'
        return str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])+'\n'


def save_table(frame,name,out,description):
    frame.to_csv(out/f'{name}.csv',index=False)
    (out/f'{name}.md').write_text(description+'\n\n'+markdown(frame),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/geoai_rqs.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out, figures = ROOT/cfg['result_dir'],ROOT/cfg['figure_dir']
    figures.mkdir(parents=True,exist_ok=True)
    rq1,rq2,ood = [pd.read_csv(out/name) for name in ['rq1_runs.csv','rq2_runs.csv','rq3_scores.csv']]
    curve_path = out/'rq3_risk_coverage.csv'
    curves = pd.read_csv(curve_path if curve_path.exists() else out/'rq3_risk_coverage.csv.gz')
    deciles = pd.read_csv(out/'rq3_quantiles.csv')
    all_target = pd.concat([rq2,rq1.query("split == 'target_val' and method != 'source_only'").assign(trees=cfg['fewshot_trees'])],ignore_index=True)
    keys = ['representation','method','regime','budget','trees']
    measures = ['precision','recall','f1','auroc','auprc']
    summary = all_target.groupby(keys).agg(**{f'{m}_{stat}':(m,stat) for m in measures for stat in ['mean','std']},
        n_seeds=('seed','nunique')).reset_index()
    baseline = summary.query("method == 'source_only'").set_index('representation').f1_mean
    reference = summary.query("method == 'target_full'").set_index('representation').f1_mean
    summary['f1_gain_over_source_mean'] = summary.f1_mean-summary.representation.map(baseline)
    summary['remaining_gap_to_target_full_mean'] = summary.representation.map(reference)-summary.f1_mean
    save_table(summary,'table2_adaptation',out,
        '# Table 2: Shanghai weak-reference adaptation\n\nMeans and sample SD over seeds; seed SD is not a spatial confidence interval. '
        'Gains/gaps are descriptive differences of means. Source/unsupervised/full-pool RFs: 500 trees, 5 seeds. '
        'Few-shot RFs: 300 trees, 30 seeds except 250 labels (5). Budget 0 unsupervised methods see pool features. '
        'Full-pool reference uses 9,249 weak labels and is not a guaranteed upper bound.')
    table1=[]
    for rep in NAMES:
        a = rq1.query("representation == @rep and method == 'source_only'")
        source = a.query("split == 'source_test'").f1
        target = a.query("split == 'target_val'").f1
        row=dict(representation=rep,source_f1_mean=source.mean(),source_f1_std=source.std(),
                 target_zero_shot_f1_mean=target.mean(),target_zero_shot_f1_std=target.std(),
                 transfer_drop=source.mean()-target.mean(),source_seeds=3)
        for budget in [50,100,250,500]:
            f = rq1.query("representation == @rep and method == 'target_only' and regime == 'distributed' and budget == @budget").f1
            row.update({f'f1_{budget}_mean':f.mean(),f'f1_{budget}_std':f.std(),f'n_seeds_{budget}':len(f)})
        table1.append(row)
    table1=pd.DataFrame(table1)
    save_table(table1,'table1_representation',out,
        '# Table 1: matched representation transfer\n\nSource/zero-shot reuse 3 frozen RF seeds. '
        'Few-shot columns are distributed target-only weak-label RFs, with shared draws and fixed test rows. '
        'Source and few-shot tree counts differ as in the original benchmark; compare representations within each condition.')
    save_table(ood,'table3_ood_failure',out,
        '# Table 3: domain discrimination versus target failure prediction\n\nAll scores use the same seed-42 source RF within each representation. '
        'Domain AUROC is scored on all held-out source/target rows; target-domain prevalence is reported because domain AUPRC depends on it. '
        'Failure means disagreement with the weak reference. Sample Spearman is distinct from block-mean Spearman. '
        'Error AUROC intervals use 500 spatial-block bootstrap draws; no model retraining. '
        'Coverage operating points use label-independent random tie breaking in expectation; no score direction is reversed after evaluation.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':300,'svg.fonttype':'none'})
    def save(fig,name):
        for ext in ['png','svg']:
            fig.savefig(figures/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,5))
    ax.set(xlim=(0,12),ylim=(0,5)); ax.axis('off')
    boxes=[(.2,3.05,'Jiangxi source\n850 training / 314 test'),(4.2,3.05,'Representation systems\nTemporal · AE · Presto · Galileo'),
           (8.2,3.05,'Frozen source RF\nSource / Shanghai evaluation'),
           (.2,.65,'Shanghai adaptation pool\n9,249 weak labels / features'),(4.2,.65,'Adaptation → Shanghai evaluation\nFew-shot · CORAL · weights · pseudo'),
           (8.2,.65,'Source-only risk & abstention\nDomain ≠ failure · risk–coverage')]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),3.6,1.1,boxstyle='round,pad=.08',facecolor='#EAF2F8',edgecolor='#35658B'))
        ax.text(x+1.8,y+.55,label,ha='center',va='center',fontsize=10)
    for start,end in [((3.85,3.6),(4.1,3.6)),((7.85,3.6),(8.1,3.6)),((3.85,1.2),(4.1,1.2)),((10,2.95),(10,1.9))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.5,color='#35658B'))
    ax.text(6,4.7,'Geographic transfer, adaptation and reliability',ha='center',fontsize=16,weight='bold')
    ax.text(6,2.3,'Same 2,751 held-out Shanghai points for every representation and adaptation method',ha='center',fontsize=10)
    ax.text(6,.1,'Spatial blocks preserved · training-only preprocessing · evaluation labels used only for metrics',ha='center',fontsize=10)
    save(fig,'figure1_framework')

    fig,ax=plt.subplots(figsize=(9,4.8))
    xx=np.arange(4)
    for i,(mean,std,label,color) in enumerate([
        ('source_f1_mean','source_f1_std','Jiangxi source test','#455A64'),
        ('target_zero_shot_f1_mean','target_zero_shot_f1_std','Shanghai zero-shot','#0072B2'),
        ('f1_500_mean','f1_500_std','Shanghai: 500 weak labels (target-only)','#D55E00')]):
        ax.bar(xx+(i-1)*.25,table1[mean],.24,yerr=table1[std],label=label,color=color,capsize=3)
    ax.set_xticks(xx,[NAMES[x] for x in table1.representation]); ax.set_ylim(0,1.08)
    ax.set_ylabel('F1 (Shanghai: weak-reference agreement)'); ax.set_title('RQ1 | Representation transfer')
    ax.legend(loc='lower left',fontsize=9); save(fig,'figure2_representation_transfer')

    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True)
    for ax,(method,regime) in zip(axes.flat,[(m,r) for m in ['target_only','joint'] for r in ['distributed','clustered']]):
        for color,rep in zip(COLORS,NAMES):
            g=summary.query('representation == @rep and method == @method and regime == @regime').sort_values('budget')
            ax.plot(g.budget,g.f1_mean,'o-',color=color,label=NAMES[rep],ms=3)
            ax.fill_between(g.budget,g.f1_mean-g.f1_std,g.f1_mean+g.f1_std,color=color,alpha=.12)
            zero=table1.set_index('representation').loc[rep,'target_zero_shot_f1_mean']
            ax.scatter([0],[zero],color=color,marker='x',s=30)
        ax.set_title(f'{method.replace("_"," ").title()} | {regime}')
        ax.set_xlabel('Shanghai weak-label budget'); ax.set_ylabel('Target weak-reference F1')
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('RQ1/RQ2 | Few-shot learning (bands: seed SD; ×: source-only)',y=1.01)
    fig.tight_layout(); save(fig,'figure3_fewshot')

    fig,axes=plt.subplots(1,2,figsize=(13,5.7),sharey=True)
    for ax,metric,title in zip(axes,['domain_auroc','error_auroc'],['Source vs target','Incorrect vs correct target prediction']):
        data=ood.pivot(index='score',columns='representation',values=metric).reindex(index=list(SCORES),columns=list(NAMES))
        im=ax.imshow(data,vmin=0,vmax=1,cmap='cividis',aspect='auto')
        ax.set_xticks(range(4),list(NAMES.values()),rotation=25,ha='right')
        ax.set_yticks(range(len(SCORES)),list(SCORES.values())); ax.grid(False); ax.set_title(title)
        for j in range(len(SCORES)):
            for k in range(4):
                value=data.iloc[j,k]
                ax.text(k,j,f'{value:.3f}',ha='center',va='center',fontsize=8,color='white' if value<.55 else 'black')
    fig.colorbar(im,ax=axes,shrink=.7,label='AUROC (chance = 0.5)')
    fig.suptitle('RQ3 | Geographic separability does not establish failure predictiveness')
    save(fig,'figure4_domain_vs_failure')

    selected=['cosine_knn','entropy','mahalanobis','vote_uncertainty','domain_probability']
    score_colors=['#0072B2','#D55E00','#009E73','#CC79A7','#666666']
    for kind,name in [('quantile','figure5_ood_error'),('coverage','figure6_risk_coverage')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7.8),sharey=True)
        for ax,rep in zip(axes.flat,NAMES):
            for score,color in zip(selected,score_colors):
                if kind=='quantile':
                    g=deciles.query('representation == @rep and score == @score')
                    ax.plot(g['quantile'],g.error_rate,'o-',ms=3,color=color,label=SCORES[score])
                    ax.set_xlabel('Score quantile (low → high; ties kept together)')
                else:
                    g=curves.query('representation == @rep and score == @score')
                    ax.plot(g.coverage,g.risk,color=color,label=SCORES[score],lw=1.6)
                    ax.set_xlabel('Coverage: accepted / all target predictions'); ax.set_xlim(0,1)
            full=ood.query('representation == @rep').error_prevalence.iloc[0]
            ax.axhline(full,color='black',ls=':',lw=1,label='Random acceptance reference')
            ax.set_title(NAMES[rep]); ax.set_ylabel('Risk: weak-reference disagreement')
        axes[0,0].legend(fontsize=7,loc='upper left')
        fig.suptitle('RQ3 | '+('Error across risk-score quantiles' if kind=='quantile' else 'Selective prediction without evaluation labels in scores'),y=1.01)
        fig.tight_layout(); save(fig,name)

    artifacts=[*out.glob('*.csv'), *out.glob('*.csv.gz'), *out.glob('*.md'),
               *[p for p in out.glob('*.json') if p.name!='artifact_manifest.json'],
               *figures.glob('*'), args.config, Path(__file__),
               ROOT/'src/experiments/run_geoai_rqs.py', ROOT/'src/experiments/geoai_methods.py',
               ROOT/'scripts/validate_geoai_results.py', ROOT/'scripts/check_geoai_weight_degeneracy.py']
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    (out/'artifact_manifest.json').write_text(json.dumps({'experiment':cfg['experiment'],'files':manifest},indent=2)+'\n',encoding='utf-8')
    print(f'Rebuilt three tables and six figures in {cfg["experiment"]}')


if __name__=='__main__':
    main()
