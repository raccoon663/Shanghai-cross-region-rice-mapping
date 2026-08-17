from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; P=ROOT/"results/phase3"; P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":240,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
COL={"alphaearth":"#1768AC","temporal":"#F28E2B","combined":"#2A9D8F","confidence":"#8E6C8A","knn10_cosine":"#D1495B"}

def save(fig,name): fig.tight_layout(); fig.savefig(P/name,bbox_inches="tight"); plt.close(fig)

le=pd.read_csv(P/"label_efficiency_robust.csv")
fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
for ax,(reg,title) in zip(axs,[("distributed","Spatially distributed labels"),("clustered","Spatially clustered labels")]):
  for rep in ["alphaearth","temporal"]:
    d=le[(le.regime==reg)&(le.representation==rep)]
    ax.plot(d.budget_total,d.f1_mean,"o-",label=rep.title(),color=COL[rep]); ax.fill_between(d.budget_total,d.f1_ci_low,d.f1_ci_high,color=COL[rep],alpha=.18)
  ax.set(title=title,xlabel="Target weak-label budget",ylabel="Shanghai reference agreement (F1)"); ax.grid(alpha=.2); ax.legend()
save(fig,"fig_label_efficiency_robust.png")

ood=pd.read_csv(P/"ood_metric_robustness.csv"); d=ood[ood.row_type=="method"].copy(); labels={"nn_standardized":"1NN standardized","nn_cosine":"1NN cosine","knn10_cosine":"10NN cosine","mahalanobis":"Mahalanobis","domain_probability":"Domain probability"}
fig,axs=plt.subplots(1,2,figsize=(11,4.4)); y=np.arange(len(d)); axs[0].barh(y,d.error_auroc,xerr=[d.error_auroc-d.error_auroc_ci_low,d.error_auroc_ci_high-d.error_auroc],color="#457B9D",alpha=.9); axs[0].axvline(.5,color="k",ls="--",lw=1); axs[0].set(yticks=y,yticklabels=[labels[x] for x in d.method_1],xlabel="Error-discrimination AUROC",title="OOD metric robustness",xlim=(.3,.9)); axs[0].invert_yaxis()
axs[1].scatter(d.block_spearman,d.decile_spearman,s=70,color="#D1495B");
for _,r in d.iterrows(): axs[1].annotate(labels[r.method_1],(r.block_spearman,r.decile_spearman),xytext=(4,3),textcoords="offset points",fontsize=8)
axs[1].axhline(0,color="k",lw=.7); axs[1].axvline(0,color="k",lw=.7); axs[1].set(xlabel="Spatial-block Spearman",ylabel="OOD-decile Spearman",title="Geographic and ordinal consistency"); axs[1].grid(alpha=.2)
save(fig,"fig_ood_metric_comparison.png")

r=pd.read_csv(P/"frozen_risk_coverage.csv"); s=r.groupby(["risk_score","target_coverage"]).agg(coverage=("actual_coverage","mean"),error=("error_rate","mean"),err_sd=("error_rate","std")).reset_index()
fig,axs=plt.subplots(1,2,figsize=(10.8,4.3))
for score,g in s.groupby("risk_score"):
  axs[0].plot(g.coverage,g.error,"o-",label=score,color=COL[score]); axs[1].plot(g.target_coverage,g.coverage,"o-",label=score,color=COL[score])
axs[0].axhline(.2261,color="gray",ls="--",label="full-coverage error"); axs[0].set(xlabel="Actual held-block coverage",ylabel="Error rate",title="Frozen selective risk"); axs[1].plot([.48,.92],[.48,.92],"k--",lw=1); axs[1].set(xlabel="Target coverage set on tuning blocks",ylabel="Actual held-block coverage",title="Threshold transfer across spatial blocks")
for ax in axs: ax.grid(alpha=.2); ax.legend(fontsize=8)
save(fig,"fig_frozen_risk_coverage.png")

comp=pd.read_csv(P/"error_complementarity_robust.csv"); c=comp[comp.regime=="distributed"].groupby(["budget_total","error_group"]).fraction.mean().unstack(); order=["both_correct","alphaearth_only_correct","temporal_only_correct","both_wrong"]
fig,ax=plt.subplots(figsize=(8,4.5)); bottom=np.zeros(len(c)); colors=["#2A9D8F","#1768AC","#F28E2B","#D1495B"]
for name,color in zip(order,colors): ax.bar(c.index,c[name],bottom=bottom,width=22 if c.index.min()==20 else 15,label=name.replace("_"," "),color=color); bottom+=c[name].to_numpy()
ax.set(xlabel="Target weak-label budget",ylabel="Fraction of validation samples",title="Repeated error complementarity (30 seeds)",ylim=(0,1)); ax.legend(ncol=2,fontsize=8)
save(fig,"fig_error_complementarity_robust.png")

sb=pd.read_csv(P/"spatial_block_stability.csv"); b=sb[sb.budget_total==500].groupby(["representation","spatial_block"]).agg(error=("error_rate","mean"),ood=("mean_ood","mean"),ece=("ece","mean"),n=("n","mean")).reset_index()
fig,axs=plt.subplots(1,2,figsize=(10.8,4.4))
for rep,g in b.groupby("representation"): axs[0].scatter(g.ood,g.error,s=np.sqrt(g.n)*12,alpha=.7,label=rep.title(),color=COL[rep]); axs[1].scatter(g.ood,g.ece,s=np.sqrt(g.n)*12,alpha=.7,label=rep.title(),color=COL[rep])
axs[0].set(xlabel="Mean 10NN cosine distance",ylabel="Block error rate",title="Error varies across Shanghai blocks"); axs[1].set(xlabel="Mean 10NN cosine distance",ylabel="Block ECE",title="Calibration varies across Shanghai blocks")
for ax in axs: ax.grid(alpha=.2); ax.legend()
save(fig,"fig_spatial_block_stability.png")

ref=pd.read_csv(P/"reference_sensitivity.csv"); z=ref[ref.analysis=="zero_shot"]
a=ref[(ref.analysis=="reference_quality_adaptation")&(ref.regime=="distributed")&(ref.budget_total==500)].groupby("representation").agg(f1=("f1","mean"),sd=("f1","std")).reset_index()
fig,axs=plt.subplots(1,2,figsize=(10.5,4.3)); pivot=z.pivot(index="representation",columns="subset",values="f1").loc[["alphaearth","temporal"]]; x=np.arange(2); axs[0].bar(x-.18,pivot["all"],.36,label="All reference samples",color="#8DA0CB"); axs[0].bar(x+.18,pivot["interior_large_patch"],.36,label="Interior / large patch",color="#66C2A5"); axs[0].set(xticks=x,xticklabels=["AlphaEarth","Temporal"],ylabel="Zero-shot F1",title="Reference-quality sensitivity",ylim=(.65,.88)); axs[0].legend(fontsize=8)
axs[1].bar(a.representation,a.f1,yerr=a.sd,color=[COL[x] for x in a.representation]); axs[1].set(ylabel="F1 on strict subset",title="500-label adaptation remains ordered",ylim=(.75,.97)); axs[1].grid(axis="y",alpha=.2)
save(fig,"fig_reference_sensitivity.png")

workflow=ROOT/"deliverables/sentinel_fusion_rice_final/figures/jiangxi_to_shanghai_fusion_workflow.png"
if workflow.exists(): shutil.copy2(workflow,P/"fig_study_design.png")
print("Phase III figures complete")
