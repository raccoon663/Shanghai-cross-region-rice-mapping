from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    root=Path("outputs/experiments/weak_label_budget_m4_v1")
    data=pd.read_csv(root/"rf_budget_summary.csv")
    colors={"source_only":"#777777","target_only":"#3182bd","joint":"#238b45"}
    labels={"source_only":"Source only","target_only":"Target only","joint":"Source + target"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharex=True)
    for method,group in data.groupby("method"):
        group=group.sort_values("budget_total")
        for ax,metric,std,title in [(axes[0],"f1_mean","f1_std","Weak-label F1"),(axes[1],"auprc_mean","auprc_std","Weak-label AUPRC")]:
            ax.errorbar(group.budget_total,group[metric],yerr=group[std].fillna(0),marker="o",capsize=3,lw=2,color=colors[method],label=labels[method])
            ax.set_title(title,weight="bold"); ax.set_xlabel("Shanghai official-product label budget (total)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Score on spatially held-out weak labels")
    axes[1].legend(frameon=False)
    fig.suptitle("RF Fusion target adaptation under limited weak-label budgets",weight="bold")
    fig.text(.5,-.02,"Official-product agreement only; not independent ground-truth accuracy",ha="center",color="#8c2d04")
    fig.tight_layout(); fig.savefig(root/"rf_label_budget_curves.png",dpi=220,bbox_inches="tight",facecolor="white")


if __name__=="__main__": main()

