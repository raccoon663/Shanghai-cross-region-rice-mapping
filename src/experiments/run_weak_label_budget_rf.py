from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score

from src.data.common import feature_groups, load_config, resolve_data_path, resolve_project_path
from src.data.build_target_weak_splits import build


def choose_budget(pool: pd.DataFrame, budget: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed); chosen=[]
    for label in (0,1):
        need=budget//2
        part=pool[pool.class_id==label].sample(frac=1,random_state=seed+label).copy()
        diverse=part.drop_duplicates("spatial_block").index.to_numpy()
        if len(diverse)>=need: picked=rng.choice(diverse,need,replace=False)
        else:
            rest=np.setdiff1d(part.index.to_numpy(),diverse)
            picked=np.r_[diverse,rng.choice(rest,need-len(diverse),replace=False)]
        chosen.extend(picked)
    return np.asarray(chosen)


def metrics(y, probability):
    pred=probability>=.5
    return {"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"iou":jaccard_score(y,pred,zero_division=0),"balanced_accuracy":balanced_accuracy_score(y,pred),"auroc":roc_auc_score(y,probability),"auprc":average_precision_score(y,probability)}


def run(config_file):
    config_path=Path(config_file).resolve(); config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config,data_path=load_config(config["data_config"]); output=resolve_project_path(config["output_dir"],config_path); output.mkdir(parents=True,exist_ok=True)
    target_path=output/"target_weak_samples.csv"
    if not target_path.exists(): build(config_file)
    target=pd.read_csv(target_path); source=pd.read_csv(resolve_data_path(data_config["data"]["source_features"],data_path))
    source_split=pd.read_csv(resolve_project_path(data_config["project"]["output_dir"],data_path)/"source_splits.csv")
    source=source.merge(source_split[["sample_id","split"]],on="sample_id",validate="one_to_one")
    features=feature_groups(list(source.columns))[config["feature_set"]]
    source=source[source.split=="source_train"].reset_index(drop=True); pool=target[target.split=="target_pool"]; val=target[target.split=="target_val"]
    xs=source[features].to_numpy("float32"); ys=source.class_id.to_numpy("uint8"); xv=val[features].to_numpy("float32"); yv=val.class_id.to_numpy("uint8")
    medians=np.nanmedian(np.where(xs > -9990, xs, np.nan),axis=0).astype("float32")
    xs=np.where(np.isfinite(xs)&(xs>-9990),xs,medians); xv=np.where(np.isfinite(xv)&(xv>-9990),xv,medians)
    rows=[]; rf=config["random_forest"]
    for seed in config["seeds"]:
      for budget in config["budgets"]:
        selected=choose_budget(pool,budget,seed) if budget else np.array([],dtype=int)
        xt=pool.loc[selected,features].to_numpy("float32") if budget else np.empty((0,len(features)),dtype="float32"); yt=pool.loc[selected,"class_id"].to_numpy("uint8") if budget else np.empty(0,dtype="uint8")
        if budget: xt=np.where(np.isfinite(xt)&(xt>-9990),xt,medians)
        for method in config["methods"]:
          if method=="target_only" and budget==0: continue
          if method=="source_only": xtrain,ytrain,weight=xs,ys,None
          elif method=="target_only": xtrain,ytrain,weight=xt,yt,None
          else:
            xtrain,ytrain=np.vstack([xs,xt]),np.r_[ys,yt]
            weight=np.r_[np.ones(len(ys)),np.full(len(yt),len(ys)/max(1,len(yt)))] if budget else np.ones(len(ys))
          model=RandomForestClassifier(n_estimators=rf["n_estimators"],min_samples_leaf=rf["min_samples_leaf"],max_features=rf["max_features"],n_jobs=rf["n_jobs"],class_weight="balanced_subsample" if method!="joint" else None,random_state=seed)
          model.fit(xtrain,ytrain,sample_weight=weight); probability=model.predict_proba(xv)[:,1]
          rows.append({"method":method,"budget_total":budget,"seed":seed,"n_train":len(ytrain),"evaluation":"official_product_weak_label_target_val",**metrics(yv,probability)})
    runs=pd.DataFrame(rows); runs.to_csv(output/"rf_budget_runs.csv",index=False)
    summary=runs.groupby(["method","budget_total"]).agg(f1_mean=("f1","mean"),f1_std=("f1","std"),auprc_mean=("auprc","mean"),auprc_std=("auprc","std"),precision_mean=("precision","mean"),recall_mean=("recall","mean")).reset_index(); summary.to_csv(output/"rf_budget_summary.csv",index=False); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/weak_label_budget.yaml"); a=p.parse_args(); print(run(a.config).to_string(index=False))


if __name__=="__main__": main()
