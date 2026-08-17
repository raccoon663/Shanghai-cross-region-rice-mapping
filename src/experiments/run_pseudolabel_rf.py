from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score
from src.data.common import load_config,resolve_data_path,resolve_project_path,feature_groups


def run(config_file):
 cpath=Path(config_file).resolve(); c=yaml.safe_load(cpath.read_text(encoding="utf-8")); dc,dp=load_config(c["data_config"]); wcpath=Path(c["weak_label_config"]).resolve(); wc=yaml.safe_load(wcpath.read_text(encoding="utf-8")); weak_dir=resolve_project_path(wc["output_dir"],wcpath)
 source=pd.read_csv(resolve_data_path(dc["data"]["source_features"],dp)); splits=pd.read_csv(resolve_project_path(dc["project"]["output_dir"],dp)/"source_splits.csv"); source=source.merge(splits[["sample_id","split"]],on="sample_id"); source=source[source.split=="source_train"]
 target=pd.read_csv(weak_dir/"target_weak_samples.csv"); pool=target[target.split=="target_pool"].reset_index(drop=True); val=target[target.split=="target_val"].reset_index(drop=True); features=feature_groups(list(source.columns))["s1_s2_fusion"]
 xs=source[features].to_numpy("float32"); ys=source.class_id.to_numpy("uint8"); xp=pool[features].to_numpy("float32"); xv=val[features].to_numpy("float32"); yv=val.class_id.to_numpy("uint8"); med=np.nanmedian(np.where(xs>-9990,xs,np.nan),axis=0); xs=np.where(np.isfinite(xs)&(xs>-9990),xs,med); xp=np.where(np.isfinite(xp)&(xp>-9990),xp,med); xv=np.where(np.isfinite(xv)&(xv>-9990),xv,med)
 rows=[]
 for seed in c["seeds"]:
  model=RandomForestClassifier(n_estimators=c["n_estimators"],min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=seed); model.fit(xs,ys)
  for round_id in range(c["rounds"]+1):
   pv=model.predict_proba(xv)[:,1]; pred=pv>=.5; rows.append({"seed":seed,"round":round_id,"pseudo_n":0 if round_id==0 else len(chosen),"precision":precision_score(yv,pred),"recall":recall_score(yv,pred),"f1":f1_score(yv,pred),"auprc":average_precision_score(yv,pv),"evaluation":"official_product_weak_label_target_val"})
   if round_id==c["rounds"]: break
   pp=model.predict_proba(xp)[:,1]; neg=np.where(pp<=1-c["confidence_threshold"])[0]; pos=np.where(pp>=c["confidence_threshold"])[0]; k=min(c["top_k_per_class"],len(neg),len(pos)); neg=neg[np.argsort(pp[neg])[:k]]; pos=pos[np.argsort(pp[pos])[-k:]]; chosen=np.r_[neg,pos]; pseudo=np.r_[np.zeros(k,dtype="uint8"),np.ones(k,dtype="uint8")]
   weights=np.r_[np.ones(len(ys)),np.full(len(chosen),c["pseudo_weight"])]; model=RandomForestClassifier(n_estimators=c["n_estimators"],min_samples_leaf=2,max_features="sqrt",n_jobs=-1,random_state=seed+round_id+1); model.fit(np.vstack([xs,xp[chosen]]),np.r_[ys,pseudo],sample_weight=weights)
 out=resolve_project_path(c["output_dir"],cpath); out.mkdir(parents=True,exist_ok=True); runs=pd.DataFrame(rows); runs.to_csv(out/"pseudolabel_runs.csv",index=False); summary=runs.groupby("round").agg(pseudo_n=("pseudo_n","mean"),f1_mean=("f1","mean"),f1_std=("f1","std"),auprc_mean=("auprc","mean"),auprc_std=("auprc","std"),precision_mean=("precision","mean"),recall_mean=("recall","mean")).reset_index(); summary.to_csv(out/"pseudolabel_summary.csv",index=False); return summary


def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pseudolabel_rf.yaml"); a=p.parse_args(); print(run(a.config).to_string(index=False))


if __name__=="__main__": main()
