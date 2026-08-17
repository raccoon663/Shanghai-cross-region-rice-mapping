from pathlib import Path
import argparse,json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio,yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,f1_score,log_loss,precision_score,recall_score
from src.data.common import load_config,resolve_data_path,resolve_project_path,feature_groups
from src.experiments.run_weak_label_budget_rf import choose_budget

def ece(y,p,bins=10):
 edges=np.linspace(0,1,bins+1); out=0
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
  if m.any(): out+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(out)

def score(y,p):
 pred=p>=.5
 return {"n":len(y),"precision":precision_score(y,pred),"recall":recall_score(y,pred),"f1":f1_score(y,pred),"auprc":average_precision_score(y,p),"brier":brier_score_loss(y,p),"nll":log_loss(y,np.c_[1-p,p]),"ece":ece(y,p)}

def main_run(config_file):
 cp=Path(config_file).resolve(); c=yaml.safe_load(cp.read_text(encoding="utf-8")); dc,dp=load_config(c["data_config"]); wp=Path(c["weak_label_config"]).resolve(); wc=yaml.safe_load(wp.read_text(encoding="utf-8")); wd=resolve_project_path(wc["output_dir"],wp); out=resolve_project_path(c["output_dir"],cp); out.mkdir(parents=True,exist_ok=True)
 source=pd.read_csv(resolve_data_path(dc["data"]["source_features"],dp)); features=feature_groups(list(source.columns))["s1_s2_fusion"]; target=pd.read_csv(wd/"target_weak_samples.csv"); pool=target[target.split=="target_pool"]; val=target[target.split=="target_val"]
 chosen=choose_budget(pool,c["target_budget"],c["seed"]); train=pool.loc[chosen]; x=train[features].to_numpy("float32"); y=train.class_id.to_numpy("uint8"); med=np.nanmedian(np.where(x>-9990,x,np.nan),axis=0); x=np.where(np.isfinite(x)&(x>-9990),x,med)
 model=RandomForestClassifier(n_estimators=c["n_estimators"],min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=c["seed"]); model.fit(x,y)
 blocks=np.array(sorted(val.spatial_block.unique())); rng=np.random.default_rng(c["seed"]); rng.shuffle(blocks); cal_blocks=set(blocks[:int(len(blocks)*c["calibration_fraction"])]); cal=val[val.spatial_block.isin(cal_blocks)]; ev=val[~val.spatial_block.isin(cal_blocks)]
 def matrix(frame):
  z=frame[features].to_numpy("float32"); return np.where(np.isfinite(z)&(z>-9990),z,med)
 pc=model.predict_proba(matrix(cal))[:,1]; pe=model.predict_proba(matrix(ev))[:,1]; logits=np.log(np.clip(pc,1e-6,1-1e-6)/(1-np.clip(pc,1e-6,1-1e-6))).reshape(-1,1); platt=LogisticRegression(C=1e6).fit(logits,cal.class_id); pe_cal=platt.predict_proba(np.log(np.clip(pe,1e-6,1-1e-6)/(1-np.clip(pe,1e-6,1-1e-6))).reshape(-1,1))[:,1]
 metrics=pd.DataFrame([{"variant":"uncalibrated",**score(ev.class_id.to_numpy(),pe)},{"variant":"platt",**score(ev.class_id.to_numpy(),pe_cal)}]); metrics.to_csv(out/"calibration_metrics.csv",index=False)
 target_raster=resolve_data_path(dc["data"]["target_features"],dp)
 with rasterio.open(target_raster) as src:
  lookup={n:i+1 for i,n in enumerate(src.descriptions)}; indexes=[lookup[f] for f in features]; fp=src.profile.copy(); fp.update(count=1,dtype="float32",nodata=-9999.,compress="deflate"); bp=src.profile.copy(); bp.update(count=1,dtype="uint8",nodata=255,compress="deflate")
  paths={"prob":out/"rice_probability_uncalibrated.tif","cal":out/"rice_probability_calibrated.tif","unc":out/"rice_uncertainty_entropy.tif","bin":out/"rice_binary_calibrated.tif","mask":out/"valid_data_mask.tif"}
  with rasterio.open(paths["prob"],"w",**fp) as d0,rasterio.open(paths["cal"],"w",**fp) as d1,rasterio.open(paths["unc"],"w",**fp) as d2,rasterio.open(paths["bin"],"w",**bp) as d3,rasterio.open(paths["mask"],"w",**bp) as d4:
   for _,w in src.block_windows(1):
    cube=src.read(indexes,window=w,out_dtype="float32"); shape=cube.shape[1:]; z=cube.reshape(len(features),-1).T; obs=np.isfinite(z)&(z>-9990); valid=obs.sum(1)>=23; z=np.where(obs,z,med); p=np.full(len(z),-9999.,"float32"); q=p.copy(); ent=p.copy(); b=np.full(len(z),255,"uint8"); m=np.zeros(len(z),"uint8")
    if valid.any():
     p[valid]=model.predict_proba(z[valid])[:,1]; lg=np.log(np.clip(p[valid],1e-6,1-1e-6)/(1-np.clip(p[valid],1e-6,1-1e-6))).reshape(-1,1); q[valid]=platt.predict_proba(lg)[:,1]; ent[valid]=-(q[valid]*np.log(np.clip(q[valid],1e-6,1))+(1-q[valid])*np.log(np.clip(1-q[valid],1e-6,1)))/np.log(2); b[valid]=(q[valid]>=c["threshold"]).astype("uint8"); m[valid]=1
    for dst,arr in ((d0,p),(d1,q),(d2,ent),(d3,b),(d4,m)): dst.write(arr.reshape(shape),1,window=w)
 joblib.dump({"model":model,"platt":platt,"features":features,"medians":med,"label_source":"official_product_weak_label","budget":c["target_budget"]},out/"final_target_only_rf_calibrated.joblib")
 fig,ax=plt.subplots(figsize=(5.5,4.5));
 for label,p in (("Uncalibrated",pe),("Platt",pe_cal)):
  bins=pd.qcut(p,10,duplicates="drop"); g=pd.DataFrame({"p":p,"y":ev.class_id.to_numpy(),"b":bins}).groupby("b",observed=True).mean(); ax.plot(g.p,g.y,"o-",label=label)
 ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="Mean predicted probability",ylabel="Weak-label rice frequency",title="Reliability diagram (spatially held-out weak labels)"); ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"reliability_diagram.png",dpi=220)
 (out/"run_metadata.json").write_text(json.dumps({"label_source":"official_product_weak_label","independent_ground_truth":False,"calibration_blocks":len(cal_blocks),"evaluation_blocks":len(set(ev.spatial_block))},indent=2),encoding="utf-8")
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/final_mapping.yaml");a=p.parse_args();print(main_run(a.config))
if __name__=="__main__":main()
