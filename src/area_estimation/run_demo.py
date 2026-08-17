from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio,yaml
from src.data.common import load_config,resolve_data_path,resolve_project_path

def adjusted_proportion(map_values,ref_values,total_by_map,rng,reps=1000):
 total=sum(total_by_map.values()); weights={k:v/total for k,v in total_by_map.items()}
 rates={k:ref_values[map_values==k].mean() for k in (0,1)}; estimate=sum(weights[k]*rates[k] for k in (0,1)); boots=[]
 for _ in range(reps):
  boots.append(sum(weights[k]*rng.choice(ref_values[map_values==k],len(ref_values[map_values==k]),replace=True).mean() for k in (0,1)))
 return estimate,np.asarray(boots),weights,rates

def run(config_file):
 cp=Path(config_file).resolve();c=yaml.safe_load(cp.read_text(encoding="utf-8"));dc,dp=load_config(c["data_config"]);out=resolve_project_path(c["output_dir"],cp); pred_path=out/"rice_binary_calibrated.tif"; ref_path=resolve_data_path(dc["data"]["target_reference"],dp);rng=np.random.default_rng(c["seed"])
 with rasterio.open(pred_path) as ps,rasterio.open(ref_path) as rs:
  pred=ps.read(1);ref=rs.read(1);valid=pred!=255;pixel_area=abs(ps.transform.a*ps.transform.e);total_by={k:int(((pred==k)&valid).sum()) for k in (0,1)}; rows=[]
  for k in (0,1):
   rr,cc=np.where((pred==k)&valid);n=min(c["area_samples_per_map_class"],len(rr));idx=rng.choice(len(rr),n,replace=False)
   for i in idx: rows.append({"row":int(rr[i]),"col":int(cc[i]),"map_class":k,"reference_class":int(ref[rr[i],cc[i]]),"inclusion_probability":n/len(rr),"sample_weight":len(rr)/n,"stratum":f"map_class_{k}"})
 samples=pd.DataFrame(rows); est,boots,weights,rates=adjusted_proportion(samples.map_class.to_numpy(),samples.reference_class.to_numpy(),total_by,rng,c["bootstrap_replicates"]);total_area=sum(total_by.values())*pixel_area/1e6;naive=total_by[1]*pixel_area/1e6;adjusted=est*total_area;lo,hi=np.quantile(boots,[.025,.975])*total_area;official=(ref[valid]>0).sum()*pixel_area/1e6
 result=pd.DataFrame([{"status":"demonstration_using_official_product_weak_labels","naive_mapped_area_km2":naive,"weak_label_adjusted_area_km2":adjusted,"lower_95_km2":lo,"upper_95_km2":hi,"bootstrap_se_km2":boots.std(ddof=1)*total_area,"official_product_area_km2":official,"total_valid_area_km2":total_area,"n_samples":len(samples)}]);samples.to_csv(out/"area_demo_samples.csv",index=False);result.to_csv(out/"area_estimation_demo.csv",index=False);return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/final_mapping.yaml");a=p.parse_args();print(run(a.config).to_string(index=False))
if __name__=="__main__":main()
