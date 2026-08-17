from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import from_bounds
from shapely import make_valid


ROOT=Path(__file__).resolve().parents[3]
P4=ROOT/"outputs/phase4_chongming_staged"
BASE=ROOT/"outputs/phase5_parcel_safeguards"
OUT=BASE/"products"; TABLE=BASE/"tables"; FIG=BASE/"figures"
RAW=P4/"products/shanghai_chongming_parcels.gpkg"; PROB=ROOT/"outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
RGB=P4/"products/prototype_rgb_10m.tif"; DW=P4/"products/dynamic_world_mode_10m.tif"; INDEX=P4/"tiles/tile_index.csv"; CONTRACT=BASE/"phase5_safeguard_contract.json"
THRESHOLD,HIGH=.50,.80


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
    return h.hexdigest()


def reasons(values:list[str])->str:
    return ";".join(dict.fromkeys(values))


def write_raster(path,data,profile,dtype,nodata,desc,tags=None):
    p=profile.copy();p.update(driver="GTiff",count=1,dtype=dtype,nodata=nodata,compress="deflate",tiled=True,blockxsize=256,blockysize=256)
    with rasterio.open(path,"w",**p) as ds:
        ds.write(data.astype(dtype),1);ds.set_band_description(1,desc)
        if tags:ds.update_tags(**{k:str(v) for k,v in tags.items()})


def parcel_values(geom,ds,arr,valid):
    w=from_bounds(*geom.bounds,transform=ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0,0,ds.width,ds.height))
    z=arr[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)];v=valid[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)]
    inside=geometry_mask([geom],out_shape=z.shape,transform=ds.window_transform(w),invert=True,all_touched=False);return z[inside&v]


def neighbor_disagreement(binary,valid):
    h=valid[:,:-1]&valid[:,1:];v=valid[:-1,:]&valid[1:,:];num=int(((binary[:,:-1]!=binary[:,1:])&h).sum()+((binary[:-1,:]!=binary[1:,:])&v).sum());den=int(h.sum()+v.sum());return num/den if den else 0.0


def main()->None:
    for d in [OUT,TABLE,FIG]:d.mkdir(parents=True,exist_ok=True)
    contract=json.loads(CONTRACT.read_text(encoding="utf-8"));raw_hash=sha256(RAW)
    if raw_hash!=contract["immutable_phase4_inputs"]["raw_parcels_sha256"]:raise RuntimeError("Immutable Phase 4 parcel hash changed")
    shutil.copy2(RAW,OUT/"A_raw_ftw_parcels.gpkg");g=gpd.read_file(RAW,layer="parcels");g["phase4_area_m2"]=g.geometry.area;g["phase4_geometry_wkt"]=g.geometry.to_wkt()

    # Geometry-only residual overlap repair; rice probability is not opened until after safeguards are frozen and applied.
    repair_rows=[];participants=set();iteration=0
    while True:
        candidates=[]
        for i,geom in enumerate(g.geometry):
            for j in g.sindex.query(geom,predicate="intersects"):
                if j<=i or g.source_tile_id.iloc[j]==g.source_tile_id.iloc[i]:continue
                area=float(geom.intersection(g.geometry.iloc[j]).area)
                if area>0:candidates.append((area,i,int(j)))
        if not candidates:break
        area,i,j=max(candidates);participants.update([g.temp_id.iloc[i],g.temp_id.iloc[j]])
        di=float(g.rep_distance_to_halo_edge_m.iloc[i]);dj=float(g.rep_distance_to_halo_edge_m.iloc[j]);winner,loser=(i,j) if (di,g.geometry.iloc[i].area,-i)>=(dj,g.geometry.iloc[j].area,-j) else (j,i)
        old=g.geometry.iloc[loser];new=make_valid(old.difference(g.geometry.iloc[winner]));g.at[loser,"geometry"]=new
        repair_rows.append({"iteration":iteration,"winner_temp_id":g.temp_id.iloc[winner],"loser_temp_id":g.temp_id.iloc[loser],"assigned_overlap_m2":area,"loser_area_before_m2":old.area,"loser_area_after_m2":new.area,"winner_rule":"representative point farther from source halo edge"});iteration+=1
        if iteration>100:raise RuntimeError("Overlap repair did not converge")
    pd.DataFrame(repair_rows).to_csv(TABLE/"geometry_only_overlap_repairs.csv",index=False)
    g["geometry_repaired_flag"]=g.temp_id.isin(participants);g["repaired_area_m2"]=g.geometry.area

    hard_all=[];geom_all=[];land_all=[]
    for _,r in g.iterrows():
        hard=[];geom=[];land=[];a=float(r.phase4_area_m2);water=float(r.get("dw_water_fraction",0));built=float(r.get("dw_built_fraction",0));tree=float(r.get("dw_trees_fraction",0));bare=float(r.get("dw_bare_fraction",0))
        if water>=.80:hard.append("water_dominant");land.append("water_dominant")
        elif water>=.50:land.append("water_dominant")
        if built>=.80:hard.append("built_dominant");land.append("built_dominant")
        elif built>=.50:land.append("built_dominant")
        if tree>=.50:land.append("tree_dominant")
        if a>200000 and water+bare>=.70:hard.append("tidal_flat_like");land.append("tidal_flat_like")
        if a>500000:hard.append("extreme_area");geom.append("extreme_area")
        elif a>200000:geom.append("extreme_area")
        if r.temp_id in participants:geom.append("cross_tile_overlap")
        if float(r.repaired_area_m2)<2500:hard.append("cross_tile_overlap");geom.append("cross_tile_overlap")
        if bool(r.deployment_edge_flag):geom.append("deployment_edge")
        if float(r.tile_disagreement_fraction)>=.25:geom.append("high_tile_disagreement")
        hard_all.append(reasons(hard));geom_all.append(reasons(geom));land_all.append(reasons(land))
    g["parcel_exclusion_reason"]=hard_all;g["parcel_geometry_risk"]=geom_all;g["parcel_landcover_risk"]=land_all;g["parcel_valid_for_mapping"]=g.parcel_exclusion_reason=="";g["safeguard_QA_flag"]=(g.parcel_geometry_risk!="")|(g.parcel_landcover_risk!="")
    g["mapping_status"]=np.where(~g.parcel_valid_for_mapping,"excluded",np.where(g.safeguard_QA_flag,"retained_QA_risk","retained_clear"))
    safe=g[g.parcel_valid_for_mapping].copy();risk=g[(~g.parcel_valid_for_mapping)|g.safeguard_QA_flag].copy()
    for path in [OUT/"B_deployment_safe_parcels.gpkg",OUT/"C_QA_risk_only.gpkg"]:
        if path.exists():path.unlink()
    safe.to_file(OUT/"B_deployment_safe_parcels.gpkg",layer="deployment_safe",driver="GPKG");risk.to_file(OUT/"C_QA_risk_only.gpkg",layer="QA_risk",driver="GPKG")

    # Rice outputs begin only here, after safeguard status has been assigned from label-independent evidence.
    with rasterio.open(PROB) as ds:prob=ds.read(1).astype("float32");profile=ds.profile.copy();nodata=float(ds.nodata);valid=np.isfinite(prob)&(prob!=nodata);transform=ds.transform
    raw_mask=rasterize(((x,1) for x in g.geometry),out_shape=prob.shape,transform=transform,fill=0,all_touched=False,dtype="uint8").astype(bool);safe_mask=rasterize(((x,1) for x in safe.geometry),out_shape=prob.shape,transform=transform,fill=0,all_touched=False,dtype="uint8").astype(bool)
    with rasterio.open(DW) as ds:dw=ds.read(1)
    allowed=np.isin(dw,[3,4]).reshape(prob.shape[0],2,prob.shape[1],2).sum(axis=(1,3))>=2;safe_m1b=safe_mask&allowed&valid
    m1=np.full(prob.shape,nodata,dtype="float32");m1[safe_mask&valid]=prob[safe_mask&valid];m1b=np.full(prob.shape,nodata,dtype="float32");m1b[safe_m1b]=prob[safe_m1b]
    write_raster(OUT/"field_mask_deployment_safe_20m.tif",safe_mask.astype("uint8"),profile,"uint8",255,"Phase 5 deployment-safe parcel mask")
    write_raster(OUT/"M1_deployment_safe_probability.tif",m1,profile,"float32",nodata,"M1 recomputed on Phase 5 deployment-safe parcels")
    write_raster(OUT/"M1b_deployment_safe_probability.tif",m1b,profile,"float32",nodata,"M1b recomputed on Phase 5 deployment-safe parcels and frozen Dynamic World rule")
    rows=[]
    with rasterio.open(PROB) as ds:
        for _,r in safe.iterrows():
            vals=parcel_values(r.geometry,ds,prob,valid);rec=r.drop(labels="geometry").to_dict();rec.update({"geometry":r.geometry,"n_valid_pixels_phase5":int(len(vals)),"rice_prob_mean_phase5":float(np.mean(vals)) if len(vals) else np.nan,"rice_prob_median_phase5":float(np.median(vals)) if len(vals) else np.nan,"rice_prob_std_phase5":float(np.std(vals)) if len(vals) else np.nan,"rice_positive_fraction_phase5":float((vals>=THRESHOLD).mean()) if len(vals) else np.nan});rows.append(rec)
    m2=gpd.GeoDataFrame(rows,geometry="geometry",crs=g.crs);m2["class_by_mean_phase5"]=np.where(m2.rice_prob_mean_phase5>=THRESHOLD,"Rice","Non-rice");m2["class_by_median_phase5"]=np.where(m2.rice_prob_median_phase5>=THRESHOLD,"Rice","Non-rice");m2["class_by_positive_fraction_phase5"]=np.where(m2.rice_positive_fraction_phase5>=THRESHOLD,"Rice","Non-rice")
    m2["summary_disagreement_phase5"] = m2[["class_by_mean_phase5","class_by_median_phase5","class_by_positive_fraction_phase5"]].nunique(axis=1)>1
    original_qa=["large_parcel_flag","very_large_parcel_flag","low_pixel_count_flag","built_dominant_flag","tree_dominant_flag","water_dominant_flag","tile_disagreement_flag","cross_tile_reconciliation_flag","deployment_edge_flag"]
    m2["phase5_qa_flag_count"]=m2[original_qa].sum(axis=1).astype(int)+m2.summary_disagreement_phase5.astype(int)+m2.safeguard_QA_flag.astype(int)
    m2["parcel_class_phase5"]=np.where(m2.phase5_qa_flag_count>0,"Uncertain / QA-risk",m2.class_by_mean_phase5)
    p=OUT/"M2_deployment_safe_parcel_predictions.gpkg"
    if p.exists():p.unlink()
    m2.to_file(p,layer="parcel_predictions",driver="GPKG")

    rawpos=valid&(prob>=THRESHOLD);high=valid&(prob>=HIGH);p4=json.loads((P4/"phase4_diagnostics.json").read_text(encoding="utf-8"))
    def stage(mask):return {"retained_aoi_fraction":float(mask.sum()/valid.sum()),"positive_pixels":int((rawpos&mask).sum()),"predicted_rice_area_ha":float((rawpos&mask).sum()*.04),"high_probability_pixels":int((high&mask).sum())}
    before_lc={k:{"count":int((g[f"{k}_dominant_flag"]==True).sum()),"area_ha":float(g.loc[g[f"{k}_dominant_flag"]==True,"phase4_area_m2"].sum()/10000)} for k in ["water","built","tree"]}
    after_lc={k:{"count":int((safe[f"{k}_dominant_flag"]==True).sum()),"area_ha":float(safe.loc[safe[f"{k}_dominant_flag"]==True,"repaired_area_m2"].sum()/10000)} for k in ["water","built","tree"]}
    rawbin=(prob>=THRESHOLD).astype("uint8");parcelmean=rasterize(((x,float(v)) for x,v in zip(m2.geometry,m2.rice_prob_mean_phase5)),out_shape=prob.shape,transform=transform,fill=nodata,all_touched=False,dtype="float32")
    rawnoise=neighbor_disagreement(rawbin,safe_mask&valid);parcelnoise=neighbor_disagreement((parcelmean>=THRESHOLD).astype("uint8"),parcelmean!=nodata)

    idx=pd.read_csv(INDEX);p4t=pd.read_csv(P4/"tables/tile_qa.csv");freeze4=json.loads((P4/"phase4_freeze_manifest.json").read_text(encoding="utf-8"));th=freeze4["tile_warning_thresholds"];tile_rows=[]
    for _,r in idx.iterrows():
        sub=m2[m2.source_tile_id==r.tile_id];w=from_bounds(r.core_west,r.core_south,r.core_east,r.core_north,transform=transform).round_offsets().round_lengths();z=safe_mask[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)];n=max(len(sub),1)
        tile_rows.append({"tile_id":r.tile_id,"parcel_count_after":len(sub),"ftw_retained_fraction_after":float(z.mean()),"built_dominant_fraction_after":float(sub.built_dominant_flag.sum()/n),"tree_dominant_fraction_after":float(sub.tree_dominant_flag.sum()/n),"water_dominant_fraction_after":float(sub.water_dominant_flag.sum()/n),"over_20ha_fraction_after":float(sub.large_parcel_flag.sum()/n),"over_50ha_count_after":int(sub.very_large_parcel_flag.sum()),"over_50ha_fraction_after":float(sub.very_large_parcel_flag.sum()/n),"tile_disagreement_fraction_after":float(sub.tile_disagreement_flag.sum()/n),"summary_disagreement_fraction_after":float(sub.summary_disagreement_phase5.sum()/n),"cross_tile_reconciliation_count_after":int(sub.cross_tile_reconciliation_flag.sum())})
    tq=pd.DataFrame(tile_rows);med=float(tq.cross_tile_reconciliation_count_after.median());mad=float(np.median(np.abs(tq.cross_tile_reconciliation_count_after-med)));clim=med+3*max(mad,1)
    tq["warn_ftw_retained_after"]=tq.ftw_retained_fraction_after>th["ftw_retained_fraction_warning_gt"];tq["warn_built_after"]=tq.built_dominant_fraction_after>th["built_dominant_parcel_fraction_warning_gt"];tq["warn_tree_after"]=tq.tree_dominant_fraction_after>th["tree_dominant_parcel_fraction_warning_gt"];tq["warn_water_after"]=tq.water_dominant_fraction_after>th["water_dominant_parcel_fraction_warning_gt"];tq["warn_over20_after"]=tq.over_20ha_fraction_after>th["over_20ha_parcel_fraction_warning_gt"];tq["warn_over50_after"]=(tq.over_50ha_count_after>=2)|(tq.over_50ha_fraction_after>.03247);tq["warn_disagreement_after"]=tq.tile_disagreement_fraction_after>th["tile_disagreement_parcel_fraction_warning_gt"];tq["warn_summary_after"]=tq.summary_disagreement_fraction_after>th["summary_disagreement_parcel_fraction_warning_gt"];tq["warn_cross_tile_after"]=tq.cross_tile_reconciliation_count_after>clim
    wc=[c for c in tq.columns if c.startswith("warn_")];tq["warning_count_after"]=tq[wc].sum(axis=1);tq["review_required_after"]=tq.warning_count_after>0
    comp=p4t[["tile_id","warning_count","review_required"]].merge(tq,on="tile_id",how="left");comp.to_csv(TABLE/"tile_QA_before_after.csv",index=False)

    stats={"safeguard_counts":{"raw":len(g),"retained":len(safe),"excluded":int((~g.parcel_valid_for_mapping).sum()),"retained_clear":int((g.mapping_status=="retained_clear").sum()),"retained_QA_risk":int((g.mapping_status=="retained_QA_risk").sum()),"QA_risk_layer":len(risk)},"area":{"raw_parcel_area_ha":float(g.phase4_area_m2.sum()/10000),"retained_repaired_area_ha":float(safe.repaired_area_m2.sum()/10000),"retained_fraction":float(safe.repaired_area_m2.sum()/g.phase4_area_m2.sum())},"exclusion_reason_counts":g.loc[~g.parcel_valid_for_mapping,"parcel_exclusion_reason"].str.get_dummies(sep=";").sum().to_dict(),"overlap_repair":{"operations":len(repair_rows),"participants":len(participants),"unresolved_positive_area_pairs":0,"repaired_fragments_below_2500":int(((g.repaired_area_m2<2500)&g.geometry_repaired_flag).sum())},"leakage_before":before_lc,"leakage_after":after_lc,"parcel_size":{"before_over20":int(g.large_parcel_flag.sum()),"after_over20":int(safe.large_parcel_flag.sum()),"before_over50":int(g.very_large_parcel_flag.sum()),"after_over50":int(safe.very_large_parcel_flag.sum())},"M1_before":p4["M1"],"M1_after":stage(safe_mask&valid),"M1b_before":p4["M1b"],"M1b_after":stage(safe_m1b),"aggregation":{"parcel_count":len(m2),"mean_median_correlation":float(m2[["rice_prob_mean_phase5","rice_prob_median_phase5"]].corr().iloc[0,1]),"mean_median_MAD":float(np.abs(m2.rice_prob_mean_phase5-m2.rice_prob_median_phase5).mean()),"three_summary_agreement":float((~m2.summary_disagreement_phase5).mean()),"summary_disagreement_count":int(m2.summary_disagreement_phase5.sum()),"raw_neighbor_disagreement":rawnoise,"parcel_neighbor_disagreement":parcelnoise,"salt_and_pepper_reduction":float((rawnoise-parcelnoise)/max(rawnoise,1e-12))},"tile_QA":{"before_warning_tiles":int(p4t.review_required.sum()),"after_warning_tiles":int(tq.review_required_after.sum()),"warning_counts_after":{c:int(tq[c].sum()) for c in wc},"cross_tile_outlier_limit_after":clim}}
    (BASE/"phase5_diagnostics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")

    with rasterio.open(RGB) as ds:rgb=ds.read().astype("float32");bounds=ds.bounds
    rv=np.all(rgb>0,axis=0);display=np.zeros((rgb.shape[1],rgb.shape[2],3),dtype="uint8")
    for b in range(3):lo,hi=np.percentile(rgb[b][rv],[1,99]);display[...,b]=np.clip((rgb[b]-lo)*255/max(hi-lo,1),0,255).astype("uint8")
    ext=(bounds.left,bounds.right,bounds.bottom,bounds.top)
    fig,axes=plt.subplots(1,3,figsize=(21,8),sharex=True,sharey=True)
    axes[0].imshow(display,extent=ext);g.boundary.plot(ax=axes[0],color="#00e5ff",linewidth=.05);axes[0].set_title(f"A Raw FTW parcels (n={len(g):,})")
    axes[1].imshow(display,extent=ext);safe.boundary.plot(ax=axes[1],color="#00e676",linewidth=.05);axes[1].set_title(f"B Deployment-safe (n={len(safe):,})")
    axes[2].imshow(display,extent=ext);risk.plot(ax=axes[2],column="mapping_status",categorical=True,legend=True,alpha=.55,linewidth=0);axes[2].set_title(f"C Excluded or QA-risk (n={len(risk):,})")
    for ax in axes:ax.ticklabel_format(style="plain",useOffset=False)
    fig.suptitle("Phase 5 label-independent parcel safeguards");fig.tight_layout();fig.savefig(FIG/"phase5_raw_safe_risk_overview.png",dpi=220,bbox_inches="tight");plt.close(fig)

    # Four label-independent zoom selections.
    tqraw=p4t.copy();coastal=tqraw.sort_values(["water_dominant_fraction","over_50ha_count"],ascending=False).iloc[0].tile_id;built=tqraw.sort_values("built_dominant_fraction",ascending=False).iloc[0].tile_id;good="r02c01"
    for tid,name,title in [(coastal,"zoom_northern_coastal_failure.png","Northern coastal/water failure"),(built,"zoom_greenhouse_built_mosaic.png","Greenhouse/built mosaic"),(good,"zoom_good_agricultural_tile.png","Good agricultural structure")]:
        r=idx[idx.tile_id==tid].iloc[0];e=(r.core_west,r.core_east,r.core_south,r.core_north);sub=g[g.source_tile_id==tid];fig,axes=plt.subplots(1,2,figsize=(14,7),sharex=True,sharey=True)
        for ax in axes:ax.imshow(display,extent=ext);ax.set_xlim(e[0],e[1]);ax.set_ylim(e[2],e[3]);ax.ticklabel_format(style="plain",useOffset=False)
        sub.boundary.plot(ax=axes[0],color="#00e5ff",linewidth=.3);axes[0].set_title(f"Raw FTW — {tid}")
        sub[sub.parcel_valid_for_mapping].boundary.plot(ax=axes[1],color="#00e676",linewidth=.3);sub[~sub.parcel_valid_for_mapping].plot(ax=axes[1],color="#d50000",alpha=.45,edgecolor="#d50000",linewidth=.3);sub[(sub.safeguard_QA_flag)&sub.parcel_valid_for_mapping].boundary.plot(ax=axes[1],color="#ffab00",linewidth=.5);axes[1].set_title("Safe green; excluded red; retained risk amber")
        fig.suptitle(f"{title}: {tid}");fig.tight_layout();fig.savefig(FIG/name,dpi=220,bbox_inches="tight");plt.close(fig)
    if repair_rows:
        candidates=[]
        for rr in repair_rows:
            pair=g[g.temp_id.isin([rr["winner_temp_id"],rr["loser_temp_id"]])]
            if len(pair)==2 and pair.parcel_valid_for_mapping.all():candidates.append((pair.geometry.union_all().envelope.area,rr,pair))
        _,rr,pair=min(candidates,key=lambda x:x[0]) if candidates else (0,repair_rows[0],g[g.temp_id.isin([repair_rows[0]["winner_temp_id"],repair_rows[0]["loser_temp_id"]])])
        from shapely import from_wkt
        raw_pair=pair.copy();raw_pair["geometry"]=from_wkt(raw_pair.phase4_geometry_wkt.to_numpy());bb=raw_pair.geometry.union_all().bounds;pad=max(100,(bb[2]-bb[0]+bb[3]-bb[1])*.08);e=(bb[0]-pad,bb[2]+pad,bb[1]-pad,bb[3]+pad)
        fig,axes=plt.subplots(1,2,figsize=(14,7),sharex=True,sharey=True)
        for ax in axes:ax.imshow(display,extent=ext);ax.set_xlim(e[0],e[1]);ax.set_ylim(e[2],e[3]);ax.ticklabel_format(style="plain",useOffset=False)
        raw_pair.boundary.plot(ax=axes[0],color=["#00e5ff","#ffab00"],linewidth=1.2);axes[0].set_title(f"Before: overlap {rr['assigned_overlap_m2']:.0f} m2")
        pair.boundary.plot(ax=axes[1],color=["#00e676","#d500f9"],linewidth=1.2);axes[1].set_title("After: shared area assigned geometrically; overlap 0")
        fig.suptitle(f"Cross-tile repair: {rr['winner_temp_id']} / {rr['loser_temp_id']}");fig.tight_layout();fig.savefig(FIG/"zoom_cross_tile_reconciliation.png",dpi=220,bbox_inches="tight");plt.close(fig)
    manifest={"phase":"5","contract_sha256":sha256(CONTRACT),"immutable_phase4_hashes":{"raw":raw_hash,"phase4_run_manifest":sha256(P4/"run_manifest.json")},"script_sha256":sha256(Path(__file__)),"diagnostics":stats,"outputs_sha256":{str(p.relative_to(BASE)):sha256(p) for d in [OUT,TABLE,FIG] for p in d.iterdir() if p.is_file()},"rules_used_rice_evidence":False,"classifier_retrained":False,"scope":"current central/eastern Chongming prototype only"};(BASE/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (BASE/"processing.log").write_text(f"{datetime.now().astimezone().isoformat(timespec='seconds')} Phase 5 complete raw={len(g)} retained={len(safe)} excluded={int((~g.parcel_valid_for_mapping).sum())} overlap_unresolved=0\n",encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":main()
