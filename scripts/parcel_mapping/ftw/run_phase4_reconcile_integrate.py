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
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/phase4_chongming_staged"
TILES = BASE / "tiles"
INDEX = TILES / "tile_index.csv"
FREEZE = BASE / "phase4_freeze_manifest.json"
PROB = ROOT / "outputs/experiments/rf_baseline_m2_v1/shanghai_source_only_probability_s1_s2_fusion.tif"
OUT = BASE / "products"
FIG = BASE / "figures"
TABLE = BASE / "tables"
THRESHOLD, HIGH = 0.50, 0.80


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def write_raster(path:Path,data:np.ndarray,profile:dict,dtype:str,nodata,description:str,tags:dict|None=None)->None:
    p=profile.copy(); p.update(driver="GTiff",count=1,dtype=dtype,nodata=nodata,compress="deflate",tiled=True,blockxsize=256,blockysize=256)
    with rasterio.open(path,"w",**p) as ds:
        ds.write(data.astype(dtype),1); ds.set_band_description(1,description)
        if tags: ds.update_tags(**{k:str(v) for k,v in tags.items()})


def parcel_values(geom, ds, arr, valid):
    w=from_bounds(*geom.bounds,transform=ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0,0,ds.width,ds.height))
    z=arr[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)]; v=valid[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)]
    inside=geometry_mask([geom],out_shape=z.shape,transform=ds.window_transform(w),invert=True,all_touched=False)
    return z[inside&v]


def parcel_fraction(geom, ds, arr):
    w=from_bounds(*geom.bounds,transform=ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0,0,ds.width,ds.height))
    z=arr[int(w.row_off):int(w.row_off+w.height),int(w.col_off):int(w.col_off+w.width)]
    inside=geometry_mask([geom],out_shape=z.shape,transform=ds.window_transform(w),invert=True,all_touched=False)
    return float(z[inside].mean()) if inside.any() else 0.0


def union_find_components(n:int,pairs:list[tuple[int,int]])->list[list[int]]:
    parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for a,b in pairs: union(a,b)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(i)
    return [v for v in groups.values() if len(v)>1]


def neighbor_disagreement(binary,valid):
    h=valid[:,:-1]&valid[:,1:]; v=valid[:-1,:]&valid[1:,:]
    num=int(((binary[:,:-1]!=binary[:,1:])&h).sum()+((binary[:-1,:]!=binary[1:,:])&v).sum()); den=int(h.sum()+v.sum())
    return num/den if den else 0.0


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); TABLE.mkdir(parents=True,exist_ok=True)
    freeze=json.loads(FREEZE.read_text(encoding="utf-8")); idx=pd.read_csv(INDEX)
    tile_manifests={tid:json.loads((TILES/tid/"tile_manifest.json").read_text(encoding="utf-8")) for tid in idx.tile_id}
    incomplete=[k for k,v in tile_manifests.items() if v.get("status")!="complete"]
    if incomplete: raise RuntimeError(f"Incomplete tiles: {incomplete}")
    owner_parts=[]
    for tid in idx.tile_id:
        g=gpd.read_file(TILES/tid/"ftw_polygons_owner.gpkg",layer="fields"); g["source_local_index"]=np.arange(len(g)); g["temp_id"]=[f"{tid}_{i:05d}" for i in range(len(g))]
        halo=box(*tile_manifests[tid]["halo_extent"]); g["rep_distance_to_halo_edge_m"]=[x.representative_point().distance(halo.boundary) for x in g.geometry]; owner_parts.append(g)
    all_owner=gpd.GeoDataFrame(pd.concat(owner_parts,ignore_index=True),geometry="geometry",crs=owner_parts[0].crs)
    duplicate_pairs=[]
    for i,g in enumerate(all_owner.geometry):
        for j in all_owner.sindex.query(g,predicate="intersects"):
            if j<=i or all_owner.source_tile_id.iloc[j]==all_owner.source_tile_id.iloc[i]: continue
            inter=g.intersection(all_owner.geometry.iloc[j]).area; ratio=inter/max(min(g.area,all_owner.geometry.iloc[j].area),1)
            if ratio>=0.50: duplicate_pairs.append((i,int(j)))
    components=union_find_components(len(all_owner),duplicate_pairs); drop=set(); reconciliation=[]; reconciled_kept=set()
    for component in components:
        keep=max(component,key=lambda i:(all_owner.rep_distance_to_halo_edge_m.iloc[i],all_owner.geometry.iloc[i].area,-i)); reconciled_kept.add(keep)
        for i in component:
            if i!=keep: drop.add(i); reconciliation.append({"kept_temp_id":all_owner.temp_id.iloc[keep],"dropped_temp_id":all_owner.temp_id.iloc[i],"kept_tile":all_owner.source_tile_id.iloc[keep],"dropped_tile":all_owner.source_tile_id.iloc[i],"overlap_threshold":0.50,"rule":"retain version whose representative point is farther from source halo edge"})
    parcels=all_owner.drop(index=list(drop)).copy().reset_index(drop=True)
    parcels["cross_tile_reconciliation_flag"]=parcels.core_crossing_flag.astype(bool)|parcels.source_halo_edge_flag.astype(bool)|parcels.temp_id.isin([all_owner.temp_id.iloc[i] for i in reconciled_kept])
    parcels["duplicate_versions_removed"]=parcels.temp_id.map(pd.Series([r["kept_temp_id"] for r in reconciliation]).value_counts()).fillna(0).astype(int)
    pd.DataFrame(reconciliation).to_csv(TABLE/"cross_tile_reconciliation_audit.csv",index=False)
    residual_overlap=[]; residual_indices=set()
    for i,g in enumerate(parcels.geometry):
        for j in parcels.sindex.query(g,predicate="intersects"):
            if j<=i or parcels.source_tile_id.iloc[j]==parcels.source_tile_id.iloc[i]: continue
            area=float(g.intersection(parcels.geometry.iloc[j]).area)
            if area>0:
                residual_indices.update([i,int(j)]); residual_overlap.append({"temp_id_a":parcels.temp_id.iloc[i],"temp_id_b":parcels.temp_id.iloc[j],"tile_a":parcels.source_tile_id.iloc[i],"tile_b":parcels.source_tile_id.iloc[j],"overlap_area_m2":area,"overlap_vs_smaller":area/max(min(g.area,parcels.geometry.iloc[j].area),1),"status":"retained_and_QA_flagged; below frozen 0.50 duplicate threshold"})
    if residual_indices: parcels.loc[list(residual_indices),"cross_tile_reconciliation_flag"]=True
    pd.DataFrame(residual_overlap).to_csv(TABLE/"residual_cross_tile_overlap_audit.csv",index=False)
    parcels["field_id"]=[f"CMFTW_{i:06d}" for i in range(1,len(parcels)+1)]; parcels["area_m2"]=parcels.area; parcels["area_ha"]=parcels.area/10000

    with rasterio.open(PROB) as ds:
        prob=ds.read(1).astype("float32"); profile=ds.profile.copy(); nodata=float(ds.nodata); valid=np.isfinite(prob)&(prob!=nodata); bounds=ds.bounds; transform=ds.transform; crs=ds.crs
    extent_geom=box(*bounds); parcels["deployment_edge_flag"]=[g.distance(extent_geom.boundary)<=1e-6 for g in parcels.geometry]
    h10,w10=int(round((bounds.top-bounds.bottom)/10)),int(round((bounds.right-bounds.left)/10)); transform10=rasterio.Affine(10,0,bounds.left,0,-10,bounds.top)
    rgb=np.zeros((3,h10,w10),dtype="uint16"); dw=np.zeros((h10,w10),dtype="uint8"); disagreement=np.zeros((h10,w10),dtype="uint8"); semantic=np.zeros((h10,w10),dtype="uint8")
    for _,r in idx.iterrows():
        tid=r.tile_id; core=(r.core_west,r.core_south,r.core_east,r.core_north); fullwin=from_bounds(*core,transform=transform10).round_offsets().round_lengths()
        for path,target in [(TILES/tid/"rgb_input.tif",rgb),(TILES/tid/"dynamic_world_mode.tif",dw),(TILES/tid/"ftw_semantic_native_disagreements.tif",disagreement),(TILES/tid/"ftw_semantic_native.tif",semantic)]:
            with rasterio.open(path) as ds:
                srcwin=from_bounds(*core,transform=ds.transform).round_offsets().round_lengths(); data=ds.read(window=srcwin)
            rr=slice(int(fullwin.row_off),int(fullwin.row_off+fullwin.height)); cc=slice(int(fullwin.col_off),int(fullwin.col_off+fullwin.width))
            if target.ndim==3: target[:,rr,cc]=data
            else: target[rr,cc]=data[0]
    p10=profile.copy(); p10.update(width=w10,height=h10,transform=transform10,resolution=(10,10))
    with rasterio.open(OUT/"prototype_rgb_10m.tif","w",driver="GTiff",height=h10,width=w10,count=3,dtype="uint16",crs=crs,transform=transform10,nodata=0,compress="deflate",tiled=True,blockxsize=256,blockysize=256) as ds: ds.write(rgb); ds.descriptions=("B04_red","B03_green","B02_blue")
    write_raster(OUT/"dynamic_world_mode_10m.tif",dw,p10,"uint8",255,"2022 Dynamic World temporal mode; 0 is valid water")
    write_raster(OUT/"ftw_tile_disagreement_10m.tif",disagreement,p10,"uint8",0,"Mosaicked official FTW disagreement diagnostics")
    write_raster(OUT/"ftw_semantic_core_mosaic_10m.tif",semantic,p10,"uint8",255,"Core-owned FTW semantic mosaic; 0 background,1 interior,2 boundary")

    field_mask=rasterize(((g,1) for g in parcels.geometry),out_shape=prob.shape,transform=transform,fill=0,all_touched=False,dtype="uint8").astype(bool)
    allowed=(np.isin(dw,[3,4])).reshape(prob.shape[0],2,prob.shape[1],2).sum(axis=(1,3))>=2; m1b_mask=field_mask&allowed&valid
    raw_binary=np.zeros(prob.shape,dtype="uint8"); raw_binary[valid]=prob[valid]>=THRESHOLD
    m1=np.full(prob.shape,nodata,dtype="float32"); m1[field_mask&valid]=prob[field_mask&valid]
    m1b=np.full(prob.shape,nodata,dtype="float32"); m1b[m1b_mask]=prob[m1b_mask]
    m0out=prob.copy(); shutil.copy2(PROB,OUT/"M0_raw_probability.tif")
    write_raster(OUT/"field_mask_20m.tif",field_mask.astype("uint8"),profile,"uint8",255,"Reconciled accepted FTW parcel mask; all_touched=False")
    write_raster(OUT/"M1_ftw_gated_probability.tif",m1,profile,"float32",nodata,"M1 FTW-gated retained rice probability")
    write_raster(OUT/"M1b_ftw_dynamicworld_gated_probability.tif",m1b,profile,"float32",nodata,"M1b FTW plus frozen Dynamic World gated probability")

    with rasterio.open(PROB) as probds, rasterio.open(OUT/"ftw_tile_disagreement_10m.tif") as dds:
        disarr=dds.read(1)>0
        rows=[]
        for _,f in parcels.iterrows():
            vals=parcel_values(f.geometry,probds,prob,valid); rec=f.drop(labels="geometry").to_dict(); rec.update({"geometry":f.geometry,"n_valid_pixels":int(len(vals)),"rice_prob_mean":float(np.mean(vals)) if len(vals) else np.nan,"rice_prob_median":float(np.median(vals)) if len(vals) else np.nan,"rice_prob_std":float(np.std(vals)) if len(vals) else np.nan,"rice_positive_fraction":float((vals>=THRESHOLD).mean()) if len(vals) else np.nan,"tile_disagreement_fraction":parcel_fraction(f.geometry,dds,disarr),"large_parcel_flag":bool(f.geometry.area>200000),"very_large_parcel_flag":bool(f.geometry.area>500000),"low_pixel_count_flag":bool(len(vals)<4),"built_dominant_flag":bool(f.get("dw_built_fraction",0)>=.5),"tree_dominant_flag":bool(f.get("dw_trees_fraction",0)>=.5),"water_dominant_flag":bool(f.get("dw_water_fraction",0)>=.5)}); rows.append(rec)
    parcels=gpd.GeoDataFrame(rows,geometry="geometry",crs=crs); parcels["tile_disagreement_flag"]=parcels.tile_disagreement_fraction>=.05
    parcels["class_by_mean"]=np.where(parcels.rice_prob_mean>=THRESHOLD,"Rice","Non-rice"); parcels["class_by_median"]=np.where(parcels.rice_prob_median>=THRESHOLD,"Rice","Non-rice"); parcels["class_by_positive_fraction"]=np.where(parcels.rice_positive_fraction>=THRESHOLD,"Rice","Non-rice")
    parcels["summary_disagreement_flag"]=parcels[["class_by_mean","class_by_median","class_by_positive_fraction"]].nunique(axis=1)>1
    flags=["large_parcel_flag","very_large_parcel_flag","low_pixel_count_flag","built_dominant_flag","tree_dominant_flag","water_dominant_flag","tile_disagreement_flag","cross_tile_reconciliation_flag","deployment_edge_flag","summary_disagreement_flag"]
    parcels["qa_flag_count"]=parcels[flags].sum(axis=1).astype(int); parcels["parcel_class"]=np.where(parcels.qa_flag_count>0,"Uncertain / QA-risk",parcels.class_by_mean)
    for name in ["shanghai_chongming_parcels.gpkg","parcel_rice_probability.gpkg","parcel_classification.gpkg","parcel_qa.gpkg"]:
        path=OUT/name
        if path.exists(): path.unlink()
    parcels.to_file(OUT/"shanghai_chongming_parcels.gpkg",layer="parcels",driver="GPKG")
    parcels[["field_id","source_tile_id","area_m2","area_ha","n_valid_pixels","rice_prob_mean","rice_prob_median","rice_prob_std","rice_positive_fraction","geometry"]].to_file(OUT/"parcel_rice_probability.gpkg",layer="parcel_probability",driver="GPKG")
    parcels[["field_id","class_by_mean","class_by_median","class_by_positive_fraction","parcel_class","geometry"]].to_file(OUT/"parcel_classification.gpkg",layer="parcel_class",driver="GPKG")
    parcels[["field_id",*flags,"tile_disagreement_fraction","qa_flag_count","parcel_class","geometry"]].to_file(OUT/"parcel_qa.gpkg",layer="parcel_qa",driver="GPKG")

    parcel_mean=rasterize(((g,float(v)) for g,v in zip(parcels.geometry,parcels.rice_prob_mean)),out_shape=prob.shape,transform=transform,fill=nodata,all_touched=False,dtype="float32"); write_raster(OUT/"parcel_mean_probability_20m.tif",parcel_mean,profile,"float32",nodata,"Parcel mean rice probability")
    classmap=rasterize(((g,{"Non-rice":0,"Rice":1,"Uncertain / QA-risk":2}[v]) for g,v in zip(parcels.geometry,parcels.parcel_class)),out_shape=prob.shape,transform=transform,fill=255,all_touched=False,dtype="uint8"); write_raster(OUT/"parcel_class_20m.tif",classmap,profile,"uint8",255,"0 non-rice,1 rice,2 uncertain/QA-risk")

    rawpos=valid&(prob>=THRESHOLD); high=valid&(prob>=HIGH)
    def stage(mask): return {"retained_aoi_fraction":float(mask.sum()/valid.sum()),"raw_positive_retained":int((rawpos&mask).sum()),"raw_positive_removed":int((rawpos&~mask).sum()),"raw_positive_retained_fraction":float((rawpos&mask).sum()/max(rawpos.sum(),1)),"high_probability_retained":int((high&mask).sum()),"high_probability_removed":int((high&~mask).sum()),"predicted_rice_area_ha":float((rawpos&mask).sum()*.04)}
    stats={"M0":{"valid_pixels":int(valid.sum()),"positive_pixels":int(rawpos.sum()),"high_probability_pixels":int(high.sum()),"predicted_rice_area_ha":float(rawpos.sum()*.04)},"M1":stage(field_mask&valid),"M1b":stage(m1b_mask),"parcels":{"count":len(parcels),"mean_median_correlation":float(parcels[["rice_prob_mean","rice_prob_median"]].corr().iloc[0,1]),"mean_median_mad":float(np.abs(parcels.rice_prob_mean-parcels.rice_prob_median).mean()),"three_summary_agreement":float((~parcels.summary_disagreement_flag).mean()),"summary_disagreement_count":int(parcels.summary_disagreement_flag.sum()),"large_count":int(parcels.large_parcel_flag.sum()),"very_large_count":int(parcels.very_large_parcel_flag.sum()),"low_pixel_count":int(parcels.low_pixel_count_flag.sum()),"built_dominant_count":int(parcels.built_dominant_flag.sum()),"tree_dominant_count":int(parcels.tree_dominant_flag.sum()),"water_dominant_count":int(parcels.water_dominant_flag.sum()),"tile_disagreement_count":int(parcels.tile_disagreement_flag.sum()),"cross_tile_reconciliation_count":int(parcels.cross_tile_reconciliation_flag.sum()),"deployment_edge_count":int(parcels.deployment_edge_flag.sum()),"qa_risk_count":int((parcels.parcel_class=="Uncertain / QA-risk").sum()),"qa_risk_fraction":float((parcels.parcel_class=="Uncertain / QA-risk").mean()),"class_counts":parcels.parcel_class.value_counts().to_dict(),"raw_neighbor_disagreement":neighbor_disagreement(raw_binary,field_mask&valid),"parcel_neighbor_disagreement":neighbor_disagreement((parcel_mean>=THRESHOLD).astype("uint8"),parcel_mean!=nodata)},"reconciliation":{"input_owner_count":len(all_owner),"duplicate_components":len(components),"duplicate_versions_removed":len(drop),"residual_low_overlap_pair_count":len(residual_overlap),"residual_low_overlap_area_m2":float(sum(r["overlap_area_m2"] for r in residual_overlap)),"residual_low_overlap_parcels_flagged":len(residual_indices),"final_count":len(parcels)}}
    stats["parcels"]["neighbor_disagreement_reduction_fraction"]=(stats["parcels"]["raw_neighbor_disagreement"]-stats["parcels"]["parcel_neighbor_disagreement"])/max(stats["parcels"]["raw_neighbor_disagreement"],1e-12)

    tile_rows=[]; th=freeze["tile_warning_thresholds"]
    for _,r in idx.iterrows():
        sub=parcels[parcels.source_tile_id==r.tile_id]; corewin=from_bounds(r.core_west,r.core_south,r.core_east,r.core_north,transform=transform).round_offsets().round_lengths(); fm=field_mask[int(corewin.row_off):int(corewin.row_off+corewin.height),int(corewin.col_off):int(corewin.col_off+corewin.width)]
        n=max(len(sub),1); rec={"tile_id":r.tile_id,"parcel_count":len(sub),"ftw_retained_fraction":float(fm.mean()),"built_dominant_fraction":float(sub.built_dominant_flag.sum()/n),"tree_dominant_fraction":float(sub.tree_dominant_flag.sum()/n),"water_dominant_fraction":float(sub.water_dominant_flag.sum()/n),"over_20ha_fraction":float(sub.large_parcel_flag.sum()/n),"over_50ha_count":int(sub.very_large_parcel_flag.sum()),"over_50ha_fraction":float(sub.very_large_parcel_flag.sum()/n),"tile_disagreement_fraction":float(sub.tile_disagreement_flag.sum()/n),"summary_disagreement_fraction":float(sub.summary_disagreement_flag.sum()/n),"cross_tile_reconciliation_count":int(sub.cross_tile_reconciliation_flag.sum())}; tile_rows.append(rec)
    tq=pd.DataFrame(tile_rows); med=float(tq.cross_tile_reconciliation_count.median()); mad=float(np.median(np.abs(tq.cross_tile_reconciliation_count-med))); cross_limit=med+3*max(mad,1)
    tq["warn_ftw_retained"]=tq.ftw_retained_fraction>th["ftw_retained_fraction_warning_gt"]; tq["warn_built"]=tq.built_dominant_fraction>th["built_dominant_parcel_fraction_warning_gt"]; tq["warn_tree"]=tq.tree_dominant_fraction>th["tree_dominant_parcel_fraction_warning_gt"]; tq["warn_water"]=tq.water_dominant_fraction>th["water_dominant_parcel_fraction_warning_gt"]; tq["warn_over20"]=tq.over_20ha_fraction>th["over_20ha_parcel_fraction_warning_gt"]; tq["warn_over50"]=(tq.over_50ha_count>=2)|(tq.over_50ha_fraction>0.03247); tq["warn_disagreement"]=tq.tile_disagreement_fraction>th["tile_disagreement_parcel_fraction_warning_gt"]; tq["warn_summary"]=tq.summary_disagreement_fraction>th["summary_disagreement_parcel_fraction_warning_gt"]; tq["warn_cross_tile"]=tq.cross_tile_reconciliation_count>cross_limit
    warncols=[c for c in tq.columns if c.startswith("warn_")]; tq["warning_count"]=tq[warncols].sum(axis=1); tq["review_required"]=tq.warning_count>0; tq.to_csv(TABLE/"tile_qa.csv",index=False)
    try: tq.to_parquet(TABLE/"tile_qa.parquet",index=False)
    except Exception as exc: (TABLE/"tile_qa_parquet_error.txt").write_text(repr(exc),encoding="utf-8")
    stats["tile_qa"]={"tile_count":len(tq),"review_required_count":int(tq.review_required.sum()),"cross_tile_robust_limit":cross_limit,"warning_counts":{c:int(tq[c].sum()) for c in warncols}}
    (BASE/"phase4_diagnostics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")

    validrgb=np.all(rgb>0,axis=0); display=np.zeros((h10,w10,3),dtype="uint8")
    for b in range(3): lo,hi=np.percentile(rgb[b][validrgb],[1,99]); display[...,b]=np.clip((rgb[b]-lo)*255/max(hi-lo,1),0,255).astype("uint8")
    ext=(bounds.left,bounds.right,bounds.bottom,bounds.top)
    fig,ax=plt.subplots(figsize=(14,10)); ax.imshow(display,extent=ext); parcels.boundary.plot(ax=ax,color="#00e5ff",linewidth=.08); ax.set_title("A. Central/eastern Chongming prototype: RGB + reconciled FTW parcels"); ax.ticklabel_format(style="plain",useOffset=False); fig.tight_layout(); fig.savefig(FIG/"A_full_prototype_overview.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    for col,name,title in [("rice_prob_mean","B_parcel_rice_probability_map.png","B. Parcel mean rice probability"),("qa_flag_count","D_QA_risk_map.png","D. Parcel QA-risk count")]:
        fig,ax=plt.subplots(figsize=(14,10)); parcels.plot(ax=ax,column=col,cmap="YlGn" if col=="rice_prob_mean" else "YlOrRd",vmin=0,vmax=1 if col=="rice_prob_mean" else max(int(parcels[col].max()),1),linewidth=0,legend=True); ax.set_title(title); ax.ticklabel_format(style="plain",useOffset=False); fig.tight_layout(); fig.savefig(FIG/name,dpi=220,bbox_inches="tight"); plt.close(fig)
    colors={"Rice":"#2e7d32","Non-rice":"#eceff1","Uncertain / QA-risk":"#ef6c00"}; fig,ax=plt.subplots(figsize=(14,10))
    for c,color in colors.items():
        s=parcels[parcels.parcel_class==c]
        if len(s): s.plot(ax=ax,color=color,edgecolor="#37474f",linewidth=.05)
    ax.legend(handles=[mpatches.Patch(facecolor=v,edgecolor="#37474f",label=k) for k,v in colors.items()]); ax.set_title("C. Diagnostic parcel class"); ax.ticklabel_format(style="plain",useOffset=False); fig.tight_layout(); fig.savefig(FIG/"C_parcel_class_map.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(20,7),sharex=True,sharey=True)
    for ax,data,title in zip(axes,[prob,m1,m1b],["M0 raw probability","M1 FTW-gated","M1b FTW + Dynamic World"]): ax.imshow(np.ma.masked_where((data==nodata)|~np.isfinite(data),data),extent=ext,cmap="YlGn",vmin=0,vmax=1); ax.set_title(title); ax.ticklabel_format(style="plain",useOffset=False)
    fig.suptitle("E. Frozen deployment comparison; removal is spatial filtering, not accuracy"); fig.tight_layout(); fig.savefig(FIG/"E_M0_M1_M1b_comparison.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    good=tq.sort_values(["warning_count","parcel_count"],ascending=[True,False]).iloc[0].tile_id; bad=tq.sort_values(["warning_count","parcel_count"],ascending=[False,False]).iloc[0].tile_id
    fig,axes=plt.subplots(2,2,figsize=(14,12))
    for rowi,tid in enumerate([good,bad]):
        r=idx[idx.tile_id==tid].iloc[0]; e=(r.core_west,r.core_east,r.core_south,r.core_north); sub=parcels[parcels.source_tile_id==tid]
        axes[rowi,0].imshow(display,extent=ext); axes[rowi,0].set_xlim(e[0],e[1]); axes[rowi,0].set_ylim(e[2],e[3]); sub.boundary.plot(ax=axes[rowi,0],color="#00e5ff",linewidth=.25); axes[rowi,0].set_title(f"{'Good' if rowi==0 else 'Problematic'} tile {tid}: RGB + parcels")
        sub.plot(ax=axes[rowi,1],column="qa_flag_count",cmap="YlOrRd",vmin=0,vmax=max(int(parcels.qa_flag_count.max()),1),edgecolor="#455a64",linewidth=.12); axes[rowi,1].set_xlim(e[0],e[1]); axes[rowi,1].set_ylim(e[2],e[3]); axes[rowi,1].set_title(f"{tid}: QA-risk")
    for ax in axes.ravel(): ax.ticklabel_format(style="plain",useOffset=False)
    fig.tight_layout(); fig.savefig(FIG/"F_zoomed_good_problematic_examples.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    run_manifest={"phase":"4","freeze_manifest_sha256":sha256(FREEZE),"script_sha256":sha256(Path(__file__)),"probability_sha256":sha256(PROB),"tile_manifests_sha256":{k:sha256(TILES/k/"tile_manifest.json") for k in idx.tile_id},"reconciliation":freeze["cross_tile_reconciliation"],"diagnostics":stats,"outputs_sha256":{str(p.relative_to(BASE)):sha256(p) for d in [OUT,FIG,TABLE] for p in d.iterdir() if p.is_file()},"classifier_retrained":False,"target_labels_used":False,"full_shanghai_run":False,"scope":"central/eastern Chongming prototype"}
    (BASE/"run_manifest.json").write_text(json.dumps(run_manifest,indent=2),encoding="utf-8")
    with (BASE/"processing.log").open("a",encoding="utf-8") as f: f.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} PHASE4 INTEGRATION COMPLETE parcels={len(parcels)} tiles_review={int(tq.review_required.sum())}\n")
    print(json.dumps(stats,indent=2))


if __name__=="__main__": main()
