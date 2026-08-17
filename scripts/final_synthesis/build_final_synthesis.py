from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final_chongming_parcel_product"
P4 = ROOT / "outputs" / "phase4_chongming_staged"
P5 = ROOT / "outputs" / "phase5_parcel_safeguards"
FIG = FINAL / "figures"
TABLE = FINAL / "tables"
MAN = ROOT / "outputs" / "manifests"
SEED = 20260814


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def setup_ax(ax, title: str):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Easting (m), EPSG:32651", fontsize=8)
    ax.set_ylabel("Northing (m), EPSG:32651", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def raster_array(path: Path, max_dim=1800):
    with rasterio.open(path) as src:
        scale = max(1, int(max(src.height, src.width) / max_dim))
        arr = src.read(out_shape=(src.count, max(1, src.height // scale), max(1, src.width // scale)))
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        nodata = src.nodata
    return arr, extent, nodata


def probability(path: Path):
    arr, extent, nodata = raster_array(path)
    z = arr[0].astype("float32")
    if nodata is not None:
        z[z == nodata] = np.nan
    return z, extent


def save(fig, name):
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#4a5568"})
    raw = gpd.read_file(FINAL / "A_raw_ftw_parcels.gpkg")
    safe = gpd.read_file(FINAL / "B_deployment_safe_parcels.gpkg")
    risk = gpd.read_file(FINAL / "C_QA_risk_only.gpkg")
    probs = gpd.read_file(FINAL / "final_parcel_probability.gpkg")
    classes = gpd.read_file(FINAL / "final_parcel_class.gpkg")

    # Figure 1: workflow
    fig, ax = plt.subplots(figsize=(15, 4.3)); ax.axis("off")
    nodes = [
        ("Jiangxi-trained RF", "Frozen source model"), ("M0 transfer", "Raw 20 m probability"),
        ("FTW PRUE", "Field structure"), ("M1 / M1b", "Field + land-cover gates"),
        ("M2", "Parcel aggregation"), ("Phase-5 safeguards", "Geometry + land-cover QA"),
        ("Final prototype", "Rice / Non-rice / QA-risk"),
    ]
    xs = np.linspace(.07, .93, len(nodes))
    colors = ["#315A7D", "#4472A6", "#4D8B5E", "#66A061", "#D19A45", "#C86B45", "#694F8E"]
    for i, ((a, b), x, color) in enumerate(zip(nodes, xs, colors)):
        box = FancyBboxPatch((x-.063, .40), .126, .28, boxstyle="round,pad=.015", fc=color, ec="white", lw=1.5)
        ax.add_patch(box); ax.text(x, .56, a, ha="center", va="center", color="white", weight="bold", fontsize=10)
        ax.text(x, .47, b, ha="center", va="center", color="white", fontsize=8)
        if i < len(nodes)-1:
            ax.add_patch(FancyArrowPatch((x+.064,.54),(xs[i+1]-.064,.54),arrowstyle="-|>",mutation_scale=13,color="#555",lw=1.4))
    ax.text(.5,.88,"Chongming cross-region rice mapping prototype",ha="center",weight="bold",fontsize=16)
    ax.text(.5,.18,"No Shanghai rice labels used for model, FTW, or safeguard tuning  •  Independent parcel validation pending",ha="center",fontsize=10,color="#555")
    save(fig, "figure_01_study_workflow.png")

    # Figure 2
    z0, ext = probability(FINAL / "M0_raw_probability.tif")
    fig, ax = plt.subplots(figsize=(13, 6)); im=ax.imshow(z0,extent=ext,origin="upper",cmap="viridis",vmin=0,vmax=1)
    setup_ax(ax,"Figure 2. M0 raw Jiangxi→Shanghai rice probability"); fig.colorbar(im,ax=ax,label="Rice probability",shrink=.78)
    save(fig,"figure_02_raw_pixel_probability.png")

    rgb, rgbext, _ = raster_array(P4 / "products" / "prototype_rgb_10m.tif")
    rgb = np.moveaxis(rgb[:3],0,-1).astype(float)
    p2,p98=np.nanpercentile(rgb,(2,98)); rgb=np.clip((rgb-p2)/(p98-p2),0,1)
    # Figure 3
    fig, ax = plt.subplots(figsize=(13,6)); ax.imshow(rgb,extent=rgbext,origin="upper")
    raw.boundary.plot(ax=ax,color="#f6d55c",linewidth=.12,alpha=.85); setup_ax(ax,"Figure 3. FTW field structure over 2022 Sentinel-2 RGB")
    ax.legend(handles=[Patch(facecolor="none",edgecolor="#f6d55c",label=f"Raw FTW parcels (n={len(raw):,})")],loc="lower right")
    save(fig,"figure_03_ftw_field_structure.png")

    # Figure 4
    fig, axs=plt.subplots(1,2,figsize=(16,6),sharex=True,sharey=True)
    for ax,title,data,color in [(axs[0],"A  Raw FTW",raw,"#d94841"),(axs[1],"B  Deployment-safe",safe,"#167d52")]:
        ax.imshow(rgb,extent=rgbext,origin="upper"); data.boundary.plot(ax=ax,color=color,linewidth=.13); setup_ax(ax,title+f" (n={len(data):,})")
    fig.suptitle("Figure 4. Raw versus deployment-safe parcels",fontsize=15,fontweight="bold")
    save(fig,"figure_04_raw_vs_safe_parcels.png")

    # Figure 5
    fig,axs=plt.subplots(1,3,figsize=(18,5),sharex=True,sharey=True)
    for ax,title,path in zip(axs,["M0 raw transfer","M1 safe FTW gate","M1b safe FTW + DW gate"],[FINAL/"M0_raw_probability.tif",FINAL/"M1_deployment_safe_probability.tif",FINAL/"M1b_deployment_safe_probability.tif"]):
        z,e=probability(path); im=ax.imshow(z,extent=e,origin="upper",cmap="viridis",vmin=0,vmax=1); setup_ax(ax,title)
    fig.suptitle("Figure 5. Pixel-level deployment ablation",fontsize=15,fontweight="bold"); fig.colorbar(im,ax=axs.ravel().tolist(),label="Rice probability",shrink=.75,pad=.02)
    save(fig,"figure_05_m0_vs_m1_m1b.png")

    # Figure 6
    fig,ax=plt.subplots(figsize=(13,6)); probs.plot(column="rice_prob_mean_phase5",ax=ax,cmap="viridis",vmin=0,vmax=1,linewidth=0)
    setup_ax(ax,"Figure 6. Final deployment-safe parcel rice probability"); sm=plt.cm.ScalarMappable(cmap="viridis",norm=plt.Normalize(0,1)); fig.colorbar(sm,ax=ax,label="Parcel mean rice probability",shrink=.78)
    save(fig,"figure_06_final_parcel_probability.png")

    # Figure 7
    palette={"Rice":"#2f8f46","Non-rice":"#e6c84f","Uncertain / QA-risk":"#d46b5d"}
    fig,ax=plt.subplots(figsize=(13,6))
    for label,color in palette.items():
        part=classes[classes.parcel_class_phase5==label]; part.plot(ax=ax,color=color,linewidth=0,label=f"{label} ({len(part):,})")
    setup_ax(ax,"Figure 7. Final Rice / Non-rice / QA-risk parcel map")
    ax.legend(handles=[Patch(facecolor=c,edgecolor="none",label=f"{k} ({len(classes[classes.parcel_class_phase5==k]):,})") for k,c in palette.items()],loc="lower right",frameon=True)
    save(fig,"figure_07_final_parcel_class.png")

    # Figure 8
    fig,ax=plt.subplots(figsize=(13,6)); safe.plot(ax=ax,color="#d9e2e8",linewidth=0)
    retained_risk=risk[risk.mapping_status=="retained_QA_risk"]; excluded=risk[risk.mapping_status=="excluded"]
    retained_risk.plot(ax=ax,color="#f3a712",linewidth=.05,label=f"Retained QA-risk ({len(retained_risk):,})")
    excluded.plot(ax=ax,color="#c9362b",linewidth=.05,label=f"Excluded ({len(excluded):,})")
    setup_ax(ax,"Figure 8. QA-risk and excluded parcels")
    ax.legend(handles=[Patch(facecolor="#f3a712",label=f"Retained QA-risk ({len(retained_risk):,})"),Patch(facecolor="#c9362b",label=f"Excluded ({len(excluded):,})")],loc="lower right")
    save(fig,"figure_08_qa_risk_excluded.png")

    # Figure 9: compose already verified Phase-5 zooms.
    zooms=[("Good agricultural morphology","zoom_good_agricultural_tile.png"),("Coastal failure: before / after","zoom_northern_coastal_failure.png"),("Village / greenhouse morphology","zoom_greenhouse_built_mosaic.png"),("Cross-tile repair","zoom_cross_tile_reconciliation.png")]
    fig,axs=plt.subplots(2,2,figsize=(16,11))
    for ax,(title,name) in zip(axs.ravel(),zooms):
        ax.imshow(Image.open(P5/"figures"/name)); ax.axis("off"); ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
    fig.suptitle("Figure 9. Representative morphology and safeguard zooms",fontsize=16,fontweight="bold")
    save(fig,"figure_09_representative_zooms.png")


def make_ablation():
    TABLE.mkdir(parents=True,exist_ok=True)
    rows=[
        ["M0 raw transfer","20 m pixel","none",31079.76,None,None,None,None,None,"Raw positive area; not validated accuracy"],
        ["M1 FTW gating","20 m pixel inside raw FTW", "FTW",9919.00,None,None,None,None,None,"Phase-4 gate; removed pixels are not classification errors"],
        ["M1b FTW + independent land-cover gate","20 m pixel","FTW + Dynamic World",8399.88,None,None,None,None,None,"Phase-4 conservative intersection"],
        ["M2 parcel aggregation","raw FTW parcel","FTW",None,8227,.991681,.985049,.926204,.408168,"Phase-4 structural metrics"],
        ["M2 + deployment safeguards","deployment-safe parcel","FTW + label-independent safeguards",None,7330,.991621,.986494,.907436,2461/7330,"Final prototype; 6 overlap pairs resolved"],
    ]
    cols=["method","spatial_unit","deployment_structure","predicted_rice_area_ha","parcel_count","mean_median_correlation","three_summary_agreement","fragmentation_reduction","qa_risk_fraction","interpretation_limit"]
    pd.DataFrame(rows,columns=cols).to_csv(TABLE/"final_ablation_table.csv",index=False)
    effects=[
        ["parcel_count",8227,7330,"count"],["retained_parcel_fraction",1,7330/8227,"fraction"],
        ["parcel_area",16020.72,8287.04,"ha"],["M1_predicted_rice_area",9919,7728.64,"ha"],
        ["M1b_predicted_rice_area",8399.88,6786.84,"ha"],["water_leakage",5477.04,56.65,"ha"],
        ["built_leakage",654.99,305.94,"ha"],["tree_leakage",745.52,444.23,"ha"],
        ["parcels_over_20ha",69,40,"count"],["parcels_over_50ha",28,0,"count"],
        ["tile_warning_count",26,13,"count"],["unresolved_positive_overlap_pairs",6,0,"count"],
        ["mean_median_correlation",.991681,.991621,"correlation"],["three_summary_agreement",.985049,.986494,"fraction"],
        ["fragmentation_reduction",.926204,.907436,"fraction"],["qa_risk_fraction",.408168,2461/7330,"fraction"],
    ]
    pd.DataFrame(effects,columns=["metric","raw_phase4","deployment_safe_phase5","unit"]).to_csv(TABLE/"final_deployment_effects.csv",index=False)


def spatial_sample(frame: gpd.GeoDataFrame, n: int, rng: np.random.Generator) -> gpd.GeoDataFrame:
    if len(frame)<=n: return frame.copy()
    pieces=[]; groups=[]
    for key,g in frame.groupby("source_tile_id",dropna=False): groups.append((key,g))
    sizes=np.array([len(g) for _,g in groups]); exact=n*sizes/sizes.sum(); alloc=np.floor(exact).astype(int)
    if n>=len(groups): alloc=np.maximum(alloc,1)
    while alloc.sum()>n:
        candidates=np.where(alloc>1)[0]; i=candidates[np.argmin(exact[candidates]-alloc[candidates])]; alloc[i]-=1
    while alloc.sum()<n:
        candidates=np.where(alloc<sizes)[0]; i=candidates[np.argmax(exact[candidates]-alloc[candidates])]; alloc[i]+=1
    for (_,g),k in zip(groups,alloc):
        if k: pieces.append(g.iloc[rng.choice(len(g),size=k,replace=False)])
    return pd.concat(pieces)


def make_validation_sample():
    rng=np.random.default_rng(SEED)
    safe=gpd.read_file(P5/"products"/"M2_deployment_safe_parcel_predictions.gpkg")
    excluded=gpd.read_file(FINAL/"C_QA_risk_only.gpkg"); excluded=excluded[excluded.mapping_status=="excluded"].copy()
    safe["sample_source"]="Product B"; excluded["sample_source"]="Product C"
    selected=[]; used=set()
    def take(name,candidates,n):
        candidates=candidates[~candidates.field_id.astype(str).isin(used)].copy(); s=spatial_sample(candidates,min(n,len(candidates)),rng)
        s["sampling_stratum"]=name; s["target_n"]=n; selected.append(s); used.update(s.field_id.astype(str)); return len(s)
    # Priority preserves rare deployment-risk cases before common clear parcels.
    take("large_or_extreme",pd.concat([safe[safe.large_parcel_flag==1],excluded[excluded.parcel_exclusion_reason.fillna("").str.contains("extreme_area")]]),20)
    take("cross_tile_or_edge",safe[(safe.cross_tile_reconciliation_flag==1)|(safe.deployment_edge_flag==1)],20)
    take("excluded_water_coastal",excluded[excluded.parcel_exclusion_reason.fillna("").str.contains("water_dominant|tidal_flat_like")],30)
    take("excluded_built_greenhouse",excluded[excluded.parcel_exclusion_reason.fillna("").str.contains("built_dominant")],50)
    take("retained_QA_risk",safe[safe.mapping_status=="retained_QA_risk"],80)
    take("clear_non_rice",safe[(safe.mapping_status=="retained_clear")&(safe.parcel_class_phase5=="Non-rice")],60)
    take("clear_rice",safe[(safe.mapping_status=="retained_clear")&(safe.parcel_class_phase5=="Rice")],140)
    sample=gpd.GeoDataFrame(pd.concat(selected,ignore_index=True),crs=safe.crs)
    if len(sample)<400:
        pool=pd.concat([safe,excluded]); take("spatial_balance_fill",pool,400-len(sample)); sample=gpd.GeoDataFrame(pd.concat(selected,ignore_index=True),crs=safe.crs)
    sample=sample.iloc[:400].copy(); sample.insert(0,"validation_id",[f"CMV-{i:04d}" for i in range(1,len(sample)+1)])
    for col in ["reference_label","interpreter_1","interpreter_2","adjudicated_label","reference_confidence","evidence_source","imagery_date","field_visit_date","validation_notes"]: sample[col]=""
    keep=["validation_id","field_id","sample_source","sampling_stratum","target_n","source_tile_id","mapping_status","parcel_class_phase5","area_ha","repaired_area_m2","parcel_exclusion_reason","parcel_geometry_risk","parcel_landcover_risk","reference_label","interpreter_1","interpreter_2","adjudicated_label","reference_confidence","evidence_source","imagery_date","field_visit_date","validation_notes","geometry"]
    keep=[c for c in keep if c in sample.columns]
    out_gpkg=FINAL/"independent_validation_sample_400.gpkg"; sample[keep].to_file(out_gpkg,driver="GPKG",layer="validation_sample")
    csv=sample[keep[:-1]].copy(); csv["centroid_x"]=sample.geometry.centroid.x; csv["centroid_y"]=sample.geometry.centroid.y
    out_csv=TABLE/"independent_validation_sample_400.csv"; csv.to_csv(out_csv,index=False)
    manifest={"design_version":"chongming_parcel_validation_v1","frozen_before_labels":True,"random_seed":SEED,"sample_n":len(sample),"crs":str(sample.crs),"stratum_counts":sample.sampling_stratum.value_counts().to_dict(),"files":{"gpkg":{"path":out_gpkg.relative_to(ROOT).as_posix(),"sha256":sha256(out_gpkg)},"csv":{"path":out_csv.relative_to(ROOT).as_posix(),"sha256":sha256(out_csv)}},"label_fields_initially_blank":True}
    path=MAN/"independent_validation_sampling_manifest.json"; path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    make_ablation(); make_figures(); make_validation_sample(); print("Final synthesis tables, figures, and validation sample created.")
