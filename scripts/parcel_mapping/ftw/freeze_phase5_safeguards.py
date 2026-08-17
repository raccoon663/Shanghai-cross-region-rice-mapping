from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
P4 = ROOT / "outputs/phase4_chongming_staged"
OUT = ROOT / "outputs/phase5_parcel_safeguards"
RAW = P4 / "products/shanghai_chongming_parcels.gpkg"
OVERLAP = P4 / "tables/residual_cross_tile_overlap_audit.csv"
FREEZE4 = P4 / "phase4_freeze_manifest.json"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    contract={
        "phase":"5","name":"targeted label-independent field-geometry and deployment safeguards","frozen_at":datetime.now().astimezone().isoformat(timespec="seconds"),
        "immutable_phase4_inputs":{"raw_parcels":str(RAW.relative_to(ROOT)),"raw_parcels_sha256":sha256(RAW),"residual_overlap_audit":str(OVERLAP.relative_to(ROOT)),"residual_overlap_audit_sha256":sha256(OVERLAP),"phase4_freeze_manifest_sha256":sha256(FREEZE4)},
        "evidence_allowed":["Dynamic World class fractions","parcel area","shape metrics","official FTW tile disagreement","cross-tile reconciliation metadata","deployment edge context"],
        "evidence_forbidden":["rice probability for rule definition","Shanghai rice labels","official Shanghai rice product","target-dependent threshold tuning"],
        "hard_exclusion_rules":{
            "water_dominant":"dw_water_fraction >= 0.80",
            "built_dominant":"dw_built_fraction >= 0.80",
            "tidal_flat_like":"area > 20 ha AND (dw_water_fraction + dw_bare_fraction) >= 0.70",
            "extreme_area":"area > 50 ha",
            "cross_tile_overlap":"only if geometry-only repair leaves a fragment below the frozen 2,500 m2 accepted minimum"
        },
        "soft_QA_rules":{
            "water_dominant":"0.50 <= dw_water_fraction < 0.80",
            "built_dominant":"0.50 <= dw_built_fraction < 0.80",
            "tree_dominant":"dw_trees_fraction >= 0.50",
            "extreme_area":"20 ha < area <= 50 ha",
            "deployment_edge":"touches the retained prototype boundary",
            "high_tile_disagreement":"tile_disagreement_fraction >= 0.25",
            "cross_tile_overlap":"participates in one of the Phase 4 residual overlap pairs, even after repair"
        },
        "overlap_repair":{
            "algorithm":"process residual cross-tile overlap pairs from largest overlap to smallest; assign shared area to the polygon whose representative point is farther from its source halo edge; subtract only the shared area from the other polygon; repeat until no positive-area cross-tile overlap remains",
            "minimum_repaired_area_m2":2500,
            "if_below_minimum":"preserve feature in raw and QA-risk products, set parcel_valid_for_mapping=false with cross_tile_overlap exclusion reason",
            "rice_evidence_used":False
        },
        "fields":{"parcel_valid_for_mapping":"false only for a hard exclusion","parcel_exclusion_reason":"semicolon-separated hard reasons","parcel_geometry_risk":"semicolon-separated geometry hard/soft reasons","parcel_landcover_risk":"semicolon-separated land-cover hard/soft reasons","mapping_status":["retained_clear","retained_QA_risk","excluded"]},
        "product_semantics":{"A":"immutable raw Phase 4 parcel copy","B":"geometry-repaired parcels with parcel_valid_for_mapping=true; ambiguous soft-risk parcels retained","C":"all hard-excluded or soft-flagged parcels, including their reasons"},
        "rice_recomputation":"only after this contract is written; safe set uses parcel_valid_for_mapping=true without rice-dependent selection",
        "scope":"current central/eastern Chongming prototype only; no all-Shanghai extension"
    }
    path=OUT/"phase5_safeguard_contract.json"; path.write_text(json.dumps(contract,indent=2),encoding="utf-8")
    print(json.dumps({"frozen_at":contract["frozen_at"],"contract_sha256":sha256(path),"raw_phase4_sha256":sha256(RAW)},indent=2))


if __name__=="__main__": main()
