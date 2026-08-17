# Parcel-mapping scripts

These scripts are public copies of the executed field/parcel workflow previously retained under the ignored `work/` directory. Copying them here does not change any frozen result.

## Recommended reading order

1. `select_aoi_label_free.py` and `ftw/select_and_freeze_aoi2.py` — label-independent AOI selection.
2. `run_dav2_aoi.py` — DAv2 AOI baseline.
3. `ftw/run_ftw_official.py`, `ftw/polygonize_ftw.py`, and `ftw/qa_compare_ftw_dav2.py` — FTW inference and controlled comparison.
4. `ftw/run_phase3a_parcel_integration.py` and `ftw/run_phase3b_parcel_integration.py` — two-AOI M1/M2 integration.
5. `ftw/freeze_phase4_contract.py`, `ftw/run_phase4_tiles.py`, and `ftw/run_phase4_reconcile_integrate.py` — staged 30-tile prototype and reconciliation.
6. `ftw/freeze_phase5_safeguards.py` and `ftw/run_phase5_safeguards.py` — label-independent deployment safeguards.
7. `../final_synthesis/` — final product freeze, tables, figures, and validation sample.

`select_aoi.py` is preserved only as the invalidated label-aided selection path. The executed valid Phase-2 experiment used `select_aoi_label_free.py`.

External FTW/DAv2 source trees, checkpoints, virtual environments, Earth Engine credentials, and large intermediate rasters are intentionally excluded. See the execution manifests and technical reports for exact commits, hashes, environments, and input contracts.
