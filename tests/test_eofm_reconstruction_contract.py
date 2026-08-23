import importlib.util
import json
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results/manifests/eofm_temporal_grid_freeze.json"
CONFIG = ROOT / "configs/eofm_input_reconstruction.yaml"
SCRIPT = ROOT / "scripts/eofm/02_submit_presto_point_exports.py"
PLAN = ROOT / "results/manifests/eofm_presto_export_plan.json"
SUBMISSION = ROOT / "results/manifests/eofm_ee_submission_freeze.json"
INPUT_VALIDATION = ROOT / "results/manifests/eofm_presto_input_validation.json"
SMOKE = ROOT / "results/manifests/eofm_presto_smoke_test.json"
PREPARE = ROOT / "scripts/eofm/03_prepare_presto_inputs.py"


def load_submitter():
    spec = importlib.util.spec_from_file_location("eofm_submitter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_prepare():
    spec = importlib.util.spec_from_file_location("eofm_prepare", PREPARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_temporal_grid_is_exact_and_ordered():
    grid = json.loads(GRID.read_text(encoding="utf-8"))
    assert grid["timestep_count"] == 23 == len(grid["windows"])
    assert [w["tag"] for w in grid["windows"]] == [f"t{i:02d}" for i in range(1, 24)]
    assert grid["windows"][0]["anchor"] == "2022-03-02"
    assert grid["windows"][-1]["anchor"] == "2022-11-12"
    for window in grid["windows"]:
        assert (date.fromisoformat(window["end_exclusive"]) - date.fromisoformat(window["start_inclusive"])).days == 17


def test_presto_native_band_order_and_schema():
    module = load_submitter()
    config, grid = module.load_contract(CONFIG)
    assert config["presto_input"]["s1_band_order"] == ["VV", "VH"]
    assert config["presto_input"]["s2_band_order"] == [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
    ]
    schema = module.band_schema(config, grid)
    assert len(schema) == 23 * 18
    assert schema[:5] == ["s2_b2_t01", "s2_b3_t01", "s2_b4_t01", "s2_b5_t01", "s2_b6_t01"]
    assert schema[-5:] == ["s1_vv_t23", "s1_vh_t23", "s1_rvi_t23", "s1_valid_t23", "s1_count_t23"]
    assert len(schema) == len(set(schema))


def test_export_metadata_excludes_raw_coordinates_from_output():
    module = load_submitter()
    assert "longitude" not in module.EXPORT_PROPERTIES
    assert "latitude" not in module.EXPORT_PROPERTIES
    assert "spatial_block" in module.EXPORT_PROPERTIES
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["presto_input"]["coordinate_policy"].endswith("never_downstream")


def test_sampling_grid_matches_frozen_target_grid():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["earth_engine"]
    assert config["project"] == "eng-artifact-503507-k7"
    assert config["sample_scale_m"] == 20
    assert config["sample_crs"] == "EPSG:32651"
    assert config["sample_crs_transform"] == [20, 0, 362320, 0, -20, 3510320]


def test_chunk_plan_is_deterministic_and_complete():
    module = load_submitter()
    first = module.chunk_plan(13429, 500)
    second = module.chunk_plan(13429, 500)
    assert first == second
    assert len(first) == 27
    assert first[0]["start"] == 0 and first[0]["stop"] == 500
    assert first[-1]["start"] == 13000 and first[-1]["stop"] == 13429
    assert sum(item["rows"] for item in first) == 13429
    frozen = json.loads(PLAN.read_text(encoding="utf-8"))
    assert frozen["status"] == "completed_and_validated"
    assert frozen["task_ids"] == "results/manifests/eofm_ee_submission_freeze.json"
    assert [item["rows"] for item in frozen["chunks"]] == [item["rows"] for item in first]
    assert [item["output_prefix"] for item in frozen["chunks"]] == [item["description"] for item in first]
    submission = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    assert submission["summary"] == {
        "effective_chunks": 27,
        "completed": 27,
        "failed": 0,
        "active": 0,
        "completed_rows": 13429,
        "planned_rows": 13429,
        "status": "completed_and_downloaded",
    }
    assert len(submission["initial_tasks"]) == 27
    assert len(submission["retry_tasks"]) == 27
    assert all(item["state"] == "COMPLETED" for item in submission["retry_tasks"])


def test_presto_preparation_and_smoke_are_frozen():
    prepare = load_prepare()
    columns = prepare.expected_columns(
        ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        [f"t{i:02d}" for i in range(1, 24)],
    )
    assert len(columns) == 416
    validation = json.loads(INPUT_VALIDATION.read_text(encoding="utf-8"))
    assert validation["status"] == "validated_ready_for_presto_constructor"
    assert validation["arrays"]["s1_shape"] == [13429, 23, 2]
    assert validation["arrays"]["s2_shape"] == [13429, 23, 10]
    assert len(validation["source"]["chunks"]) == 27
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    assert smoke["status"] == "passed"
    assert smoke["presto"]["commit"] == "11e207a668a34336ced1d8e492a1bd5849b96c4a"
    assert smoke["result"]["embedding_shape"] == [100, 128]
    assert smoke["result"]["repeat_bitwise_equal"] is True
