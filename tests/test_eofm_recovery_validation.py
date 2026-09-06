import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/eofm/11_validate_recovery_chunks.py"
SPEC = importlib.util.spec_from_file_location("recovery_validation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_frame():
    columns = ["manifest_row", "region", "patch_row", "patch_col", "value"]
    frame = pd.DataFrame([(0, "jiangxi", r, c, 1.0) for r in range(3) for c in range(3)], columns=columns)
    return frame, columns, pd.DataFrame({"region": ["jiangxi"]})


def test_valid_patch_and_row_permutation():
    frame, columns, manifest = fixture_frame()
    result = MODULE.validate_frame(frame.iloc[::-1], columns, manifest, 0, 1, True)
    assert result["rows"] == 9
    assert result["duplicate_keys"] == 0


@pytest.mark.parametrize("damage", ["fractional_key", "region_on_nonfirst_pixel", "duplicate", "missing", "nonfinite", "schema"])
def test_rejects_corrupt_patch(damage):
    frame, columns, manifest = fixture_frame()
    if damage == "fractional_key":
        frame["patch_row"] = frame.patch_row.astype(float)
        frame.loc[1, "patch_row"] = 0.5
    elif damage == "region_on_nonfirst_pixel":
        frame.loc[1, "region"] = "shanghai"
    elif damage == "duplicate":
        frame.loc[1] = frame.loc[0]
    elif damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "nonfinite":
        frame.loc[1, "value"] = float("nan")
    elif damage == "schema":
        frame = frame[columns[::-1]]
    with pytest.raises(ValueError):
        MODULE.validate_frame(frame, columns, manifest, 0, 1, True)


def test_final_short_monthly_chunk():
    manifest = pd.DataFrame({"region": ["shanghai"] * 13429})
    frame = pd.DataFrame({"manifest_row": range(13000, 13429), "region": "shanghai", "value": 1.0})
    result = MODULE.validate_frame(frame, frame.columns.tolist(), manifest, 13000, 13429, False)
    assert result["rows"] == 429
