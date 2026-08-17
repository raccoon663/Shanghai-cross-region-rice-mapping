import pandas as pd
from pathlib import Path


def test_spatial_blocks_do_not_cross_splits():
    path = Path("data_metadata/alphaearth_sample_manifest.csv")
    frame = pd.read_csv(path)
    frame = frame.loc[frame.region == "jiangxi"].copy()
    assert frame.groupby("spatial_block").split.nunique().max() == 1
    assert set(frame.split) == {"source_train", "source_val", "source_test"}
    assert frame.sample_id.is_unique
    assert frame.groupby("split").class_id.nunique().min() == 2
