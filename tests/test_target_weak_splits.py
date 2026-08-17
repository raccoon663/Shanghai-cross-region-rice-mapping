import pandas as pd
from pathlib import Path


def test_target_weak_blocks_are_disjoint():
    path=Path("data_metadata/alphaearth_sample_manifest.csv")
    frame=pd.read_csv(path)
    frame=frame.loc[frame.region == "shanghai"].copy()
    assert frame.sample_id.is_unique
    assert frame.groupby("spatial_block").split.nunique().max()==1
    assert set(frame.split)=={"target_pool","target_val"}
    assert set(frame.label_source)=={"official_product_weak_label"}
