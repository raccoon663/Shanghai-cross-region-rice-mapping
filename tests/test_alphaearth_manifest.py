import pandas as pd


def test_alphaearth_manifest_has_frozen_disjoint_splits():
    frame = pd.read_csv("data_metadata/alphaearth_sample_manifest.csv")
    assert len(frame) == 13429
    assert frame.duplicated(["region", "sample_id"]).sum() == 0
    assert set(frame[frame.region == "jiangxi"].split) == {"source_train", "source_val", "source_test"}
    assert set(frame[frame.region == "shanghai"].split) == {"target_pool", "target_val"}
    assert frame.groupby(["region", "spatial_block"]).split.nunique().max() == 1
    assert frame.longitude.between(115, 123).all()
    assert frame.latitude.between(27, 33).all()
