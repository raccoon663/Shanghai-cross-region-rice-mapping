import numpy as np
import pandas as pd
import pytest

from src.experiments.run_eofm_benchmark import balanced_draw, fit_medians, impute, paired_interval, scientific_config


def test_retired_publication_metadata_does_not_relax_scientific_freeze():
    old = {'threshold': .5, 'rf': {'source_trees': 500}, 'publication': 'retired'}
    release = {'threshold': .5, 'rf': {'source_trees': 500}}
    assert scientific_config(old) == scientific_config(release)
    assert scientific_config(old) != scientific_config(dict(release, threshold=.6))
    assert scientific_config(old) != scientific_config(dict(release, rf={'source_trees': 300}))


def test_imputation_is_fit_on_training_rows_only():
    train = np.array([[1., np.nan], [3., np.nan]])
    med = fit_medians(train)
    np.testing.assert_array_equal(med, [2., 0.])
    np.testing.assert_array_equal(impute(np.array([[np.nan, 99.]]), med), [[2., 99.]])
    np.testing.assert_array_equal(impute(np.array([[np.nan, -999.]]), med), [[2., -999.]])


@pytest.mark.parametrize('regime', ['distributed', 'clustered'])
def test_shared_draw_is_repeatable_balanced_and_pool_only(regime):
    pool = pd.DataFrame({'class_id': np.repeat([0,1],20), 'spatial_block': np.tile(np.repeat(['a','b','c','d'],5),2),
                         'split': 'target_pool'}, index=np.arange(100,140))
    first = balanced_draw(pool, regime, 20, 42)
    np.testing.assert_array_equal(first, balanced_draw(pool, regime, 20, 42))
    assert len(set(first)) == 20
    assert pool.loc[first].class_id.sum() == 10
    broken = pool.assign(split='target_val')
    with pytest.raises(ValueError, match='evaluation'):
        balanced_draw(broken, regime, 20, 42)


def test_paired_bootstrap_preserves_pairing_and_direction():
    y = np.array([0,1,0,1,0,1,0,1])
    blocks = np.repeat(np.arange(4),2)
    good = np.tile(y.astype(float),(3,1))
    same = paired_interval(y,good,good,blocks,100,42)
    assert same['delta_f1'] == same['ci95_low'] == same['ci95_high'] == 0
    better = paired_interval(y,1-good,good,blocks,100,42)
    assert better['delta_f1'] == better['ci95_low'] == better['ci95_high'] == 1
