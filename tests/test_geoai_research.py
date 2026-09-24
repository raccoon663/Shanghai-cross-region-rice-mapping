import itertools

import numpy as np
import pytest

from src.experiments.geoai_methods import coral_source, importance_weights, selective_curve


def test_risk_curve_direction_endpoints_and_aurc():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .1, .1, .1])
    curve = selective_curve(y, p, np.arange(4))
    np.testing.assert_allclose(curve.risk, [0, 0, 1/3, .5])
    assert curve.risk.mean() == pytest.approx((1/3+.5)/4)
    assert curve.iloc[-1].coverage == 1
    assert curve.iloc[-1].accuracy == .5


def test_tied_risk_is_exact_permutation_expectation_and_order_invariant():
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .1, .9, .9])
    tied = selective_curve(y, p, np.zeros(4))
    risks = []
    for order in itertools.permutations(range(4)):
        ix = np.array(order)
        risks.append(np.cumsum((p[ix] >= .5) != y[ix])/np.arange(1, 5))
        np.testing.assert_allclose(selective_curve(y[ix], p[ix], np.zeros(4)).risk, tied.risk)
    np.testing.assert_allclose(tied.risk, np.mean(risks, axis=0))


def test_coral_is_finite_with_singular_features_and_does_not_mutate_inputs():
    source = np.array([[1., 1., 0.], [2., 2., 0.], [3., 3., 0.]])
    pool = source+4
    original = source.copy()
    adapted, scaler = coral_source(source, pool)
    assert np.isfinite(adapted).all()
    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(adapted.mean(0), scaler.transform(pool).mean(0))
    np.testing.assert_allclose(scaler.mean_, source.mean(0))


def test_importance_weights_are_finite_normalized_and_group_crossfit():
    rng = np.random.default_rng(42)
    source, pool = rng.normal(size=(40, 3)), rng.normal(.2, size=(80, 3))
    weights, diagnostics = importance_weights(source, pool, np.repeat(np.arange(8), 5),
        np.repeat(np.arange(8), 10), dict(folds=4, logistic_c=1., max_iter=500, clip=[.1, 10.]))
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert weights.mean() == pytest.approx(1)
    assert 0 < diagnostics['effective_sample_size'] <= 40


def test_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError):
        selective_curve([0, 1], [.1, .9], [0, np.nan])
