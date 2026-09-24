"""Small domain-adaptation and selective-prediction primitives.

Fit functions accept permitted training features only. No function constructing
a risk score accepts target-evaluation labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def covariance_power(cov, power):
    values, vectors = np.linalg.eigh(cov)
    floor = max(float(values.max()) * 1e-8, 1e-8)
    return (vectors * np.maximum(values, floor) ** power) @ vectors.T


def coral_source(source, pool):
    """Source whitening / target-pool coloring in source-standardized space.

    Returns transformed source and the source-fitted scaler. New target features
    are only standardized with that scaler. Shrinkage stabilizes rank deficiency.
    """
    scaler = StandardScaler().fit(source)
    zs, zt = scaler.transform(source), scaler.transform(pool)
    cs, ct = LedoitWolf().fit(zs).covariance_, LedoitWolf().fit(zt).covariance_
    matrix = covariance_power(cs, -.5) @ covariance_power(ct, .5)
    return (zs - zs.mean(0)) @ matrix + zt.mean(0), scaler


def importance_weights(source, pool, source_blocks, pool_blocks, cfg):
    """Out-of-fold target/source density odds; domain labels are not rice labels.

    Correction n_source/n_target is computed separately in each training fold.
    Groups prevent a block appearing on both sides of a domain-model fold.
    """
    x = np.concatenate([source, pool])
    domain = np.r_[np.zeros(len(source)), np.ones(len(pool))]
    groups = np.r_[np.char.add('s:', source_blocks.astype(str)),
                   np.char.add('t:', pool_blocks.astype(str))]
    weights = np.full(len(source), np.nan)
    for train, held in GroupKFold(n_splits=cfg['folds']).split(x, domain, groups):
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=cfg['logistic_c'], max_iter=cfg['max_iter'])
        model.fit(scaler.transform(x[train]), domain[train])
        selected = held[held < len(source)]
        if len(selected):
            p = np.clip(model.predict_proba(scaler.transform(x[selected]))[:, 1], 1e-6, 1-1e-6)
            prior = (domain[train] == 0).sum() / (domain[train] == 1).sum()
            weights[selected] = p / (1-p) * prior
    if not np.isfinite(weights).all():
        raise ValueError('Missing or nonfinite out-of-fold weights')
    clipped = np.clip(weights, *cfg['clip'])
    diagnostics = dict(raw_min=float(weights.min()), raw_max=float(weights.max()),
                       clipped_fraction=float(np.mean(weights != clipped)),
                       effective_sample_size=float(clipped.sum() ** 2 / (clipped @ clipped)))
    return clipped / clipped.mean(), diagnostics


def predictive_scores(model, x):
    p = model.predict_proba(x)[:, 1]
    tree_p = np.stack([t.predict_proba(x)[:, 1] for t in model.estimators_])
    vote = (tree_p >= .5).mean(axis=0)
    return dict(entropy=-xlogy(p, p)-xlogy(1-p, 1-p),
                confidence=1-np.maximum(p, 1-p), margin=1-np.abs(2*p-1),
                vote_uncertainty=1-np.maximum(vote, 1-vote),
                tree_variance=tree_p.var(axis=0))


def selective_curve(y, probability, score, threshold=.5):
    """Expected confusion counts under label-independent random tie breaking.

    Every k=1..n is retained. A partial tied group contributes its expected
    confusion counts; risk is the exact expected risk, F1 is F1 of expected
    counts (not the expectation of a nonlinear F1 statistic). AURC is the mean
    of these n risks, i.e. a right-step empirical integral, with no invented
    zero-coverage observation.
    """
    y, probability, score = map(np.asarray, (y, probability, score))
    if len(y) == 0 or not np.isfinite(score).all():
        raise ValueError('Nonempty finite scores required')
    codes = 2*y.astype(int) + (probability >= threshold).astype(int)
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    group_cm = np.bincount(inverse*4+codes, minlength=len(counts)*4).reshape(-1, 4)
    total_before = np.r_[0, counts.cumsum()[:-1]]
    cm_before = np.vstack([np.zeros(4), group_cm.cumsum(0)[:-1]])
    k = np.arange(1, len(y)+1)
    group = np.searchsorted(counts.cumsum(), k, side='left')
    cm = cm_before[group] + group_cm[group] * ((k-total_before[group])/counts[group])[:, None]
    risk = (cm[:, 1]+cm[:, 2])/k
    denom = 2*cm[:, 3]+cm[:, 1]+cm[:, 2]
    f1 = np.divide(2*cm[:, 3], denom, out=np.zeros(len(k)), where=denom > 0)
    return pd.DataFrame(dict(n_accepted=k, coverage=k/len(y), risk=risk,
                             accuracy=1-risk, f1_expected_counts=f1))
