"""Frozen four-representation benchmark; all results live in a new local namespace."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from src.data.common import feature_groups, sha256
from src.experiments.run_alphaearth_benchmark import metrics as classification_metrics
from src.experiments.run_phase2_ood import matched_by_class
from src.experiments.run_phase3_robustness import clustered_sample, distributed_sample, ece

ROOT = Path(__file__).resolve().parents[2]
SAMPLERS = {'distributed': distributed_sample, 'clustered': clustered_sample}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def log(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


def scientific_config(config):
    """Ignore the retired publication-status field, never experimental settings."""
    return {key: value for key, value in config.items() if key != 'publication'}


def fit_medians(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nan_to_num(np.nanmedian(x, axis=0), nan=0.0)


def impute(x, medians):
    return np.where(np.isnan(x), medians, x).astype(np.float32)


def metrics(y, p, threshold=.5):
    pred = (p >= threshold).astype(np.int8)
    values = classification_metrics(y, pred)
    values.update(error_rate=float(np.mean(pred != y)), ece=ece(y, p),
                  brier=float(brier_score_loss(y, p)),
                  auroc=float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float('nan'))
    return values


def validate_splits(frame):
    counts = {'source_train': 850, 'source_val': 265, 'source_test': 314,
              'target_pool': 9249, 'target_val': 2751}
    if frame.split.value_counts().to_dict() != counts:
        raise ValueError('Frozen split membership/counts changed')
    if frame.duplicated(['region', 'sample_id']).any():
        raise ValueError('Duplicate sample identity')
    if frame.groupby(['region', 'spatial_block']).split.nunique().max() != 1:
        raise ValueError('Spatial block leakage between splits')
    if not np.array_equal(frame.region.eq('jiangxi'), frame.split.str.startswith('source_')):
        raise ValueError('Source/target split identity mismatch')


def balanced_draw(pool, regime, budget, seed):
    selected = SAMPLERS[regime](pool, budget, seed)
    if len(selected) != budget or selected.index.duplicated().any():
        raise ValueError('Invalid draw membership')
    if selected.class_id.value_counts().to_dict() != {0: budget // 2, 1: budget // 2}:
        raise ValueError('Unbalanced draw')
    if not selected.split.eq('target_pool').all():
        raise ValueError('Draw contains evaluation samples')
    return selected.index.to_numpy(dtype=np.int32)


def block_counts(y, p, blocks, threshold=.5):
    """Return per-block TN, FP, FN, TP for each fixed seed."""
    _, inverse = np.unique(blocks, return_inverse=True)
    result = []
    for probability in np.atleast_2d(p):
        codes = 2 * y.astype(int) + (probability >= threshold).astype(int)
        result.append(np.bincount(inverse * 4 + codes, minlength=(inverse.max()+1)*4).reshape(-1, 4))
    return np.asarray(result)


def paired_interval(y, a, b, blocks, reps, seed):
    ca, cb = block_counts(y, a, blocks), block_counts(y, b, blocks)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(ca.shape[1], np.full(ca.shape[1], 1/ca.shape[1]), size=reps)
    def f1(counts):
        denominator = 2*counts[..., 3] + counts[..., 1] + counts[..., 2]
        return np.divide(2*counts[..., 3], denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    bootstrap_a = f1(np.einsum('rb,sbc->rsc', weights, ca)).mean(axis=1)
    bootstrap_b = f1(np.einsum('rb,sbc->rsc', weights, cb)).mean(axis=1)
    delta = f1(cb.sum(axis=1)).mean() - f1(ca.sum(axis=1)).mean()
    low, high = np.quantile(bootstrap_b-bootstrap_a, [.025, .975])
    return dict(delta_f1=float(delta), ci95_low=float(low), ci95_high=float(high),
                n_blocks=ca.shape[1], n_seeds=ca.shape[0], bootstrap_reps=reps)


class Benchmark:
    def __init__(self, config):
        self.config_path = config.resolve()
        self.cfg = yaml.safe_load(config.read_text())
        self.out = ROOT / self.cfg['output_dir']
        self.private = self.out / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        self.frame = pd.read_csv(ROOT / self.cfg['sample_manifest'], dtype={'sample_id': str})
        if sha256(ROOT / self.cfg['sample_manifest']) != self.cfg['sample_manifest_sha256']:
            raise ValueError('Frozen sample hash changed')
        validate_splits(self.frame)
        self.indices = {name: np.flatnonzero(self.frame.split.eq(name)) for name in self.frame.split.unique()}
        self.reps = self.cfg['representations']

    def prepare(self):
        frozen_path = self.out / 'protocol_freeze.json'
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            if (frozen['config_sha256'] != sha256(self.config_path)
                    and scientific_config(frozen['config']) != scientific_config(self.cfg)):
                raise ValueError('Do not change the frozen protocol after results exist')
            if sha256(self.private / 'canonical.npz') != frozen['canonical_sha256']:
                raise ValueError('Canonical data changed')
            if sha256(self.private / 'shared_draws.npz') != frozen['shared_draws_sha256']:
                raise ValueError('Shared draws changed')
            self.arrays = dict(np.load(self.private / 'canonical.npz'))
            self.draws = dict(np.load(self.private / 'shared_draws.npz'))
            return
        sources = {}
        chunks = sorted((ROOT / self.cfg['temporal_chunks']).glob('eofm_presto_points_2022_c*.csv'))
        original = json.loads((ROOT / 'results/manifests/eofm_presto_input_validation.json').read_text())
        expected_hashes = {item['file']: item['sha256'] for item in original['source']['chunks']}
        if {p.name for p in chunks} != set(expected_hashes):
            raise ValueError('Temporal chunk inventory changed')
        for path in chunks:
            if sha256(path) != expected_hashes[path.name]:
                raise ValueError('Temporal chunk hash changed')
        raw = pd.concat([pd.read_csv(p) for p in chunks], ignore_index=True)
        if not np.array_equal(raw.manifest_row, np.arange(len(self.frame))) or not np.array_equal(raw.region, self.frame.region):
            raise ValueError('Temporal row identity mismatch')
        columns = feature_groups(raw.columns.tolist())['s1_s2_fusion']
        self.arrays = {'temporal_reconstructed': raw[columns].replace(-9999, np.nan).to_numpy(np.float32)}
        sources['temporal_reconstructed'] = {'data_version': 'reconstructed_2022_not_legacy', 'dimensions': 92,
                                            'chunk_hashes': expected_hashes, 'feature_columns': columns}
        legacy = Path(os.environ['RICE_FUSION_LEGACY_ROOT'])
        alpha_path = legacy / self.cfg['alphaearth_relative']
        if sha256(alpha_path) != self.cfg['alphaearth_sha256']:
            raise ValueError('AlphaEarth hash changed')
        alpha = pd.read_csv(alpha_path, dtype={'sample_id': str})
        for column in ['sample_id', 'region', 'class_id', 'spatial_block', 'split']:
            if not np.array_equal(alpha[column], self.frame[column]):
                raise ValueError(f'AlphaEarth {column} mismatch')
        self.arrays['alphaearth'] = alpha[[f'A{i:02d}' for i in range(64)]].to_numpy(np.float32)
        sources['alphaearth'] = {'sha256': sha256(alpha_path), 'dimensions': 64}
        for rep, freeze, field, logical_field in [
            ('presto_primary', 'eofm_presto_primary_embedding_freeze.json', 'extraction', 'runtime_file_sha256'),
            ('galileo', 'eofm_galileo_full_embedding_freeze.json', 'embedding', 'runtime_npz_sha256')]:
            path = ROOT / self.cfg[rep]
            record = json.loads((ROOT / 'results/manifests' / freeze).read_text())
            if sha256(path) != record[field][logical_field]:
                raise ValueError(f'{rep} embedding checksum mismatch')
            with np.load(path) as data:
                if not np.array_equal(data['manifest_row'], np.arange(len(self.frame))) or not np.array_equal(data['region'], self.frame.region):
                    raise ValueError(f'{rep} identity mismatch')
                self.arrays[rep] = data['embeddings'].astype(np.float32)
            sources[rep] = {'sha256': sha256(path), 'dimensions': 128, 'freeze': freeze}
        if any(not np.isfinite(self.arrays[rep]).all() for rep in self.reps if rep != 'temporal_reconstructed'):
            raise ValueError('Non-finite embedding')
        if np.isinf(self.arrays['temporal_reconstructed']).any():
            raise ValueError('Infinite temporal feature')
        pool = self.frame.loc[self.indices['target_pool']]
        self.draws = {}
        draw_summary = []
        for regime, seed, budget in itertools.product(self.cfg['regimes'], self.cfg['few_shot_seeds'], self.cfg['budgets']):
            key = f'{regime}_{seed}_{budget}'
            rows = balanced_draw(pool, regime, budget, seed)
            self.draws[key] = rows
            draw_summary.append(dict(regime=regime, seed=seed, budget=budget, n_blocks=int(self.frame.loc[rows].spatial_block.nunique()),
                                     row_hash=hashlib.sha256(rows.astype('<i4').tobytes()).hexdigest()))
        np.savez_compressed(self.private / 'canonical.npz', **self.arrays)
        np.savez_compressed(self.private / 'shared_draws.npz', **self.draws)
        pd.DataFrame(draw_summary).to_csv(self.out / 'shared_draw_summary.csv', index=False)
        write_json(frozen_path, dict(config=self.cfg, config_sha256=sha256(self.config_path),
            created_before_target_evaluation=True, sources=sources, rows=len(self.frame),
            exclusions=0, common_sample_membership='all_13429_frozen_rows', splits=self.frame.split.value_counts().to_dict(),
            canonical_sha256=sha256(self.private / 'canonical.npz'), shared_draws_sha256=sha256(self.private / 'shared_draws.npz'),
            shared_draw_count=len(self.draws), sampler_source_sha256=sha256(ROOT / 'src/experiments/run_phase3_robustness.py'),
            threshold=.5, hyperparameter_search=False, legacy_results_preserved=True,
            comparison_scope='representation systems with different upstream inputs, not architecture-only isolation'))
        log('protocol_frozen', rows=len(self.frame), shared_draws=len(self.draws))

    def rf(self, seed, source=False, joint=False):
        cfg = self.cfg['rf']
        return RandomForestClassifier(n_estimators=cfg['source_trees' if source else 'adaptation_trees'],
            min_samples_leaf=cfg['min_samples_leaf'], max_features=cfg['max_features'],
            class_weight=None if joint else cfg['class_weight'], n_jobs=cfg['n_jobs'], random_state=seed)

    def source(self):
        freeze = self.out / 'source_prediction_freeze.json'
        if freeze.exists():
            for filename, digest in json.loads(freeze.read_text())['files'].items():
                if sha256(self.private / filename) != digest:
                    raise ValueError('Frozen source prediction/model changed')
            return
        train = self.indices['source_train']
        evaluate = np.concatenate([self.indices[x] for x in ['source_val', 'source_test', 'target_val']])
        files = {}
        for rep, seed in itertools.product(self.reps, self.cfg['source_seeds']):
            x = self.arrays[rep]
            med = fit_medians(x[train])
            model = self.rf(seed, source=True).fit(impute(x[train], med), self.frame.loc[train, 'class_id'])
            probabilities = model.predict_proba(impute(x[evaluate], med))[:, 1]
            name = f'source_{rep}_{seed}'
            np.savez_compressed(self.private / f'{name}.npz', rows=evaluate, probabilities=probabilities)
            joblib.dump(dict(model=model, medians=med), self.private / f'{name}.joblib')
            files[f'{name}.npz'] = sha256(self.private / f'{name}.npz')
            files[f'{name}.joblib'] = sha256(self.private / f'{name}.joblib')
            log('source_prediction_saved', representation=rep, seed=seed)
        write_json(freeze, dict(status='frozen_before_any_target_metric', files=files,
            training_split='source_train_only', threshold=self.cfg['threshold'],
            protocol_sha256=sha256(self.out / 'protocol_freeze.json')))

    def evaluate_source(self):
        if not (self.out / 'source_prediction_freeze.json').exists():
            raise ValueError('Predictions must be frozen first')
        rows = []
        for rep, seed in itertools.product(self.reps, self.cfg['source_seeds']):
            data = np.load(self.private / f'source_{rep}_{seed}.npz')
            for split in ['source_val', 'source_test', 'target_val']:
                mask = self.frame.loc[data['rows'], 'split'].eq(split).to_numpy()
                indices = data['rows'][mask]
                rows.append(dict(representation=rep, seed=seed, split=split,
                                 **metrics(self.frame.loc[indices, 'class_id'].to_numpy(), data['probabilities'][mask])))
        pd.DataFrame(rows).to_csv(self.out / 'source_zero_shot_runs.csv', index=False)
        log('source_evaluated', models=len(rows)//3)

    def fewshot(self):
        source, target = self.indices['source_train'], self.indices['target_val']
        y = self.frame.class_id.to_numpy()
        jobs = list(itertools.product(self.cfg['regimes'], self.cfg['few_shot_seeds'], self.cfg['budgets'], self.cfg['methods'], self.reps))
        def task(spec):
            regime, seed, budget, method, rep = spec
            key = f'{regime}_{seed}_{budget}_{method}_{rep}'
            path = self.private / 'fewshot' / f'{key}.npz'
            meta = dict(regime=regime, seed=seed, budget=budget, method=method, representation=rep, key=key)
            if path.exists():
                p = np.load(path)['probabilities']
            else:
                draw = self.draws[f'{regime}_{seed}_{budget}']
                train = np.r_[source, draw] if method == 'joint' else draw
                x = self.arrays[rep]
                med = fit_medians(x[train])
                weights = np.r_[np.ones(len(source)), np.full(len(draw), len(source)/len(draw))] if method == 'joint' else None
                model = self.rf(seed, joint=method == 'joint')
                model.fit(impute(x[train], med), y[train], sample_weight=weights)
                p = model.predict_proba(impute(x[target], med))[:, 1]
                np.savez_compressed(path, probabilities=p)
            return {**meta, **metrics(y[target], p)}
        (self.private / 'fewshot').mkdir(exist_ok=True)
        result = []
        with ThreadPoolExecutor(max_workers=self.cfg['parallel_fits']) as executor:
            for future in as_completed([executor.submit(task, spec) for spec in jobs]):
                result.append(future.result())
                if len(result) % 100 == 0:
                    log('fewshot_progress', completed=len(result), total=len(jobs))
        frame = pd.DataFrame(result).sort_values(['regime', 'method', 'budget', 'representation', 'seed'])
        frame.to_csv(self.out / 'fewshot_runs.csv', index=False)
        summary = frame.groupby(['regime','method','budget','representation']).agg(
            n_seeds=('seed','nunique'), f1_mean=('f1','mean'), f1_sd=('f1','std'),
            precision_mean=('precision','mean'), recall_mean=('recall','mean'), ece_mean=('ece','mean'), brier_mean=('brier','mean')).reset_index()
        summary.to_csv(self.out / 'fewshot_summary.csv', index=False)

    def ood(self):
        source, target = self.indices['source_train'], self.indices['target_val']
        pool = self.frame.loc[self.indices['target_pool']]
        test_source = self.indices['source_test']
        y = self.frame.loc[target,'class_id'].to_numpy()
        blocks = self.frame.loc[target,'spatial_block'].to_numpy()
        score_rows, deciles, risk_rows, domain_rows = [], [], [], []
        rng = np.random.default_rng(self.cfg['ood']['bootstrap_seed'])
        unique = np.unique(blocks)
        boot_indices = [np.concatenate([np.flatnonzero(blocks == b) for b in rng.choice(unique, len(unique), replace=True)])
                        for _ in range(self.cfg['ood']['bootstrap_reps'])]
        seed = self.cfg['ood']['source_seed']
        for rep in self.reps:
            x = self.arrays[rep]
            med = fit_medians(x[source])
            xs, xt = impute(x[source],med), impute(x[target],med)
            scaler = StandardScaler().fit(xs)
            zs, zt = scaler.transform(xs), scaler.transform(xt)
            cosine = NearestNeighbors(n_neighbors=10,metric='cosine').fit(xs).kneighbors(xt)[0]
            scores = dict(nn_standardized=NearestNeighbors(n_neighbors=1).fit(zs).kneighbors(zt)[0][:,0],
                          nn_cosine=cosine[:,0], knn10_cosine=cosine.mean(axis=1),
                          mahalanobis=np.sqrt(LedoitWolf().fit(zs).mahalanobis(zt)))
            domain_target = matched_by_class(pool, self.frame.loc[source].class_id.value_counts().to_dict(), seed)
            # matched_by_class resets row indices; join original identity rather than using reset indices.
            row_lookup = self.frame.reset_index().set_index(['region','sample_id'])['index']
            td = np.array([row_lookup.loc[(r.region,r.sample_id)] for r in domain_target.itertuples()])
            domain_x = impute(x[np.r_[source,td]],med)
            domain_y = np.r_[np.zeros(len(source)),np.ones(len(td))]
            model = self.rf(seed)
            model.fit(domain_x,domain_y)
            scores['domain_probability'] = model.predict_proba(xt)[:,1]
            matched_test = matched_by_class(self.frame.loc[target], self.frame.loc[test_source].class_id.value_counts().to_dict(), seed+100)
            tt = np.array([row_lookup.loc[(r.region,r.sample_id)] for r in matched_test.itertuples()])
            dx = impute(x[np.r_[test_source,tt]],med)
            dy = np.r_[np.zeros(len(test_source)),np.ones(len(tt))]
            true_auc = roc_auc_score(dy,model.predict_proba(dx)[:,1])
            model.fit(domain_x,np.random.default_rng(seed+900).permutation(domain_y))
            perm_auc = roc_auc_score(dy,model.predict_proba(dx)[:,1])
            domain_rows.append(dict(representation=rep, true_domain_auroc=true_auc, permuted_train_label_auroc=perm_auc))
            pred = np.load(self.private / f'source_{rep}_{seed}.npz')
            p = pred['probabilities'][self.frame.loc[pred['rows'],'split'].eq('target_val').to_numpy()]
            error = (p >= self.cfg['threshold']) != y
            np.savez_compressed(self.private / f'ood_{rep}.npz', probabilities=p, **scores)
            for name, values in scores.items():
                estimates = [roc_auc_score(error[idx],values[idx]) for idx in boot_indices if np.unique(error[idx]).size==2]
                grouped = pd.DataFrame(dict(block=blocks,score=values,error=error)).groupby('block').mean()
                score_rows.append(dict(representation=rep, score=name, error_auroc=roc_auc_score(error,values),
                    error_auroc_ci_low=np.quantile(estimates,.025), error_auroc_ci_high=np.quantile(estimates,.975),
                    error_auprc=average_precision_score(error,values), error_prevalence=error.mean(),
                    block_spearman=spearmanr(grouped.score,grouped.error).statistic))
                bins = pd.qcut(values,10,labels=False,duplicates='drop')
                for decile in np.unique(bins):
                    m = bins == decile
                    deciles.append(dict(representation=rep,score=name,decile=int(decile)+1,n=int(m.sum()),error_rate=float(error[m].mean())))
            distrust = {'confidence': 1-np.abs(2*p-1), 'knn10_cosine': scores['knn10_cosine']}
            for split_seed in self.cfg['few_shot_seeds']:
                shuffled = unique.copy()
                np.random.default_rng(split_seed).shuffle(shuffled)
                tune = np.isin(blocks,shuffled[:len(shuffled)//2])
                test = ~tune
                for score, values in distrust.items():
                    for cov in self.cfg['ood']['coverages']:
                        threshold = np.quantile(values[tune],cov)
                        kept = test & (values <= threshold)
                        risk_rows.append(dict(representation=rep,seed=split_seed,score=score,target_coverage=cov,
                            actual_coverage=kept.sum()/test.sum(), threshold=float(threshold),
                            held_block_full_error=float(error[test].mean()), n=int(kept.sum()),
                            error_rate=float(error[kept].mean()) if kept.any() else np.nan))
            log('ood_complete', representation=rep)
        pd.DataFrame(score_rows).to_csv(self.out / 'ood_summary.csv',index=False)
        pd.DataFrame(deciles).to_csv(self.out / 'ood_deciles.csv',index=False)
        pd.DataFrame(risk_rows).to_csv(self.out / 'risk_coverage.csv',index=False)
        pd.DataFrame(domain_rows).to_csv(self.out / 'domain_audit.csv',index=False)

    def paired(self):
        target = self.indices['target_val']
        y, blocks = self.frame.loc[target,'class_id'].to_numpy(), self.frame.loc[target,'spatial_block'].to_numpy()
        rows = []
        settings = [('zero_shot','source_only',0)] + list(itertools.product(self.cfg['regimes'],self.cfg['methods'],self.cfg['budgets']))
        for regime, method, budget in settings:
            probabilities = {}
            seeds = self.cfg['source_seeds'] if budget==0 else self.cfg['few_shot_seeds']
            for rep in self.reps:
                values=[]
                for seed in seeds:
                    if budget==0:
                        data=np.load(self.private / f'source_{rep}_{seed}.npz')
                        values.append(data['probabilities'][self.frame.loc[data['rows'],'split'].eq('target_val').to_numpy()])
                    else:
                        values.append(np.load(self.private / 'fewshot' / f'{regime}_{seed}_{budget}_{method}_{rep}.npz')['probabilities'])
                probabilities[rep]=np.stack(values)
            for a,b in itertools.combinations(self.reps,2):
                values=paired_interval(y,probabilities[a],probabilities[b],blocks,
                    self.cfg['paired_bootstrap_reps'],self.cfg['paired_bootstrap_seed'])
                rows.append(dict(regime=regime,method=method,budget=budget,representation_a=a,representation_b=b,
                    estimand='B_minus_A_seed_mean_F1', interval='paired_spatial_block_bootstrap_fixed_seed_set',**values))
        pd.DataFrame(rows).to_csv(self.out/'paired_f1_intervals.csv',index=False)
        log('paired_statistics_complete', comparisons=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/eofm_benchmark.yaml')
    parser.add_argument('--stage',choices=['prepare','source','fewshot','ood','paired','all'],default='all')
    args=parser.parse_args()
    experiment=Benchmark(args.config)
    started=time.time()
    with threadpool_limits(limits=1):
        experiment.prepare()
        if args.stage in ['source','all']:
            experiment.source()
            experiment.evaluate_source()
        if args.stage in ['fewshot','all']:
            if not (experiment.out/'source_prediction_freeze.json').exists():
                raise ValueError('Freeze source-only predictions before adaptation')
            experiment.fewshot()
        if args.stage in ['ood','all']:
            if not (experiment.out/'source_prediction_freeze.json').exists():
                raise ValueError('Freeze source-only predictions first')
            experiment.ood()
        if args.stage in ['paired','all']:
            experiment.paired()
    log('stage_complete',stage=args.stage,seconds=round(time.time()-started,2))


if __name__=='__main__':
    main()
