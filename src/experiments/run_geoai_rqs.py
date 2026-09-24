"""Additive RQ1/RQ2/RQ3 experiment runner on the frozen common-sample benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from src.data.common import sha256
from src.experiments.run_eofm_benchmark import (ROOT, Benchmark, balanced_draw,
    fit_medians, impute, metrics, write_json, log)
from src.experiments.geoai_methods import (coral_source, importance_weights,
    predictive_scores, selective_curve)


def check_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Frozen artifact changed: {path}')


def classification(y, p, threshold):
    return {**metrics(y, p, threshold), 'auprc': average_precision_score(y, p)}


def rank_metrics(outcome, scores):
    if np.unique(outcome).size != 2:
        return np.nan, np.nan
    return roc_auc_score(outcome, scores), average_precision_score(outcome, scores)


class Research:
    def __init__(self, config):
        self.config = config.resolve()
        self.cfg = yaml.safe_load(config.read_text())
        self.base = Benchmark(ROOT/self.cfg['base_config'])
        self.base.prepare()  # Reads and validates existing canonical and draw hashes.
        self.f, self.x = self.base.frame, self.base.arrays
        self.ix, self.reps = self.base.indices, self.base.reps
        self.y = self.f.class_id.to_numpy()
        self.out = ROOT/self.cfg['output_dir']
        self.private = self.out/'private'
        self.result = ROOT/self.cfg['result_dir']
        for path in [self.private, self.result]:
            path.mkdir(parents=True, exist_ok=True)
        self.audit()

    def audit(self):
        frozen = json.loads((self.base.out/'source_prediction_freeze.json').read_text())
        for name, digest in frozen['files'].items():
            check_hash(self.base.private/name, digest)
        artifacts = json.loads((ROOT/'results/eofm_benchmark_v1/artifact_manifest.json').read_text())
        for name, digest in artifacts['files'].items():
            check_hash(ROOT/name, digest)
        if set(self.f.class_id) != {0, 1}:
            raise ValueError('Expected binary rice labels')
        duplicates = self.f.duplicated(['longitude','latitude'], keep=False)
        cross = self.f.groupby(['longitude','latitude']).split.nunique().gt(1).sum()
        feature_audit = {}
        for rep, x in self.x.items():
            if x.shape[0] != len(self.f) or np.isinf(x).any():
                raise ValueError('Invalid canonical feature array')
            hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in x]
            groups = pd.DataFrame({'hash': hashes, 'split': self.f.split})
            feature_audit[rep] = dict(shape=list(x.shape), missing=int(np.isnan(x).sum()),
                duplicate_feature_rows=int(groups.hash.duplicated(keep=False).sum()),
                identical_feature_groups_crossing_splits=int(groups.groupby('hash').split.nunique().gt(1).sum()))
        # All legacy draws must be identical, balanced, unique and pool-only.
        for key, rows in self.base.draws.items():
            regime, seed, budget = key.split('_')
            expected = balanced_draw(self.f.loc[self.ix['target_pool']], regime, int(budget), int(seed))
            if not np.array_equal(rows, expected):
                raise ValueError('Shared draw identity changed')
        expected_eval = np.concatenate([self.ix[s] for s in ['source_val','source_test','target_val']])
        for rep in self.reps:
            for seed in self.base.cfg['source_seeds']:
                with np.load(self.base.private/f'source_{rep}_{seed}.npz') as d:
                    if not np.array_equal(d['rows'], expected_eval):
                        raise ValueError('Prediction row identity changed')
        audit = dict(rows=len(self.f), split_counts=self.f.split.value_counts().to_dict(),
            split_class_counts=self.f.groupby(['split','class_id']).size().to_dict(),
            coordinate_duplicate_rows=int(duplicates.sum()), coordinate_groups_crossing_splits=int(cross),
            spatial_blocks_disjoint=True, geometry_audit='unavailable: point manifest has no polygon geometry',
            spatial_buffer='not guaranteed; adjacent blocks and shared encoder context remain possible',
            label_sources=self.f.groupby(['region','label_source']).size().to_dict(),
            representations=feature_audit, legacy_artifact_hashes_verified=len(artifacts['files']),
            legacy_source_files_verified=len(frozen['files']), shared_draws_verified=len(self.base.draws))
        for key in ['split_class_counts','label_sources']:
            audit[key] = {str(k): int(v) for k,v in audit[key].items()}
        write_json(self.result/'integrity_audit.json', audit)
        if cross:
            raise ValueError('Repeated coordinates cross splits; resolve before experiments')
        inputs = {str(p.relative_to(ROOT)): sha256(p) for p in [self.config,
            self.base.config_path, self.base.out/'protocol_freeze.json',
            self.base.private/'canonical.npz', self.base.private/'shared_draws.npz',
            ROOT/self.base.cfg['sample_manifest'], self.base.out/'source_prediction_freeze.json']}
        # Hash saved few-shot probabilities before reuse, including their private inventory.
        inventory = {p.name: sha256(p) for p in sorted((self.base.private/'fewshot').glob('*.npz'))}
        if len(inventory) != 2400:
            raise ValueError('Expected all 2,400 saved few-shot fits')
        inventory_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
        write_json(self.private/'legacy_fewshot_inventory.json', inventory)
        freeze = dict(config=self.cfg, inputs=inputs, legacy_fewshot_inventory_sha256=inventory_hash,
            target_evaluation_split='target_val', target_labels_for_fit=False,
            prior_target_results_seen=True, pre_registered=False,
            environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             sklearn=sklearn.__version__, scipy=scipy.__version__))
        path = self.result/'protocol_freeze.json'
        if path.exists() and json.loads(path.read_text()) != freeze:
            raise ValueError('Protocol/input/environment changed: use a new experiment namespace')
        if not path.exists():
            write_json(path, freeze)
        log('audit_complete', rows=len(self.f), representations=feature_audit)

    def rf(self, seed, trees=None, joint=False):
        cfg = self.base.cfg['rf']
        return RandomForestClassifier(n_estimators=trees or self.cfg['rf_trees'],
            min_samples_leaf=cfg['min_samples_leaf'], max_features=cfg['max_features'],
            class_weight=None if joint else cfg['class_weight'], random_state=seed, n_jobs=1)

    def cached_prediction(self, key, rows, producer):
        path = self.private/f'{key}.npz'
        meta = self.private/f'{key}.json'
        implementation = {name: sha256(ROOT/name) for name in
            ['src/experiments/run_geoai_rqs.py','src/experiments/geoai_methods.py']}
        provenance = dict(protocol_sha256=sha256(self.result/'protocol_freeze.json'), implementation=implementation)
        if path.exists():
            record = json.loads(meta.read_text())
            if any(record[k] != v for k,v in provenance.items()):
                raise ValueError('Cached prediction implementation changed; use new namespace')
            check_hash(path, record['sha256'])
            data = np.load(path)
            if not np.array_equal(rows, data['rows']):
                raise ValueError('Cached evaluation rows changed')
            return data['probabilities']
        p = producer()
        np.savez_compressed(path, rows=rows, probabilities=p)
        write_json(meta, {**provenance, 'sha256': sha256(path), 'saved_before_metrics': True})
        return p

    def rq1(self):
        rows = []
        threshold = self.cfg['threshold']
        for rep in self.reps:
            for seed in self.base.cfg['source_seeds']:
                d = np.load(self.base.private/f'source_{rep}_{seed}.npz')
                for split in ['source_test','target_val']:
                    keep = self.f.loc[d['rows'],'split'].eq(split).to_numpy()
                    rows.append(dict(representation=rep, method='source_only', regime='none', budget=0,
                        seed=seed, split=split, provenance='legacy_saved_prediction',
                        **classification(self.y[d['rows'][keep]], d['probabilities'][keep], threshold)))
        target = self.ix['target_val']
        for record in pd.read_csv(self.base.out/'fewshot_runs.csv').itertuples():
            p = np.load(self.base.private/'fewshot'/f'{record.key}.npz')['probabilities']
            if len(p) != len(target):
                raise ValueError('Legacy prediction length mismatch')
            values = classification(self.y[target], p, threshold)
            if not np.isclose(values['f1'], record.f1, atol=1e-12):
                raise ValueError('Legacy metric did not reproduce')
            rows.append(dict(representation=record.representation, method=record.method, regime=record.regime,
                budget=record.budget, seed=record.seed, split='target_val',
                provenance='legacy_saved_prediction', **values))
        jobs = [(rep, regime, seed, method) for rep in self.reps for regime in self.base.cfg['regimes']
                for seed in self.cfg['seeds'] for method in ['joint','target_only']]
        budget = self.cfg['additional_budget']
        draws = {f'{regime}_{seed}_{budget}': balanced_draw(self.f.loc[self.ix['target_pool']], regime, budget, seed)
                 for regime in self.base.cfg['regimes'] for seed in self.cfg['seeds']}
        draw_path = self.private/'additional_shared_draws.npz'
        if not draw_path.exists():
            np.savez_compressed(draw_path, **draws)
        else:
            saved = np.load(draw_path)
            if any(not np.array_equal(v, saved[k]) for k,v in draws.items()):
                raise ValueError('Additional shared draws changed')
        write_json(self.result/'additional_draw_freeze.json', dict(sha256=sha256(draw_path),
            draws={k: hashlib.sha256(v.astype('<i4').tobytes()).hexdigest() for k,v in draws.items()}))
        def task(spec):
            rep, regime, seed, method = spec
            draw = draws[f'{regime}_{seed}_{budget}']
            source = self.ix['source_train']
            train = np.r_[source, draw] if method == 'joint' else draw
            def fit():
                x = self.x[rep]
                med = fit_medians(x[train])
                weights = np.r_[np.ones(len(source)), np.full(len(draw),len(source)/len(draw))] if method == 'joint' else None
                model = self.rf(seed,self.cfg['fewshot_trees'],joint=method=='joint')
                model.fit(impute(x[train],med),self.y[train],sample_weight=weights)
                return model.predict_proba(impute(x[target],med))[:,1]
            p = self.cached_prediction(f'fewshot_{rep}_{regime}_{method}_{budget}_{seed}',target,fit)
            return dict(representation=rep, method=method, regime=regime, budget=budget, seed=seed,
                        split='target_val', provenance='new_250_label_fit', **classification(self.y[target],p,threshold))
        with ThreadPoolExecutor(max_workers=self.cfg['parallel_fits']) as pool:
            for i, result in enumerate(pool.map(task,jobs),1):
                rows.append(result)
                if i % 10 == 0:
                    log('rq1_new_fits', complete=i, total=len(jobs))
        pd.DataFrame(rows).sort_values(['representation','method','regime','budget','seed','split']).to_csv(self.result/'rq1_runs.csv',index=False)
        log('rq1_complete', rows=len(rows))

    def rq2(self):
        source, pool, target = [self.ix[s] for s in ['source_train','target_pool','target_val']]
        jobs = [(rep, seed) for rep in self.reps for seed in self.cfg['seeds']]
        weights, diagnostics = {}, []
        for rep in self.reps:
            x = self.x[rep]
            med = fit_medians(x[source])
            xs, xp = impute(x[source],med), impute(x[pool],med)
            weights[rep], diag = importance_weights(xs,xp,self.f.loc[source,'spatial_block'].to_numpy(),
                self.f.loc[pool,'spatial_block'].to_numpy(),self.cfg['importance'])
            diagnostics.append(dict(representation=rep,**diag))
        pd.DataFrame(diagnostics).to_csv(self.result/'importance_diagnostics.csv',index=False)
        def task(spec):
            rep, seed = spec
            x = self.x[rep]
            med = fit_medians(x[source])
            xs,xp,xt = [impute(x[ix],med) for ix in [source,pool,target]]
            if seed in self.base.cfg['source_seeds']:
                source_model = joblib.load(self.base.private/f'source_{rep}_{seed}.joblib')['model']
            else:
                source_model = self.rf(seed).fit(xs,self.y[source])
            results = []
            for method in ['source_only','coral','importance','pseudo_label','target_full']:
                details = {}
                def fit():
                    if method == 'source_only':
                        return source_model.predict_proba(xt)[:,1]
                    if method == 'coral':
                        adapted, scaler = coral_source(xs,xp)
                        model = self.rf(seed).fit(adapted,self.y[source])
                        return model.predict_proba(scaler.transform(xt))[:,1]
                    if method == 'importance':
                        model = self.rf(seed).fit(xs,self.y[source],sample_weight=weights[rep])
                        return model.predict_proba(xt)[:,1]
                    if method == 'pseudo_label':
                        prob = source_model.predict_proba(xp)[:,1]
                        confidence = np.maximum(prob,1-prob)
                        chosen = np.flatnonzero(confidence >= self.cfg['pseudo']['threshold'])
                        # Ties resolve by seeded random key, without target labels.
                        rng = np.random.default_rng(seed)
                        chosen = chosen[np.lexsort((rng.random(len(chosen)),-confidence[chosen]))][:self.cfg['pseudo']['max_samples']]
                        details['pseudo_samples'] = len(chosen)
                        if not len(chosen):
                            return source_model.predict_proba(xt)[:,1]
                        w = np.r_[np.ones(len(source)),np.full(len(chosen),len(source)/len(chosen)*self.cfg['pseudo']['total_weight_relative_to_source'])]
                        model = self.rf(seed).fit(np.r_[xs,xp[chosen]],np.r_[self.y[source],(prob[chosen]>=.5).astype(int)],sample_weight=w)
                        return model.predict_proba(xt)[:,1]
                    full_med = fit_medians(x[pool])
                    model = self.rf(seed).fit(impute(x[pool],full_med),self.y[pool])
                    return model.predict_proba(impute(x[target],full_med))[:,1]
                key = f'adapt_{rep}_{method}_{seed}'
                p = self.cached_prediction(key,target,fit)
                if details:
                    write_json(self.private/f'{key}_diagnostics.json',details)
                results.append(dict(representation=rep,method=method,regime='none',seed=seed,
                    budget=len(pool) if method=='target_full' else 0, split='target_val', trees=self.cfg['rf_trees'],
                    **classification(self.y[target],p,self.cfg['threshold'])))
            log('rq2_rep_seed_complete',representation=rep,seed=seed)
            return results
        with ThreadPoolExecutor(max_workers=self.cfg['parallel_fits']) as executor:
            results = [row for batch in executor.map(task,jobs) for row in batch]
        pd.DataFrame(results).to_csv(self.result/'rq2_runs.csv',index=False)

    def rq3(self):
        source, pool, st, target = [self.ix[s] for s in ['source_train','target_pool','source_test','target_val']]
        evaluate = np.r_[st,target]
        seed = self.cfg['ood']['source_seed']
        summaries, quantiles, curves, operations, deployment = [],[],[],[],[]
        blocks = self.f.loc[target,'spatial_block'].to_numpy()
        unique = np.unique(blocks)
        rng = np.random.default_rng(self.cfg['ood']['bootstrap_seed'])
        boots = [np.concatenate([np.flatnonzero(blocks==b) for b in rng.choice(unique,len(unique),replace=True)])
                 for _ in range(self.cfg['ood']['bootstrap_reps'])]
        for rep in self.reps:
            x = self.x[rep]
            bundle = joblib.load(self.base.private/f'source_{rep}_{seed}.joblib')
            model, med = bundle['model'],bundle['medians']
            xs,xp,xe = [impute(x[ix],med) for ix in [source,pool,evaluate]]
            scaler = StandardScaler().fit(xs)
            zs,zp,ze = [scaler.transform(a) for a in [xs,xp,xe]]
            all_x, all_z = np.r_[xe,xp],np.r_[ze,zp]
            cosine = NearestNeighbors(n_neighbors=self.cfg['ood']['k'],metric='cosine').fit(xs).kneighbors(all_x)[0]
            centers = np.stack([zs[self.y[source]==c].mean(0) for c in [0,1]])
            scores = dict(euclidean_nn=NearestNeighbors(n_neighbors=1).fit(zs).kneighbors(all_z)[0][:,0],
                cosine_nn=cosine[:,0],cosine_knn=cosine.mean(1),
                mahalanobis=np.sqrt(LedoitWolf().fit(zs).mahalanobis(all_z)),
                centroid=np.linalg.norm(all_z[:,None,:]-centers[None,:,:],axis=2).min(1),
                **predictive_scores(model,all_x))
            # All pool features; class-balanced DOMAIN weighting, no target rice labels.
            domain = self.rf(seed,self.cfg['ood']['domain_trees'])
            domain.fit(np.r_[xs,xp],np.r_[np.zeros(len(source)),np.ones(len(pool))])
            scores['domain_probability'] = domain.predict_proba(all_x)[:,1]
            if set(scores) != set(self.cfg['ood']['scores']):
                raise ValueError('Score config/implementation mismatch')
            p = model.predict_proba(xe)[:,1]
            legacy = np.load(self.base.private/f'source_{rep}_{seed}.npz')
            lookup = dict(zip(legacy['rows'],legacy['probabilities']))
            if not np.allclose(p,np.array([lookup[row] for row in evaluate]),atol=1e-12,rtol=0):
                raise ValueError('Reloaded source model no longer matches frozen probabilities')
            path = self.private/f'risk_scores_{rep}.npz'
            if path.exists():
                old = np.load(path)
                for name,values in scores.items():
                    if not np.array_equal(values,old[name]):
                        raise ValueError('Frozen scores changed')
            else:
                np.savez_compressed(path, rows=np.r_[evaluate,pool], probabilities=p, **scores)
            write_json(self.private/f'risk_scores_{rep}_freeze.json',dict(sha256=sha256(path),
                evaluation_labels_used=False, score_direction='larger_is_more_risky_fixed_a_priori'))
            pt = p[len(st):]
            y = self.y[target]
            error = (pt>=self.cfg['threshold']) != y
            for name, values in scores.items():
                se, sp = values[:len(evaluate)],values[len(evaluate):]
                risk = se[len(st):]
                d_auc,d_ap = rank_metrics(np.r_[np.zeros(len(st)),np.ones(len(target))],se)
                e_auc,e_ap = rank_metrics(error,risk)
                grouped = pd.DataFrame(dict(block=blocks,score=risk,error=error)).groupby('block').mean()
                interval = [rank_metrics(error[idx],risk[idx])[0] for idx in boots]
                valid = np.array(interval)[np.isfinite(interval)]
                curve = selective_curve(y,pt,risk,self.cfg['threshold'])
                curve['representation'],curve['score'] = rep,name
                curves.append(curve)
                row = dict(representation=rep,score=name,domain_auroc=d_auc,domain_auprc=d_ap,
                    domain_target_prevalence=len(target)/len(evaluate),error_auroc=e_auc,error_auprc=e_ap,
                    error_prevalence=error.mean(),sample_spearman=spearmanr(risk,error).statistic,
                    block_spearman=spearmanr(grouped.score,grouped.error).statistic,
                    error_auroc_ci_low=np.quantile(valid,.025) if len(valid) else np.nan,
                    error_auroc_ci_high=np.quantile(valid,.975) if len(valid) else np.nan,
                    bootstrap_valid=len(valid),aurc=curve.risk.mean())
                for coverage in self.cfg['ood']['coverages']:
                    k = max(1,int(np.ceil(coverage*len(y))))
                    op = curve.iloc[k-1].to_dict()
                    operations.append(dict(requested_coverage=coverage,**op))
                    row[f'risk_at_{round(coverage*100)}'] = op['risk']
                    cutoff = np.quantile(sp,coverage) if coverage < 1 else np.inf
                    kept = risk<=cutoff
                    deployment.append(dict(representation=rep,score=name,pool_quantile=coverage,
                        actual_coverage=kept.mean(),n=int(kept.sum()),
                        risk=float(error[kept].mean()) if kept.any() else np.nan,
                        f1=metrics(y[kept],pt[kept],self.cfg['threshold'])['f1'] if kept.any() else np.nan))
                summaries.append(row)
                # Ties remain together; tied distributions can yield fewer than ten bins.
                bins = pd.qcut(risk,self.cfg['ood']['quantiles'],labels=False,duplicates='drop')
                if np.isnan(bins).all():
                    bins = np.zeros(len(risk))
                for q in np.unique(bins):
                    keep = bins==q
                    quantiles.append(dict(representation=rep,score=name,quantile=int(q)+1,n=int(keep.sum()),
                        score_mean=risk[keep].mean(),error_rate=error[keep].mean()))
            log('rq3_rep_complete',representation=rep)
        for name,data in [('rq3_scores',summaries),('rq3_quantiles',quantiles),
                          ('rq3_operating_points',operations),('rq3_pool_thresholds',deployment)]:
            pd.DataFrame(data).to_csv(self.result/f'{name}.csv',index=False)
        pd.concat(curves,ignore_index=True).to_csv(self.result/'rq3_risk_coverage.csv',index=False)


def main(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/geoai_rqs.yaml')
    parser.add_argument('--stage',choices=['audit','rq1','rq2','rq3','all'],default=stage or 'all')
    args = parser.parse_args()
    start = time.time()
    with threadpool_limits(limits=1):
        research = Research(args.config)
        for name in ['rq1','rq2','rq3']:
            if args.stage in [name,'all']:
                getattr(research,name)()
    log('complete',stage=args.stage,seconds=round(time.time()-start,2))


if __name__=='__main__':
    main()
