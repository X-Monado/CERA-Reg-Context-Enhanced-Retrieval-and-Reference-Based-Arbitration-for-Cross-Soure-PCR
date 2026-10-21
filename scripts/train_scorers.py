"""Fit frozen-encoder scorers from source-domain evidence with grouped OOF predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_training(manifest):
    import numpy as np
    from cera_reg.regional import targets
    info = json.loads(manifest.read_text())
    if info.get('dataset') != '3DMatch' or info.get('split') != 'train':
        raise ValueError('Scorer fitting requires source-domain 3DMatch training packs.')
    rows = info['observations']
    packs, groups = [], {}
    if not rows:
        raise ValueError('Empty training manifest')
    for row in rows:
        path = manifest.parent / row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Training pack checksum mismatch: ' + str(path))
        with np.load(path, allow_pickle=False) as z:
            a = dict(z)
        n = len(a['features'])
        if n < 2 or a['features'].shape != (n, 40) or a['regional'].shape != (n, 64) or a['poses'].shape != (n, 4, 4):
            raise ValueError('Expected aligned features[N,40], regional[N,64], poses[N,4,4]; reference is last')
        for name in ('success', 'strict10', 'strict5'):
            if a[name].shape != (n,) or not np.isin(a[name], [0, 1]).all():
                raise ValueError('Invalid binary accuracy labels')
            a[name] = a[name].astype(bool)
        if a['quality'].shape != (n,) or not ((a['quality'] >= 0) & (a['quality'] <= 1)).all():
            raise ValueError('Invalid continuous quality')
        if np.any(a['strict5'] & ~a['strict10']) or np.any(a['strict10'] & ~a['success']):
            raise ValueError('Accuracy labels must be nested')
        if not all(np.isfinite(a[k]).all() for k in ('features', 'regional', 'poses', 'quality')):
            raise ValueError('Nonfinite training input')
        group, fold = row['group'], int(row['fold'])
        if group in groups and groups[group] != fold:
            raise ValueError('A group appears in multiple folds')
        groups[group] = fold
        targets(a)
        packs.append(a)
    if set(groups.values()) != {0, 1, 2, 3}:
        raise ValueError('Exactly four nonempty folds are required')
    for fold in range(4):
        for name in ('global', 'regional'):
            idx = [i for i, r in enumerate(rows) if r['fold'] != fold and (name == 'global' or r['regional_train'])]
            if not idx or len(np.unique(np.concatenate([targets(packs[i])[1] for i in idx]))) != 2:
                raise ValueError('Each branch/fold must contain both harm classes')
    return info, packs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--dry-run', action='store_true', help='Validate data/folds without fitting')
    args = ap.parse_args()
    if not args.dry_run and args.output is None:
        ap.error('--output is required for fitting')
    import numpy as np
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from cera_reg.selection import train_teacher, meta_features
    from cera_reg.regional import outputs, targets
    info, packs = load_training(args.manifest)
    rows = info['observations']
    print(json.dumps(dict(observations=len(rows), regional_observations=sum(r['regional_train'] for r in rows),
                         groups=len({r['group'] for r in rows}), grouping=info['grouping'], dry_run=args.dry_run)), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    xx, yy, hh, ww, audit = [], [], [], [], []
    for fold in range(4):
        # A held-out observation is never scored by a first-stage model fitted on its group.
        fit = [i for i, r in enumerate(rows) if r['fold'] != fold]
        held = [i for i, r in enumerate(rows) if r['fold'] == fold]
        models = {'global': train_teacher([packs[i] for i in fit], 'global'),
                  'regional': train_teacher([packs[i] for i in fit if rows[i]['regional_train']], 'regional')}
        for i in held:
            a = packs[i]
            choice, scores, risks = outputs(models, a['features'], a['regional'])
            rng = np.random.default_rng(25109 + i)
            base = len(a['features']) - 1
            # Sample without quality labels; include raw/protected nominees and the reference.
            sampled = np.unique(np.r_[rng.choice(base, min(32, base), replace=False), list(choice.values()), base])
            ids, meta = meta_features(a['features'], a['regional'], a['poses'], choice, scores, risks, sampled)
            utility, harm = targets(a)
            xx.append(meta); yy.append(utility[ids]); hh.append(harm[ids]); ww.append(np.ones(len(ids))/len(ids))
            audit.append(dict(observation=i, fold=fold, group=rows[i]['group'], candidates=ids.tolist()))
        print('Completed out-of-fold predictions:', fold, flush=True)
    x, y, h, w = map(np.concatenate, (xx, yy, hh, ww)); w *= len(w)/w.sum()
    if len(np.unique(h)) != 2:
        raise ValueError('Second-stage training requires both harm classes')
    kw = dict(max_iter=40, max_leaf_nodes=7, l2_regularization=10, learning_rate=.05,
              random_state=25109, early_stopping=False)
    arbiter = (HistGradientBoostingRegressor(**kw).fit(x, y, sample_weight=w),
               HistGradientBoostingClassifier(**kw).fit(x, h, sample_weight=w))
    # Keep the serialization key expected by the adopted inference loader.
    joblib.dump({'mixed80global48regional': arbiter}, args.output/'arbiter.joblib')
    for name in ('global', 'regional'):
        selected = packs if name == 'global' else [a for a, r in zip(packs, rows) if r['regional_train']]
        joblib.dump({name: train_teacher(selected, name)}, args.output/(name+'_teacher.joblib'))
    report = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  grouping=info['grouping'], encoder_updates=0, observations=len(rows), oof=audit,
                  weights={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.joblib')})
    (args.output/'training_report.json').write_text(json.dumps(report, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
