"""Resumable raw-cloud 3DCSR evaluation; predictions are saved before GT is read."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pose_errors(predicted, truth):
    import numpy as np
    rre = np.degrees(np.arccos(np.clip((np.trace(predicted[:3,:3].T @ truth[:3,:3])-1)/2, -1, 1)))
    rte = np.linalg.norm(predicted[:3,3]-truth[:3,3])
    return float(rre), float(rte)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(os.environ.get('CERA_3DCSR_ROOT', ROOT/'data/3DCSR')))
    ap.add_argument('--category', choices=['kinect_lidar', 'kinect_sfm'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--weights', type=Path, default=ROOT/'weights')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=7351)
    ap.add_argument('--limit', type=int, help='Optional smoke subset; omitted for full evaluation')
    ap.add_argument('--summarize-only', action='store_true', help='CPU-only summary of existing predictions')
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be positive')
    import numpy as np
    import torch
    from cera_reg.backbone.dataset_3dcsr import ThreeDCSRPairDataset
    from cera_reg.pipeline import CERAReg, verify_weights
    torch.set_num_threads(2)
    ds = ThreeDCSRPairDataset(str(args.root), category=args.category, split='all', use_augmentation=False)
    count = len(ds) if args.limit is None else min(args.limit, len(ds))
    args.output.mkdir(parents=True, exist_ok=True)
    weights = verify_weights(args.weights)
    protocol = dict(dataset=str(args.root.resolve()), category=args.category, seed=args.seed,
                    route='observed', weights={k:v['sha256'] for k,v in weights.items()},
                    pairs=[{k:str(Path(v).resolve()) for k,v in p.items()} for p in ds.pairs],
                    evaluation='strict RRE<15 degrees and RTE<0.30 m; successful-subset mean errors')
    pp = args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise ValueError('Output protocol differs; use a new output directory')
    if not pp.exists():
        pp.write_text(json.dumps(protocol, indent=2))
    model = None
    for i, pair in enumerate(ds.pairs[:count]):
        dest = args.output/('%03d.json' % i)
        if dest.exists() or args.summarize_only:
            continue
        ref = ds._load_point_cloud(pair['src_path'])
        src = ds._load_point_cloud(pair['tgt_path'])
        try:
            if model is None:
                model = CERAReg(args.weights, args.device)
            pose, receipt = model.register(ref, src, route='observed', seed=args.seed+i)
        except torch.cuda.OutOfMemoryError:
            (args.output/('%03d.oom.json' % i)).write_text(json.dumps(dict(index=i, status='oom', retry=True)))
            print('OOM at pair', i, '; saved predictions are reusable.', flush=True)
            break
        # Ground truth remains outside the inference call and its saved receipt.
        pending = dest.with_suffix('.tmp')
        pending.write_text(json.dumps(dict(index=i, transform=pose.tolist(), receipt=receipt), indent=2))
        pending.replace(dest)
        (args.output/('%03d.oom.json' % i)).unlink(missing_ok=True)
        print('Saved', i+1, '/', count, flush=True)
    rows = []
    for i, pair in enumerate(ds.pairs[:count]):
        dest = args.output/('%03d.json' % i)
        if not dest.exists():
            continue
        prediction = json.loads(dest.read_text())
        # Match the archived float32 GT-loading convention before error calculation.
        gt = np.loadtxt(pair['gt_path']).astype(np.float32).astype(np.float64)
        re, te = pose_errors(np.asarray(prediction['transform']), gt)
        rows.append(dict(index=i, RRE_deg=re, RTE_cm=100*te, success=re<15 and te<.30))
    good = [r for r in rows if r['success']]
    summary = dict(category=args.category, completed=len(rows), requested=count, total_pairs=len(ds),
                   complete_benchmark=len(rows)==len(ds), success_percent=100*len(good)/len(rows) if rows else None,
                   RRE_deg=float(np.mean([r['RRE_deg'] for r in good])) if good else None,
                   RTE_cm=float(np.mean([r['RTE_cm'] for r in good])) if good else None,
                   IR=None, FMR=None, correspondence_note='This pose evaluator does not export correspondences.', pairs=rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
