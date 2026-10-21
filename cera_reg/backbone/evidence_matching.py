"""Controlled, GT-free proposal retention and descriptor/geometry scoring.

The original early_global_matching module remains the immutable reference.
Each arm admits at most three priors before dense refinement. Shared computation
across arms is an experiment optimization, not a larger per-arm hypothesis beam.
"""
import numpy as np
import torch
from .early_global_matching import rigid_fit, dense_matches, alignment_score
from geotransformer.modules.ops import pairwise_distance

ARMS = {'reference': (False, 0.), 'retain': (True, 0.),
        'evidence25': (False, .25), 'evidence50': (False, .5),
        'retain_evidence25': (True, .25), 'retain_evidence50': (True, .5)}


class EvidenceScore:
    def __init__(self, matcher, output):
        self.ref, self.src = matcher.ref_points, matcher.src_points
        f = torch.exp(-pairwise_distance(output['ref_feats_c'], output['src_feats_c'], normalized=True))
        dual = (f / f.sum(1, keepdim=True).clamp(min=1e-8)) * (f / f.sum(0, keepdim=True).clamp(min=1e-8))
        self.row = dual / dual.max(1, keepdim=True).values.clamp(min=1e-8)
        self.col = dual / dual.max(0, keepdim=True).values.clamp(min=1e-8)

    def __call__(self, transform):
        distance = torch.cdist(self.ref, self.src @ transform[:3, :3].T + transform[:3, 3])
        geometry = torch.exp(-.5 * (distance / .25) ** 2)
        a, b = geometry.max(1).values.mean(), geometry.max(0).values.mean()
        g = 2 * a * b / (a + b).clamp(min=1e-8)
        a = (geometry * self.row).max(1).values.mean()
        b = (geometry * self.col).max(0).values.mean()
        e = 2 * a * b / (a + b).clamp(min=1e-8)
        return {'geometry': float(g), 'evidence': float(e)}


def proposals(matcher, baseline):
    state = matcher.last
    if not state:
        return [baseline]
    r, s, v, sc2 = (state[k] for k in ('r', 's', 'centrality', 'second'))
    ref, src = matcher.ref_points[r], matcher.src_points[s]
    result, seeds = [baseline], []
    for seed in v.argsort(descending=True).tolist():
        if any(float((ref[seed] - ref[x]).norm()) < .3 for x in seeds):
            continue
        seeds.append(seed)
        row = sc2[seed]
        ids = row.topk(min(32, len(row))).indices
        ids = ids[row[ids] > 0]
        if len(ids) >= 6:
            tf, sv = rigid_fit(src[ids], ref[ids], row[ids])
            if sv[1] > 1e-5 and torch.isfinite(tf).all():
                result.append(tf)
        if len(seeds) == 12:
            break
    return result


def value(components, alpha):
    return (1 - alpha) * components['geometry'] + alpha * components['evidence']


def choose_priors(components, retain, alpha):
    order = np.argsort([value(c, alpha) for c in components])[::-1].tolist()
    # Retention has exactly the same budget, without per-case or GT exceptions.
    return ([0] + [i for i in order if i != 0][:2]) if retain else order[:3]


def refine(output, prior, scorer):
    rp, sp, w, info = dense_matches(output['ref_points_f'], output['src_points_f'],
        output['ref_feats_f'], output['src_feats_f'], prior)
    if len(rp) < 6:
        return None, info
    # Deliberately reproduce reference stopping semantics; validity guards are
    # not conflated with the score/retention ablation.
    tf = prior
    for _ in range(3):
        tf, sv = rigid_fit(sp, rp, w)
        keep = (rp - (sp @ tf[:3, :3].T + tf[:3, 3])).norm(dim=1) < .1
        if keep.sum() < 6 or sv[1] < 1e-5:
            break
        rp, sp, w = rp[keep], sp[keep], w[keep]
    if not torch.isfinite(tf).all():
        return None, info
    result = dict(output, estimated_transform=tf, ref_corr_points=rp,
                  src_corr_points=sp, corr_scores=w)
    info = dict(info, components=scorer(tf))
    return result, info


@torch.no_grad()
def controlled_registration(output, matcher, arms):
    scorer = EvidenceScore(matcher, output)
    sparse = proposals(matcher, output['estimated_transform'])
    components = [scorer(tf) for tf in sparse]
    selections = {arm: choose_priors(components, *ARMS[arm]) for arm in arms}
    cache = {i: refine(output, sparse[i], scorer)
             for i in sorted({i for ids in selections.values() for i in ids})}
    results, records = {}, {}
    for arm in arms:
        _, alpha = ARMS[arm]
        best, quality, winner = output, value(components[0], alpha), 'baseline'
        for i in selections[arm]:
            result, info = cache[i]
            if result is not None and value(info['components'], alpha) > quality:
                best, quality, winner = result, value(info['components'], alpha), i
        results[arm] = best
        records[arm] = dict(prior_indices=selections[arm], winner=winner, score=quality,
            sparse_components=components, dense={str(i): cache[i][1] for i in selections[arm]},
            sparse_transforms=[tf.cpu().tolist() for tf in sparse],
            union_priors_computed=len(cache), alpha=alpha)
    return results, records
