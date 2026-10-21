"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from .backbone.early_global_matching import EarlySpatialMatching, rigid_fit
from .backbone.evidence_matching import EvidenceScore, refine, value
from .backbone.fine_global_prior import cover_indices, fine_pool, fine_proposals, registration


CONFIG = dict(queries=512, expanded_topk=8, pool=1024, anchors=128,
              coarse_sigma=.3, cross_sigma=.3, fine_sigma=.15,
              seeds=12, fit_neighbors=32, seed_separation=.3,
              alternative_exclusion=.125, admission_margin=1.05,
              protection_support_ratio=1.25, protection_geometry_ratio=.95)


KEYS = ('ref_points_f', 'src_points_f', 'ref_points_c', 'src_points_c',
        'ref_feats_f', 'src_feats_f', 'ref_feats_c', 'src_feats_c',
        'ref_corr_points', 'src_corr_points', 'corr_scores', 'estimated_transform')


def h(components):
    return value(components, .5)


def fused(output):
    # Normalize fine features and nearest-superpoint context separately before concatenation.
    features = []
    for side in ['ref', 'src']:
        nearest = torch.cdist(output[side + '_points_f'], output[side + '_points_c']).argmin(1)
        features.append(torch.cat([F.normalize(output[side + '_feats_f'], dim=1),
                                   F.normalize(output[side + '_feats_c'][nearest], dim=1)], 1) / 2**.5)
    return tuple(features)


@torch.no_grad()
def retrieve(ref, src, rf, sf, topk):
    # Search both directions; deduplicate endpoint pairs without enforcing mutual nearest neighbors.
    """Bidirectional global retrieval, without pose-radius or label access."""
    rf, sf = F.normalize(rf, dim=1), F.normalize(sf, dim=1)
    ri, si = cover_indices(ref, CONFIG['queries']), cover_indices(src, CONFIG['queries'])
    rows, cols, weights = [], [], []
    for reverse, points, keys, qf, kf, ids in [
        (False, ref, src, rf, sf, ri), (True, src, ref, sf, rf, si)]:
        sim = qf[ids] @ kf.T
        vals, nn = sim.topk(min(topk, len(keys)), dim=1)
        distant = torch.cdist(keys[nn[:, 0]], keys) > CONFIG['alternative_exclusion']
        alt = sim.masked_fill(~distant, -torch.inf).max(1).values
        margin = torch.where(torch.isfinite(alt), (vals[:, 0] - alt).clamp(min=0) /
                             (1-alt).clamp(min=1e-3), torch.zeros_like(alt)).clamp(max=1)
        w = ((vals+1)/2).clamp(min=1e-4) * (.05+margin[:, None])
        a = ids[:, None].expand_as(nn).reshape(-1); b = nn.reshape(-1)
        rows.append(b if reverse else a); cols.append(a if reverse else b); weights.append(w.reshape(-1))
    r, s, w = torch.cat(rows), torch.cat(cols), torch.cat(weights)
    codes = r * len(src) + s
    unique, inv = torch.unique(codes, return_inverse=True)
    maximum = w.new_zeros(len(unique)); maximum.scatter_reduce_(0, inv, w, reduce='amax', include_self=True)
    return dict(r=unique//len(src), s=unique % len(src), weights=maximum)


def retain(raw, scores=None):
    rank = raw['weights'] if scores is None else scores
    ids = rank.topk(min(CONFIG['pool'], len(rank))).indices
    return {k: v[ids] for k, v in raw.items()}, ids


def fine_state(ref, src, pool):
    r, s, w = (pool[k] for k in ['r', 's', 'weights'])
    d = (torch.cdist(ref[r], ref[r]) - torch.cdist(src[s], src[s])).abs()
    c = (1-(d/CONFIG['fine_sigma'])**2).clamp(min=0)
    c[(r[:, None] == r[None]) | (s[:, None] == s[None])] = 0
    second = c * (c@c) / max(len(r), 1)
    vector = torch.ones(len(r), device=r.device)/max(len(r), 1)**.5
    for _ in range(10):
        vector = second@vector; vector = vector/vector.norm().clamp(min=1e-8)
    centrality = (vector/vector.max().clamp(min=1e-8)).sqrt() * (w/w.max().clamp(min=1e-8)).sqrt()
    return dict(pool, second=second, centrality=centrality)


def anchor_indices(matcher):
    state = matcher.last
    if not state:
        return torch.empty(0, dtype=torch.long, device=matcher.ref_points.device)
    chosen, seen_r, seen_s = [], set(), set()
    # Endpoint uniqueness prevents replicated one-to-many matches counting as
    # independent sparse support. Spatial diversity is applied to pose seeds.
    for i in state['centrality'].argsort(descending=True).tolist():
        r, s = int(state['r'][i]), int(state['s'][i])
        if r not in seen_r and s not in seen_s:
            chosen.append(i); seen_r.add(r); seen_s.add(s)
        if len(chosen) >= CONFIG['anchors']:
            break
    return torch.tensor(chosen, device=matcher.ref_points.device)


@torch.no_grad()
def cross_state(output, matcher, raw):
    ids = anchor_indices(matcher)
    if not len(ids):
        return dict(pool=retain(raw)[0], proposals=[], fit_records=[], anchor_count=0)
    state = matcher.last; ar, ass = state['r'][ids], state['s'][ids]
    rp, sp = matcher.ref_points[ar], matcher.src_points[ass]
    delta = (torch.cdist(rp, rp) - torch.cdist(sp, sp)).abs()
    k = (1-(delta/CONFIG['coarse_sigma'])**2).clamp(min=0)
    k.fill_diagonal_(0)
    aw = state['weights'][ids]; aw = aw/aw.max().clamp(min=1e-8)
    d = (torch.cdist(rp, output['ref_points_f'][raw['r']]) -
         torch.cdist(sp, output['src_points_f'][raw['s']])).abs()
    c = (1-(d/CONFIG['cross_sigma'])**2).clamp(min=0)
    # Weighted second-order sparse/dense compatibility. Unlike a pose gate,
    # the candidate pairs remain global and every sparse anchor gives a group.
    second = c * (k @ (aw[:, None]*c)) / aw.sum().clamp(min=1e-8)
    support = second.max(0).values
    rank = (support/support.max().clamp(min=1e-8)).sqrt() * (
        raw['weights']/raw['weights'].max().clamp(min=1e-8)).sqrt()
    pool, keep = retain(raw, rank); rows = second[:, keep]
    # All retained groups share a fixed 1024-pair pool. No extra GT seed fits.
    priority = (rows*pool['weights'][None]).topk(min(CONFIG['fit_neighbors'], len(keep)), 1).values.sum(1)
    proposals, records, seeds = [], [], []
    for seed in priority.argsort(descending=True).tolist():
        if any(float((rp[seed]-rp[x]).norm()) < CONFIG['seed_separation'] for x in seeds):
            continue
        seeds.append(seed); row = rows[seed]
        order = (row*pool['weights']).topk(min(CONFIG['fit_neighbors'], len(keep))).indices
        selected, seen_r, seen_s = [], set(), set()
        for j in order.tolist():
            r, s = int(pool['r'][j]), int(pool['s'][j])
            if row[j] > 0 and r not in seen_r and s not in seen_s:
                selected.append(j); seen_r.add(r); seen_s.add(s)
        valid = False
        if len(selected) >= 6:
            ii = torch.tensor(selected, device=rp.device)
            tf, sv = rigid_fit(output['src_points_f'][pool['s'][ii]], output['ref_points_f'][pool['r'][ii]],
                               row[ii]*pool['weights'][ii])
            valid = bool(sv[1] > 1e-5 and torch.isfinite(tf).all())
            if valid:
                proposals.append(tf)
        records.append(dict(anchor=int(ids[seed]), members=len(selected), valid=valid,
                            pool_member_indices=selected))
        if len(seeds) >= CONFIG['seeds']:
            break
    return dict(pool=pool, proposals=proposals, fit_records=records, anchor_count=len(ids),
                support_max=float(support.max()), support_median=float(support.median()))

