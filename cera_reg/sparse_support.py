"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from .backbone.early_global_matching import rigid_fit


CONFIG = dict(anchors=128, seeds=12, members=32, minimum_members=6,
              pool=1024, seed_separation=.3, coarse_sigma=.3, cross_sigma=.3,
              sparse_residual=.3, sparse_refits=3, admission_margin=1.05,
              singular_value_min=1e-5, refined_union='round_robin_unique_endpoints',
              community_union_budget=384)


def unique_members(order, r, s, positive):
    chosen, rr, ss = [], set(), set()
    for j in order.tolist():
        a, b = int(r[j]), int(s[j])
        if bool(positive[j]) and a not in rr and b not in ss:
            chosen.append(j); rr.add(a); ss.add(b)
    return chosen


@torch.no_grad()
def refined_bank(matcher):
    """Fit disjoint-endpoint SC2 groups, trim by observed residual, then quota.

    Residuals refer exclusively to predicted fits. Filtering is monotonic:
    no new sparse pair is admitted during refits. Multiple communities survive
    and receive round-robin quotas, rather than one global centrality truncation.
    """
    state = matcher.last
    if not state:
        return torch.empty(0, dtype=torch.long, device=matcher.ref_points.device), []
    r, s = state['r'], state['s']
    rp, sp = matcher.ref_points[r], matcher.src_points[s]
    seeds, groups, records = [], [], []
    for seed in state['centrality'].argsort(descending=True).tolist():
        if any(float((rp[seed]-rp[x]).norm()) < CONFIG['seed_separation'] for x in seeds):
            continue
        seeds.append(seed)
        row = state['second'][seed]
        ids = unique_members(row.topk(min(CONFIG['members'], len(row))).indices, r, s, row>0)
        initial = len(ids); valid = False; final_residual = None
        for _ in range(CONFIG['sparse_refits']):
            if len(ids) < CONFIG['minimum_members']: break
            ii = torch.tensor(ids, device=rp.device, dtype=torch.long)
            tf, sv = rigid_fit(sp[ii], rp[ii], row[ii])
            if not bool(sv[1] > CONFIG['singular_value_min'] and torch.isfinite(tf).all()): break
            residual = (rp[ii]-(sp[ii]@tf[:3,:3].T+tf[:3,3])).norm(dim=1)
            keep = residual < CONFIG['sparse_residual']
            final_residual = float(residual.max())
            ids = ii[keep].tolist()
            if bool(keep.all()):
                valid = True; break
        # The final trimmed group is refit and checked, including after step3.
        if len(ids) >= CONFIG['minimum_members']:
            ii = torch.tensor(ids, device=rp.device, dtype=torch.long)
            tf, sv = rigid_fit(sp[ii], rp[ii], row[ii])
            residual = (rp[ii]-(sp[ii]@tf[:3,:3].T+tf[:3,3])).norm(dim=1)
            final_residual = float(residual.max())
            valid = bool(sv[1] > CONFIG['singular_value_min'] and torch.isfinite(tf).all()
                         and (residual < CONFIG['sparse_residual']).all())
        else: valid = False
        if valid:
            groups.append(sorted(ids, key=lambda x: float(row[x]), reverse=True))
        records.append(dict(seed=seed, initial_members=initial, members=len(ids),
                            valid=valid, member_indices=ids, max_predicted_residual=final_residual))
        if len(seeds) >= CONFIG['seeds']: break
    chosen, rr, ss = [], set(), set()
    for j in range(CONFIG['members']):
        for group in groups:
            if j >= len(group): continue
            i = group[j]; a, b = int(r[i]), int(s[i])
            if a not in rr and b not in ss:
                chosen.append(i); rr.add(a); ss.add(b)
                if len(chosen) >= CONFIG['anchors']:
                    return torch.tensor(chosen, device=rp.device, dtype=torch.long), records
    return torch.tensor(chosen, device=rp.device, dtype=torch.long), records

