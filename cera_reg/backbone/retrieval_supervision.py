"""Train the fine/context descriptors on the same full-field keys used at inference."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .fine_global_prior import cover_indices


def run8_retrieval_config(cfg):
    """Explicit legacy checkpoint profile; never mutate the shared make_cfg object."""
    cfg = deepcopy(cfg)
    cfg.dre.propagate_density = False
    cfg.dre.robust_density = False
    cfg.structure.enabled = False
    cfg.loss.weight_retrieval_loss = 1.0
    return cfg


class FullFieldRetrievalLoss(nn.Module):
    def __init__(self, queries=128, regions=64, positive_radius=.1,
                 negative_radius=.2, margin=.05, temperature=.07, negative_topk=32):
        super().__init__()
        if not 0 < positive_radius < negative_radius:
            raise ValueError('Require 0 < positive_radius < negative_radius')
        self.queries, self.regions = queries, regions
        self.positive_radius, self.negative_radius = positive_radius, negative_radius
        self.margin, self.temperature, self.negative_topk = margin, temperature, negative_topk

    def descriptors(self, output):
        points = [output[s+'_points_f'] for s in ['ref', 'src']]
        fine = [F.normalize(output[s+'_feats_f'], dim=1) for s in ['ref', 'src']]
        context = []
        for s, p in zip(['ref', 'src'], points):
            with torch.no_grad():
                ids = torch.cdist(p, output[s+'_points_c']).argmin(1)
            context.append(F.normalize(output[s+'_feats_c'], dim=1)[ids])
        fused = [torch.cat([a, b], dim=1) / 2**.5 for a, b in zip(fine, context)]
        return points, {'fine': fine, 'context': context, 'fused': fused}

    def forward(self, output, transform):
        if min(len(output[s+'_points_f']) for s in ['ref', 'src']) == 0:
            zero = sum(output[s+'_feats_'+level].sum()*0 for s in ['ref', 'src'] for level in ['f', 'c'])
            return zero, {'supported_queries': 0, 'loss_fine': 0., 'loss_context': 0., 'loss_fused': 0.}
        points, branches = self.descriptors(output)
        zero = sum(v.sum()*0 for vv in branches.values() for v in vv)
        if min(map(len, points)) == 0:
            return zero, {'supported_queries': 0, 'loss_fine': 0., 'loss_context': 0., 'loss_fused': 0.}
        with torch.no_grad():
            query = [cover_indices(p, self.queries) for p in points]
            regions = [torch.cdist(p[q], p[cover_indices(p, self.regions)]).argmin(1)
                       for p, q in zip(points, query)]
            gt = transform.detach()
            aligned = [points[0], points[1] @ gt[:3, :3].T + gt[:3, 3]]
            dist = [torch.cdist(aligned[s][query[s]], aligned[1-s]) for s in [0, 1]]
            masks = [(d < self.positive_radius, d >= self.negative_radius) for d in dist]
        losses = {}; supported = 0
        for name, feats in branches.items():
            values = []
            for side, (pos, neg) in enumerate(masks):
                valid = pos.any(1) & neg.any(1)
                if name == 'fine':
                    supported += int(valid.sum())
                if not valid.any():
                    continue
                score = feats[side][query[side][valid]] @ feats[1-side].T
                best = score.masked_fill(~pos[valid], -torch.inf).max(1).values
                bad = score.masked_fill(~neg[valid], -torch.inf).topk(min(self.negative_topk, score.shape[1]), dim=1).values
                tau = self.temperature
                agg = tau*(torch.logsumexp(bad/tau, dim=1)-torch.log(torch.isfinite(bad).sum(1).float()))
                qloss = .5*F.softplus((bad[:, 0]-best+self.margin)/tau) + .5*F.softplus((agg-best+self.margin)/tau)
                r = regions[side][valid]; weights = 1/torch.bincount(r, minlength=self.regions)[r].float()
                values.append((qloss*weights).sum()/weights.sum())
            losses[name] = torch.stack(values).mean() if values else zero
        # Fine/context are supervised separately, rather than allowing one to hide the other.
        loss = .25*losses['fine'] + .25*losses['context'] + .5*losses['fused']
        return loss, dict(supported_queries=supported, **{'loss_'+k: float(v.detach()) for k, v in losses.items()})
