"""Inference-only spatial consensus and prior-guided dense retrieval probes.

Inspired by SC2-PCR/Cross-PCR, not reproductions. No transform labels are read.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from geotransformer.modules.ops import pairwise_distance


def rigid_fit(source, reference, weights):
    weights=weights.clamp(min=0);weights=weights/weights.sum().clamp(min=1e-8)
    cs=(source*weights[:,None]).sum(0);cr=(reference*weights[:,None]).sum(0)
    u,s,vh=torch.linalg.svd((source-cs).T@((reference-cr)*weights[:,None]))
    correction=torch.eye(3,device=source.device,dtype=source.dtype)
    correction[2,2]=torch.sign(torch.linalg.det(vh.T@u.T))
    rotation=vh.T@correction@u.T
    transform=torch.eye(4,device=source.device,dtype=source.dtype)
    transform[:3,:3]=rotation;transform[:3,3]=cr-rotation@cs
    return transform,s


class EarlySpatialMatching(nn.Module):
    def __init__(self,pool=1024,budget=256,sigma=.3):
        super().__init__();self.pool=pool;self.budget=budget;self.sigma=sigma
        self.ref_points=None;self.src_points=None;self.last={}

    @torch.no_grad()
    def forward(self,ref_feats,src_feats,ref_masks=None,src_masks=None):
        ri=torch.arange(len(ref_feats),device=ref_feats.device) if ref_masks is None else ref_masks.nonzero().flatten()
        si=torch.arange(len(src_feats),device=src_feats.device) if src_masks is None else src_masks.nonzero().flatten()
        if not len(ri) or not len(si):
            self.last={}
            return ri[:0],si[:0],ref_feats.new_zeros(0)
        scores=torch.exp(-pairwise_distance(ref_feats[ri],src_feats[si],normalized=True))
        scores=(scores/scores.sum(1,keepdim=True).clamp(min=1e-8))*(scores/scores.sum(0,keepdim=True).clamp(min=1e-8))
        vals,ids=scores.topk(min(3,len(si)),dim=1)
        rows=ri[:,None].expand_as(ids).reshape(-1);cols=si[ids.reshape(-1)];vals=vals.reshape(-1)
        best=vals.topk(min(self.pool,len(vals))).indices;r=rows[best];s=cols[best];weights=vals[best]
        rp=self.ref_points[r];sp=self.src_points[s]
        discrepancy=(torch.cdist(rp,rp)-torch.cdist(sp,sp)).abs()
        compatibility=(1-(discrepancy/self.sigma)**2).clamp(min=0)
        # A one-to-many duplicate cannot vote as an independent rigid anchor.
        compatibility[(r[:,None]==r[None])|(s[:,None]==s[None])]=0
        second=compatibility*(compatibility@compatibility)/max(len(r),1)
        vector=torch.ones(len(r),device=r.device)/max(len(r),1)**.5
        for _ in range(10):
            vector=second@vector;vector=vector/vector.norm().clamp(min=1e-8)
        rank=(vector/vector.max().clamp(min=1e-8)).sqrt()*(weights/weights.max().clamp(min=1e-8)).sqrt()
        if not (rank>0).any():rank=weights
        selected=rank.topk(min(self.budget,len(rank))).indices
        self.last=dict(r=r,s=s,weights=weights,compatibility=compatibility,second=second,centrality=vector,selected=selected)
        return r[selected],s[selected],weights[selected]


def alignment_score(reference,source,transform,radius=.25):
    aligned=source@transform[:3,:3].T+transform[:3,3]
    dist=torch.cdist(reference,aligned)
    a=torch.exp(-.5*(dist.min(1).values/radius)**2).mean()
    b=torch.exp(-.5*(dist.min(0).values/radius)**2).mean()
    return float(2*a*b/(a+b).clamp(min=1e-8))


def sparse_proposals(matcher,baseline_transform,seed_budget=12):
    state=matcher.last;r=state['r'];s=state['s'];v=state['centrality'];sc2=state['second']
    reference=matcher.ref_points[r];source=matcher.src_points[s]
    proposals=[baseline_transform];seeds=[]
    for seed in v.argsort(descending=True).tolist():
        if any(float((reference[seed]-reference[x]).norm())<.3 for x in seeds):continue
        seeds.append(seed)
        row=sc2[seed];ids=row.topk(min(32,len(row))).indices;ids=ids[row[ids]>0]
        if len(ids)>=6:
            tf,sv=rigid_fit(source[ids],reference[ids],row[ids])
            # Non-collinearity is required; planar anchors remain allowed.
            if sv[1]>1e-5 and torch.isfinite(tf).all():proposals.append(tf)
        if len(seeds)==seed_budget:break
    scores=[alignment_score(matcher.ref_points,matcher.src_points,tf) for tf in proposals]
    order=np.argsort(scores)[::-1][:3]
    return [proposals[int(i)] for i in order],dict(proposals=len(proposals),best_coarse_score=max(scores))


def dense_matches(reference,source,ref_features,src_features,transform,radius=.3,neighbors=128):
    """All fine queries, prior-radius spatial candidates, bounded nearest budget.

    Bidirectional maxima use all considered pairs, not just retained row winners.
    This replaces coarse-patch admission; the pose prior is predicted, never GT.
    """
    aligned=source@transform[:3,:3].T+transform[:3,3]
    tree=cKDTree(aligned.detach().cpu().numpy())
    distances,indices=tree.query(reference.detach().cpu().numpy(),k=min(neighbors,len(source)),distance_upper_bound=radius,workers=1)
    if indices.ndim==1:indices=indices[:,None];distances=distances[:,None]
    rf=F.normalize(ref_features,dim=1);sf=F.normalize(src_features,dim=1)
    padded=torch.cat([sf,torch.zeros_like(sf[:1])]);column_best=sf.new_full((len(sf)+1,),-1e4)
    row_scores=[];row_indices=[];row_cosines=[]
    for start in range(0,len(reference),256):
        ids=torch.from_numpy(indices[start:start+256]).to(reference.device).long()
        distance=torch.from_numpy(distances[start:start+256]).to(reference.device).float()
        cosine=(rf[start:start+256,None]*padded[ids]).sum(-1)
        score=(cosine-.1*distance/radius).masked_fill(ids==len(sf),-1e4)
        column_best.scatter_reduce_(0,ids.flatten(),score.flatten(),reduce='amax',include_self=True)
        best,local=score.max(1);chosen=ids.gather(1,local[:,None]).flatten()
        row_scores.append(best);row_indices.append(chosen);row_cosines.append(cosine.gather(1,local[:,None]).flatten())
    score=torch.cat(row_scores);cols=torch.cat(row_indices);cos=torch.cat(row_cosines)
    valid=(cols<len(sf))&(score>-100)&(score>=column_best[cols]-1e-6)
    rows=valid.nonzero().flatten();cols=cols[valid];weight=((cos[valid]+1)/2).clamp(min=1e-4)
    if len(rows)>4096:
        selected=weight.topk(4096).indices;rows=rows[selected];cols=cols[selected];weight=weight[selected]
    capped=float(np.mean(np.isfinite(distances[:,-1]))) if len(source)>=neighbors else 0.
    return reference[rows],source[cols],weight,dict(matches=len(rows),neighbor_cap_fraction=capped)


@torch.no_grad()
def global_registration(output,matcher,fine_features=None):
    proposals,meta=sparse_proposals(matcher,output['estimated_transform'])
    rf,sf=(output['ref_feats_f'],output['src_feats_f']) if fine_features is None else fine_features
    best=dict(output);best_quality=alignment_score(matcher.ref_points,matcher.src_points,output['estimated_transform']);details=[];changed=False
    for prior in proposals:
        rp,sp,w,info=dense_matches(output['ref_points_f'],output['src_points_f'],rf,sf,prior)
        tf=prior
        if len(rp)>=6:
            for _ in range(3):
                tf,sv=rigid_fit(sp,rp,w)
                residual=(rp-(sp@tf[:3,:3].T+tf[:3,3])).norm(dim=1)
                keep=residual<.1
                if keep.sum()<6 or sv[1]<1e-5:break
                rp,sp,w=rp[keep],sp[keep],w[keep]
            quality=alignment_score(matcher.ref_points,matcher.src_points,tf)
            if quality>best_quality and torch.isfinite(tf).all():
                best_quality=quality;best=dict(output,ref_corr_points=rp,src_corr_points=sp,corr_scores=w,estimated_transform=tf)
                changed=True
            info['coarse_score_after_dense']=quality
        details.append(info)
    meta['dense']=details;meta['fallback']=not changed
    return best,meta
