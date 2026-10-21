"""Global fine anchors add one predicted entry to a fixed three-prior beam.

No labels/GT are read. Coarse model outputs remain fixed. This is an explicit
experimental candidate branch, not a production replacement or Cross-PCR replica.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .early_global_matching import rigid_fit
from .evidence_matching import EvidenceScore, proposals, choose_priors, refine, value
from .fine_prior_admission import admit

CONFIG=dict(query_budget=512,topk=3,pool=1024,sigma=.15,seeds=12,fit_neighbors=32,
            alternative_exclusion=.125,seed_separation=.3)


def cover_indices(points,budget):
    p=points.detach().cpu().numpy();n=len(p)
    if n<=budget:return torch.arange(n,device=points.device)
    distance=np.full(n,np.inf);ids=[];i=int(((p-p.mean(0))**2).sum(1).argmax())
    for _ in range(budget):
        ids.append(i);distance=np.minimum(distance,((p-p[i])**2).sum(1));distance[ids]=-1;i=int(distance.argmax())
    return torch.tensor(ids,device=points.device)


@torch.no_grad()
def fine_pool(ref,src,rf,sf):
    if not len(ref) or not len(src):return {}
    rf,sf=F.normalize(rf,dim=1),F.normalize(sf,dim=1)
    ri,si=cover_indices(ref,CONFIG['query_budget']),cover_indices(src,CONFIG['query_budget'])
    rows=[];cols=[];weights=[];margins=[]
    for reverse,q,keys,qf,kf,ids in [(False,ref,src,rf,sf,ri),(True,src,ref,sf,rf,si)]:
        sim=qf[ids]@kf.T;vals,nn=sim.topk(min(CONFIG['topk'],len(keys)),dim=1)
        distant=torch.cdist(keys[nn[:,0]],keys)>CONFIG['alternative_exclusion']
        alt=sim.masked_fill(~distant,-torch.inf).max(1).values
        # No distant alternative implies no evidence of distinctiveness.
        margin=torch.where(torch.isfinite(alt),(vals[:,0]-alt).clamp(min=0)/(1-alt).clamp(min=1e-3),torch.zeros_like(alt)).clamp(max=1)
        w=((vals+1)/2).clamp(min=1e-4)*(.05+margin[:,None])
        a=ids[:,None].expand_as(nn).reshape(-1);b=nn.reshape(-1)
        rows.append(b if reverse else a);cols.append(a if reverse else b);weights.append(w.reshape(-1));margins.append(margin)
    r,s,w=torch.cat(rows),torch.cat(cols),torch.cat(weights)
    code=r*len(src)+s;unique,inverse=torch.unique(code,return_inverse=True)
    best=w.new_zeros(len(unique));best.scatter_reduce_(0,inverse,w,reduce='amax',include_self=True)
    keep=best.topk(min(CONFIG['pool'],len(best))).indices
    r,s,w=unique[keep]//len(src),unique[keep]%len(src),best[keep]
    delta=(torch.cdist(ref[r],ref[r])-torch.cdist(src[s],src[s])).abs()
    c=(1-(delta/CONFIG['sigma'])**2).clamp(min=0)
    c[(r[:,None]==r[None])|(s[:,None]==s[None])]=0
    sc2=c*(c@c)/max(len(r),1);vector=torch.ones(len(r),device=r.device)/max(len(r),1)**.5
    for _ in range(10):vector=sc2@vector;vector=vector/vector.norm().clamp(min=1e-8)
    centrality=(vector/vector.max().clamp(min=1e-8)).sqrt()*(w/w.max().clamp(min=1e-8)).sqrt()
    return dict(r=r,s=s,weights=w,second=sc2,centrality=centrality,
        query_counts=[len(ri),len(si)],margin_median=[float(m.median()) for m in margins])


@torch.no_grad()
def fine_proposals(ref,src,state):
    if not state:return []
    r,s,v,sc2=(state[k] for k in ('r','s','centrality','second'));result=[];seeds=[]
    for seed in v.argsort(descending=True).tolist():
        if any(float((ref[r[seed]]-ref[r[x]]).norm())<CONFIG['seed_separation'] for x in seeds):continue
        seeds.append(seed);row=sc2[seed];order=row.topk(min(CONFIG['fit_neighbors'],len(row))).indices
        # Unique endpoints form independent rigid anchors, never duplicate padding.
        chosen=[];seen_r=set();seen_s=set()
        for i in order.tolist():
            rr,ss=int(r[i]),int(s[i])
            if row[i]>0 and rr not in seen_r and ss not in seen_s:
                chosen.append(i);seen_r.add(rr);seen_s.add(ss)
        if len(chosen)>=6:
            ids=torch.tensor(chosen,device=r.device);tf,sv=rigid_fit(src[s[ids]],ref[r[ids]],row[ids]*state['weights'][ids])
            if sv[1]>1e-5 and torch.isfinite(tf).all():result.append(tf)
        if len(seeds)==CONFIG['seeds']:break
    return result


@torch.no_grad()
def registration(output,matcher,arms,features):
    """arms: label -> (fine feature source, use global fine prior, adapted dense).

    One model forward is shared; every arm uses at most3 dense priors.
    Return trace information for POST-INFERENCE diagnostics only.
    """
    scorer=EvidenceScore(matcher,output);coarse=proposals(matcher,output['estimated_transform'])
    coarse_c=[scorer(t) for t in coarse];pools={};fine={};fine_c={}
    for label,(source,use_prior,use_dense) in arms.items():
        if use_prior and source not in pools:
            rf,sf=features[source];pool=fine_pool(output['ref_points_f'],output['src_points_f'],rf,sf)
            pools[source]=pool;fine[source]=fine_proposals(output['ref_points_f'],output['src_points_f'],pool)
            fine_c[source]=[scorer(t) for t in fine[source]]
    results={};records={};cache={}
    for label,(source,use_prior,use_dense) in arms.items():
        alpha=0. if label=='reference' else .5
        ids=choose_priors(coarse_c,label!='reference',alpha)
        chosen=[('coarse',i,coarse[i]) for i in ids]
        accepted=use_prior and bool(fine[source]) and (not label.startswith('gated') or admit(coarse_c,fine_c[source]))
        if accepted:
            nonbase=[i for i in np.argsort([value(c,alpha) for c in coarse_c])[::-1] if i!=0]
            chosen=[('coarse',0,coarse[0])]+([('coarse',int(nonbase[0]),coarse[int(nonbase[0])])] if nonbase else [])
            i=int(np.argmax([value(c,alpha) for c in fine_c[source]]));chosen.append((source,i,fine[source][i]))
        dense_source=source if use_dense else 'base';rf,sf=features[dense_source]
        dense_output=dict(output,ref_feats_f=rf,src_feats_f=sf)
        best=output;quality=value(coarse_c[0],alpha);winner='baseline';dense=[]
        for kind,i,prior in chosen:
            key=(kind,i,dense_source)
            if key not in cache:cache[key]=refine(dense_output,prior,scorer)
            candidate,info=cache[key];dense.append(dict(kind=kind,index=i,**info,
                transform=candidate['estimated_transform'].cpu().tolist() if candidate is not None else None))
            if candidate is not None and value(info['components'],alpha)>quality:
                best=candidate;quality=value(info['components'],alpha);winner=[kind,i]
        results[label]=best
        records[label]=dict(priors=[dict(kind=k,index=i,transform=t.cpu().tolist()) for k,i,t in chosen],
            dense=dense,winner=winner,coarse_transforms=[t.cpu().tolist() for t in coarse],coarse_components=coarse_c,
            fine_transforms=[t.cpu().tolist() for t in fine.get(source,[])],fine_components=fine_c.get(source,[]),
            fine_pool_count=len(pools[source]['r']) if source in pools and pools[source] else 0,score=quality,
            fine_pool_margin_median=pools[source]['margin_median'] if source in pools and pools[source] else None,
            fine_admitted=accepted)
        assert len(chosen)<=3
    return results,records,pools
