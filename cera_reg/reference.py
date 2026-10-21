"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from .retrieval import KEYS, fused, retrieve, cross_state, h, registration, EvidenceScore
from .matching import matcher_banks
from .backbone.early_global_matching import dense_matches, rigid_fit
from .backbone.evidence_matching import refine


@torch.no_grad()
def traced_refine(o,prior,scorer):
    rp,sp,w,info=dense_matches(o['ref_points_f'],o['src_points_f'],o['ref_feats_f'],o['src_feats_f'],prior)
    states=[dict(stage='prior',transform=prior.cpu().tolist(),components=scorer(prior))]
    if len(rp)<6:return None,None,dict(**info,states=states,invalid_step=None)
    tf=prior;last=prior;lastpairs=(rp,sp,w);invalid=None
    for step in range(3):
        tf,sv=rigid_fit(sp,rp,w)
        residual=(rp-(sp@tf[:3,:3].T+tf[:3,3])).norm(dim=1);keep=residual<.1
        finite=bool(torch.isfinite(tf).all());valid=finite and int(keep.sum())>=6 and not bool(sv[1]<1e-5)
        # Dense pairs are mutual row/column maxima; count actual unique points.
        states.append(dict(stage='update%d'%(step+1),transform=tf.cpu().tolist() if finite else None,
            input_members=len(rp),kept_members=int(keep.sum()),singular_values=sv.cpu().tolist(),
            valid=valid,mean_residual=float(residual.mean()),max_residual=float(residual.max()),
            input_unique_ref=len(torch.unique(rp,dim=0)),input_unique_src=len(torch.unique(sp,dim=0)),
            components=scorer(tf) if finite else None))
        if not valid:invalid=step+1;break
        rp,sp,w=rp[keep],sp[keep],w[keep];last=tf;lastpairs=(rp,sp,w)
    original=dict(o,estimated_transform=tf,ref_corr_points=rp,src_corr_points=sp,corr_scores=w) if torch.isfinite(tf).all() else None
    # Only difference: invalid update cannot overwrite last valid transform.
    a,b,c=lastpairs
    rollback=dict(o,estimated_transform=last,ref_corr_points=a,src_corr_points=b,corr_scores=c)
    return original,rollback,dict(**info,states=states,invalid_step=invalid)


@torch.no_grad()
def predict(o,rm,sm):
    assert set(o)==set(KEYS)
    m,_=matcher_banks(o,rm,sm);features=fused(o)
    br,_,_=registration(o,m,{'gated_base':('context_base',True,False)},
        {'base':(o['ref_feats_f'],o['src_feats_f']),'context_base':features})
    base=br['gated_base'];scorer=EvidenceScore(m,o)
    state=cross_state(o,m,retrieve(o['ref_points_f'],o['src_points_f'],*features,8))
    results=dict(M0=base,C1=base,R1_valid_rollback=base);trace=dict(states=[],invalid_step=None)
    if state['proposals']:
        j=max(range(len(state['proposals'])),key=lambda k:h(scorer(state['proposals'][k])))
        prior=state['proposals'][j];old,new,trace=traced_refine(o,prior,scorer)
        reference,_=refine(o,prior,scorer)
        assert (old is None)==(reference is None)
        if old is not None:assert float((old['estimated_transform']-reference['estimated_transform']).abs().max())<1e-6
        for arm,z in [('C1',old),('R1_valid_rollback',new)]:
            if z is not None and h(scorer(z['estimated_transform']))>1.05*h(scorer(base['estimated_transform'])):
                results[arm]=z
        trace.update(selected_sparse_index=j,coarse_and_fine_budget='M0 <=3 dense, rescue <=1; trace-reference replay is engineering cost')
    return results,trace

