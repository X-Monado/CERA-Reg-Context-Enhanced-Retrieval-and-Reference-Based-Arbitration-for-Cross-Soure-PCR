"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.spatial import cKDTree
from .retrieval import fused, cover_indices


def regional(o,poses):
    ps=[o['ref_points_f'],o['src_points_f']];fs=[torch.nn.functional.normalize(x,dim=1).cpu().numpy() for x in fused(o)]
    points=[p.cpu().numpy() for p in ps];qi=[cover_indices(p,128).cpu().numpy() for p in ps]
    reg=[torch.cdist(p[torch.tensor(ids,device=p.device)],p[cover_indices(p,8)]).argmin(1).cpu().numpy() for p,ids in zip(ps,qi)]
    trees=[cKDTree(p) for p in points];allrows=[]
    for start in range(0,len(poses),128):
        t=poses[start:start+128];dirs=[]
        for s,d in [(1,0),(0,1)]:
            q=points[s][qi[s]];m=q[None]@t[:,:3,:3].transpose(0,2,1)+t[:,None,:3,3] if s==1 else (q[None]-t[:,None,:3,3])@t[:,:3,:3]
            dist,nn=trees[d].query(m.reshape(-1,3),workers=2);dist=dist.reshape(len(t),-1);nn=nn.reshape(len(t),-1)
            cosine=(fs[s][qi[s]][None]*fs[d][nn]).sum(-1);support=np.exp(-.5*(dist/.1)**2);rr=[]
            for k in range(8):
                mask=reg[s]==k
                if mask.any():rr.append(np.stack([(dist[:,mask]<.1).mean(1),support[:,mask].mean(1),(support[:,mask]*cosine[:,mask]).mean(1),(dist[:,mask]<.025).mean(1)],1))
                else:rr.append(np.zeros((len(t),4)))
            dirs.append(np.stack(rr,1))
        allrows.append(np.stack(dirs,1))
    a=np.concatenate(allrows) # N, direction, region, statistic
    # Order by baseline region support; same physical regions compared per pair.
    for d in range(2):a[:,d]=a[:,d,np.argsort(-a[-1,d,:,1],kind='stable')]
    return a.reshape(len(poses),-1).astype('float32')


def paired(x,r,use_regions):
    # The final row is the reference; concatenate candidate, reference, and signed difference.
    a=np.concatenate([x,r],1) if use_regions else x
    b=np.broadcast_to(a[-1:],a.shape)
    return np.concatenate([a,b,a-b],1)


def targets(y):
    # Penalize losing any accuracy tier already satisfied by the reference pose.
    utility=y['success'].astype(float)+.5*y['strict10']+.25*y['strict5']+.1*y['quality']
    harm=np.zeros(len(utility),dtype=bool)
    for m in ['success','strict10','strict5']:harm|=bool(y[m][-1])&~y[m]
    return utility-utility[-1]-2*harm,harm


def outputs(models,x,r):
    # Each branch nominates a candidate only if both the utility and risk gates pass.
    result={};scores={};risks={}
    for name,(u,h) in models.items():
        f=paired(x,r,name=='regional');s=u.predict(f);risk=h.predict_proba(f)[:,1]
        s[-1]=0.;risk[-1]=0.;j=int(s.argmax());result[name+'_raw']=j
        allowed=(s>.02)&(risk<.1);allowed[-1]=False
        result[name+'_protected']=int(np.argmax(np.where(allowed,s,-np.inf))) if allowed.any() else len(s)-1
        scores[name]=s;risks[name]=risk
    return result,scores,risks

