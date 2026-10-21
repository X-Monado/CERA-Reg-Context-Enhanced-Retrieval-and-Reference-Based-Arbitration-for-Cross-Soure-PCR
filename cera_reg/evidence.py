"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.spatial import cKDTree
from .retrieval import fused, cover_indices


def features(o,poses):
    """No GT: equal FPS queries, geometric overlap and local descriptor agreement."""
    clouds=[o['ref_points_f'],o['src_points_f']];feat=fused(o)
    points=[x.cpu().numpy() for x in clouds]
    fs=[torch.nn.functional.normalize(x,dim=1).cpu().numpy() for x in feat]
    ids=[cover_indices(x,64).cpu().numpy() for x in clouds];trees=[cKDTree(x) for x in points]
    ans=[]
    for start in range(0,len(poses),128):
        t=poses[start:start+128];pieces=[]
        for source,target in [(1,0),(0,1)]:
            q=points[source][ids[source]]
            moved=q[None]@t[:,:3,:3].transpose(0,2,1)+t[:,None,:3,3] if source==1 else (q[None]-t[:,None,:3,3])@t[:,:3,:3]
            dist,nn=trees[target].query(moved.reshape(-1,3),workers=2);dist=dist.reshape(len(t),-1);nn=nn.reshape(len(t),-1)
            cos=(fs[source][ids[source]][None]*fs[target][nn]).sum(-1)
            v=[(dist<r).mean(1) for r in [.025,.05,.1,.2]]
            v += [np.minimum(np.quantile(dist,p,axis=1),1.) for p in [.25,.5,.75]]
            v += [cos.mean(1),np.quantile(cos,.25,axis=1),np.mean(cos*np.exp(-.5*(dist/.1)**2),axis=1)]
            pieces.append(np.stack(v,1))
        ans.append(np.concatenate([pieces[0],pieces[1],np.minimum(*pieces),np.abs(pieces[0]-pieces[1])],1))
    return np.concatenate(ans).astype('float32')

