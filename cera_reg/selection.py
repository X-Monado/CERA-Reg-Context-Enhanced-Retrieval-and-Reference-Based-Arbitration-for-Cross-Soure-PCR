"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from .regional import paired, targets


def train_teacher(packs,name):
    # Weight each observation equally, regardless of how many candidates it contains.
    xx=np.concatenate([paired(a['features'],a['regional'],name=='regional') for a in packs])
    yy=[targets(a) for a in packs];u=np.concatenate([v[0] for v in yy]);h=np.concatenate([v[1] for v in yy])
    w=np.concatenate([np.ones(len(a['features']))/len(a['features']) for a in packs]);w*=len(w)/w.sum()
    kw=dict(max_iter=100,max_leaf_nodes=15,l2_regularization=10,learning_rate=.05,random_state=24109,early_stopping=False)
    return (HistGradientBoostingRegressor(**kw).fit(xx,u,sample_weight=w),HistGradientBoostingClassifier(**kw).fit(xx,h,sample_weight=w))


def meta_features(x,r,poses,choice,s,h,training_ids=None):
    # Use candidate/reference evidence and branch disagreement; never use ground-truth errors.
    # Union of protected winners plus the baseline, without pose averaging.
    ids=np.unique([choice['global_protected'],choice['regional_protected'],len(x)-1] if training_ids is None else training_ids)
    a=r.reshape(len(r),2,8,4);summary=np.concatenate([a.mean(2).reshape(len(r),-1),a.min(2).reshape(len(r),-1)],1)
    z=np.concatenate([x,summary],1);b=np.broadcast_to(z[-1:],(len(ids),z.shape[1]))
    teacher=np.stack([s['global'][ids],s['regional'][ids],h['global'][ids],h['regional'][ids]],1)
    # Rotation-matrix and translation disagreement between the two protected nominees.
    pg,pr=poses[choice['global_protected']],poses[choice['regional_protected']]
    dis=np.array([np.linalg.norm(pg[:3,:3]-pr[:3,:3]),np.linalg.norm(pg[:3,3]-pr[:3,3])])
    constant=np.broadcast_to(dis,(len(ids),2))
    return ids,np.concatenate([z[ids],b,z[ids]-b,teacher,constant],1).astype('float32')

