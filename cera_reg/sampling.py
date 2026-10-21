"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time


def fps(points,budget):
    p=points.numpy();count=len(p)
    if count<=budget:return points
    distance=np.full(count,np.inf);ids=[];index=int(np.sum((p-p.mean(0))**2,axis=1).argmax())
    for _ in range(budget):
        ids.append(index);distance=np.minimum(distance,np.sum((p-p[index])**2,axis=1));distance[ids]=-1;index=int(distance.argmax())
    return points[torch.tensor(ids)]

