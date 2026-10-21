"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time


CONFIG=dict(topk=8,sigma=.15,first_members=64,final_members=16,power_steps=10,
    min_members=6,sv2_min=1e-5,graph_chunk=256,score_chunk=128,pose_budget=24,
    pose_action_separation=.1,validation_sigma=.1,validation_radius=.3,
    extra_validation_queries_per_side=512,H_margin=1.05,V_margin=1.25,H_floor=.95)


@torch.no_grad()
def generate(o,raw):
    # Build local compatibility communities, enforce unique endpoints, then fit source-to-reference poses.
    """Every raw fine edge seeds a local graph; no top1024/early global uniqueness."""
    p=o['ref_points_f'][raw['r']];q=o['src_points_f'][raw['s']];n=len(p)
    if n<6:return p.new_zeros((0,4,4)),torch.empty((0,16),device=p.device,dtype=torch.long),dict(raw=n,valid=0)
    start=time.time()
    c=(1-((torch.cdist(p,p)-torch.cdist(q,q))/CONFIG['sigma'])**2).clamp(min=0)
    c[(raw['r'][:,None]==raw['r'][None])|(raw['s'][:,None]==raw['s'][None])]=0
    second=c*(c@c)/n
    k=min(CONFIG['first_members']-1,n-1)
    neighbors=second.topk(k,dim=1).indices
    ids=torch.cat([torch.arange(n,device=p.device)[:,None],neighbors],1)
    del second
    poses=[];members=[];rr=raw['r'].cpu().numpy();ss=raw['s'].cpu().numpy()
    seeds_valid=[];singular=[]
    for start_i in range(0,n,CONFIG['graph_chunk']):
        ii=ids[start_i:start_i+CONFIG['graph_chunk']]
        local=c[ii[:,:,None],ii[:,None,:]]
        local2=local*(local@local)/ii.shape[1]
        vec=torch.ones_like(ii,dtype=p.dtype)/ii.shape[1]**.5
        for _ in range(CONFIG['power_steps']):
            vec=(local2@vec[:,:,None]).squeeze(2);vec=vec/vec.norm(dim=1,keepdim=True).clamp(min=1e-8)
        weights=vec*local2[:,0].clamp(min=0).sqrt()
        # Seed itself is included as the last diagonal entry has zero graph vote.
        weights[:,0]=weights.max(1).values
        orders=weights.argsort(descending=True,dim=1).cpu().numpy()
        arr=ii.cpu().numpy();ww=weights.cpu().numpy();chosen=[];masks=[];wts=[]
        for j,order in enumerate(orders):
            selected=[0] if ww[j,0]>0 else [];seenr={int(rr[arr[j,0]])};seens={int(ss[arr[j,0]])}
            for pos in order:
                a,b=int(rr[arr[j,pos]]),int(ss[arr[j,pos]])
                if ww[j,pos]>0 and a not in seenr and b not in seens:
                    selected.append(int(pos));seenr.add(a);seens.add(b)
                    if len(selected)==CONFIG['final_members']:break
            chosen.append([int(arr[j,z]) for z in selected]+[0]*(16-len(selected)))
            masks.append([True]*len(selected)+[False]*(16-len(selected)))
            wts.append([float(ww[j,z]) for z in selected]+[0.]*(16-len(selected)))
        mi=torch.tensor(chosen,device=p.device);mask=torch.tensor(masks,device=p.device)
        w=p.new_tensor(wts);w=w/w.sum(1,keepdim=True).clamp(min=1e-8)
        a,b=q[mi],p[mi];ca=(a*w[:,:,None]).sum(1);cb=(b*w[:,:,None]).sum(1)
        u,sv,vh=torch.linalg.svd((a-ca[:,None]).transpose(1,2)@((b-cb[:,None])*w[:,:,None]))
        fix=torch.eye(3,device=p.device).expand(len(mi),3,3).clone()
        fix[:,2,2]=torch.sign(torch.linalg.det(vh.transpose(1,2)@u.transpose(1,2)))
        r=vh.transpose(1,2)@fix@u.transpose(1,2)
        tf=torch.eye(4,device=p.device).expand(len(mi),4,4).clone()
        tf[:,:3,:3]=r;tf[:,:3,3]=cb-(r@ca[:,:,None]).squeeze(2)
        valid=(mask.sum(1)>=6)&(sv[:,1]>CONFIG['sv2_min'])&torch.isfinite(tf).all(2).all(1)
        poses.append(tf[valid]);members.append(mi[valid]);singular.extend(sv[valid,1].cpu().tolist())
        seeds_valid.extend((torch.arange(len(ii),device=p.device)+start_i)[valid].cpu().tolist())
    return torch.cat(poses),torch.cat(members),dict(raw=n,valid=sum(len(z) for z in poses),
        svd_fits=n,seconds=time.time()-start,valid_seed_indices=seeds_valid,second_singular=singular)

