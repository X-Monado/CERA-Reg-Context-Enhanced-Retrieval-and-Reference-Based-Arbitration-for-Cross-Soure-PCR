"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from .sampling import fps
from geotransformer.modules.ops import radius_search


def bound_coarse(data):
    native=data['lengths'][3].tolist()
    if max(native)<=2048:return data,dict(native=native,used=native,changed=False)
    nr=int(data['lengths'][3][0]);ref=fps(data['points'][3][:nr],1536);src=fps(data['points'][3][nr:],1536)
    points=torch.cat([ref,src]);lengths=torch.tensor([len(ref),len(src)])
    result=dict(data)
    for key in ['points','lengths','neighbors','subsampling','upsampling']:result[key]=list(data[key])
    result['points'][3]=points;result['lengths'][3]=lengths
    result['neighbors'][3]=radius_search(points,points,lengths,lengths,.5,48)
    result['subsampling'][2]=radius_search(points,data['points'][2],lengths,data['lengths'][2],.25,48)
    result['upsampling'][2]=radius_search(data['points'][2],points,data['lengths'][2],lengths,.5,48)
    assert all(torch.equal(result['points'][l],data['points'][l]) for l in range(3))
    return result,dict(native=native,used=lengths.tolist(),changed=True)


def canonical_item(dataset,index):
    """Raw observed radii choose a SHARED scale; GT only changes coordinate labels."""
    pair=dataset.pairs[index];ref=dataset._load_point_cloud(pair['src_path']);src=dataset._load_point_cloud(pair['tgt_path'])
    cr=np.median(ref,axis=0);cs=np.median(src,axis=0)
    radii=[np.quantile(np.linalg.norm(p-c,axis=1),.9) for p,c in [(ref,cr),(src,cs)]]
    scale=float(np.clip(2./max(max(radii),1e-6),.1,10.))
    tf=np.loadtxt(pair['gt_path']).astype(np.float32);canonical=tf.copy()
    canonical[:3,3]=scale*(tf[:3,:3]@cs+tf[:3,3]-cr)
    item=dict(transform=canonical)
    for side,points,center in [('ref',ref,cr),('src',src,cs)]:
        p=dataset._voxel_downsample((points-center)*scale)
        p=dataset._random_sample(p)
        item[side+'_points']=p;item[side+'_feats']=np.ones((len(p),1),np.float32)
    info=dict(scale=scale,ref_center=cr.tolist(),src_center=cs.tolist(),observed_radii=radii)
    return item,info,tf


def restore_output(output,info,native_gt):
    """Restore points and predicted pose before native-unit evaluation."""
    device=output['estimated_transform'].device;scale=info['scale']
    cr=torch.tensor(info['ref_center'],device=device);cs=torch.tensor(info['src_center'],device=device)
    result=dict(output)
    for key in ['ref_points','ref_points_f','ref_points_c','ref_corr_points','ref_node_corr_knn_points']:
        if key in result:result[key]=result[key]/scale+cr
    for key in ['src_points','src_points_f','src_points_c','src_corr_points','src_node_corr_knn_points']:
        if key in result:result[key]=result[key]/scale+cs
    predicted=output['estimated_transform'].clone()
    predicted[:3,3]=predicted[:3,3]/scale+cr-predicted[:3,:3]@cs
    result['estimated_transform']=predicted
    # Canonical GT node labels are not native-radius PIR: do not expose them as native coarse accuracy.
    return result,torch.as_tensor(native_gt,device=device)

