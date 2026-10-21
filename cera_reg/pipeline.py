"""Inference for the frozen manuscript model. No labels enter pose selection."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
import joblib
from .backbone.config import make_cfg
from .backbone.model import create_model, load_compatible_model_state
from .backbone.retrieval_supervision import run8_retrieval_config
from .data import seed_all, collate
from .input_protocol import bound_coarse
from .retrieval import KEYS, fused, retrieve
from .candidates import generate
from .reference import predict as reference_predict
from .evidence import features
from .regional import regional, outputs
from .selection import meta_features
from .arbitration import choose


def configuration():
    # Load the frozen manuscript profile; the legacy helper name is retained for checkpoint parity.
    return run8_retrieval_config(deepcopy(make_cfg()))


def verify_weights(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,entry in manifest.items():
        p=folder/name
        with p.open('rb') as stream:
            h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
        if h.hexdigest()!=entry['sha256']:raise ValueError(f'Weight checksum mismatch: {p}')
    return manifest


def load_scorers(folder):
    folder=Path(folder)
    models={name:joblib.load(folder/f'{name}_teacher.joblib')[name] for name in ['global','regional']}
    arbiter=joblib.load(folder/'arbiter.joblib')['mixed80global48regional']
    return models,arbiter


def select_pose(poses,x,r,models,arbiter):
    # Select from the two protected nominees and reference; no pose averaging or refinement.
    choices,s,h=outputs(models,x,r)
    ids,meta=meta_features(x,r,poses,choices,s,h)
    utility=arbiter[0].predict(meta);risk=arbiter[1].predict_proba(meta)[:,1]
    index=choose(utility,risk,ids,len(poses)-1,True)
    receipt=dict(selected=int(index),reference=len(poses)-1,bank_candidates=len(poses)-1,
                 nominees=ids.tolist(),utility=utility.tolist(),risk=risk.tolist(),
                 global_nominee=int(choices['global_protected']),regional_nominee=int(choices['regional_protected']))
    return poses[index],receipt


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value


class CERAReg:
    def __init__(self,weights=None,device='cuda',chunked_attention=True):
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. CPU scorer checks can be run with scripts/check_core.py.')
        folder=Path(weights) if weights else Path(__file__).resolve().parents[1]/'weights'
        self.manifest=verify_weights(folder);self.cfg=configuration()
        seed_all(8351)
        self.model=create_model(self.cfg).to(self.device)
        state=torch.load(folder/'encoder.pth.tar',map_location='cpu',weights_only=False)
        self.compatible_missing=load_compatible_model_state(self.model,state['model'])
        self.model.eval()
        if chunked_attention:
            from .low_memory import install
            install(self.model)
        self.models,self.arbiter=load_scorers(folder)

    @torch.no_grad()
    def predict_collated(self,raw,seed=7351):
        """Return src->ref in processing coordinates; raw contains collated points."""
        batch=move(raw,self.device)
        batch['transform']=torch.eye(4,device=self.device)
        seed_all(seed);masks={}
        hook=self.model.coarse_matching.register_forward_pre_hook(lambda m,x:masks.update(ref=x[2],src=x[3]))
        try:full=self.model(batch)
        finally:hook.remove()
        output={k:full[k] for k in KEYS}
        baseline,_=reference_predict(output,masks['ref'],masks['src'])
        bank=generate(output,retrieve(output['ref_points_f'],output['src_points_f'],*fused(output),8))[0].cpu().numpy()
        poses=np.concatenate([bank,baseline['C1']['estimated_transform'].cpu().numpy()[None]])
        pose,receipt=select_pose(poses,features(output,poses),regional(output,poses),self.models,self.arbiter)
        return pose,receipt

    def register(self,ref_points,src_points,route='observed',seed=7351,voxel_size=.025,point_limit=30000):
        """Raw metric point clouds; row-vector convention: aligned = src @ R.T + t.

        This convenience input path follows the CSR sampling protocol. For official
        3DMatch/3DLoMatch metadata evaluation, use their loader and predict_collated.
        """
        import open3d as o3d
        if route not in ('observed','native'):raise ValueError('route must be observed or native')
        if voxel_size<0 or point_limit<6:raise ValueError('Invalid sampling limits')
        clouds=[]
        for points in [ref_points,src_points]:
            p=np.asarray(points,dtype=np.float32)
            if p.ndim!=2 or p.shape[1]!=3 or len(p)<6 or not np.isfinite(p).all():
                raise ValueError('Each input must contain at least six finite XYZ points')
            clouds.append(p)
        centers=[np.zeros(3),np.zeros(3)];scale=1.
        if route=='observed':
            med=[np.median(p,axis=0) for p in clouds]
            radii=[np.quantile(np.linalg.norm(p-c,axis=1),.9) for p,c in zip(clouds,med)]
            trial=float(np.clip(2./max(max(radii),1e-6),.1,10.))
            if trial<1:scale=trial;centers=med
        seed_all(seed);item={'transform':np.eye(4,dtype=np.float32)}
        for side,p,c in zip(['ref','src'],clouds,centers):
            p=(p-c)*scale
            if voxel_size>0:
                pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(p)
                p=np.asarray(pc.voxel_down_sample(voxel_size).points,dtype=np.float32)
            if len(p)>point_limit:p=p[np.random.permutation(len(p))[:point_limit]]
            if len(p)<6:raise ValueError('Too few points after voxel sampling')
            item[side+'_points']=p.astype(np.float32);item[side+'_feats']=np.ones((len(p),1),np.float32)
        raw,guard=bound_coarse(collate(item,self.cfg))
        pose,receipt=self.predict_collated(raw,seed)
        pose=pose.copy();pose[:3,3]=pose[:3,3]/scale+centers[0]-pose[:3,:3]@centers[1]
        receipt.update(route=route,scale=scale,guard=guard,seed=seed,coordinate_convention='src @ R.T + t -> ref',half_refinement=False)
        return pose,receipt
