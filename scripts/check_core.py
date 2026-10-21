"""CPU-only installation, geometry and optional frozen-cache selection checks."""
from pathlib import Path
import sys,os,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',action='append',default=[],help='Trusted frozen NPZ cache; can be repeated')
    ap.add_argument('--load-encoder',action='store_true',help='Construct and check encoder weights on CPU; no forward')
    ap.add_argument('--report',type=Path,default=ROOT/'provenance/cpu_check.json')
    args=ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from cera_reg.pipeline import verify_weights,load_scorers,select_pose,configuration
    from cera_reg.candidates import generate
    from cera_reg.evidence import features
    from cera_reg.regional import regional
    from cera_reg.retrieval import retrieve,fused
    from cera_reg.backbone.model import create_model,load_compatible_model_state
    verify_weights(ROOT/'weights');models,arbiter=load_scorers(ROOT/'weights')
    rng=np.random.default_rng(827)
    points=torch.tensor(rng.normal(size=(24,3)),dtype=torch.float32)
    identity=torch.eye(24)
    o={side+'_points_'+level:points.clone() for side in ['ref','src'] for level in ['f','c']}
    o.update({side+'_feats_'+level:identity.clone() for side in ['ref','src'] for level in ['f','c']})
    raw=retrieve(points,points,*fused(o),1)
    assert len(raw['r'])==24 and torch.equal(raw['r'],raw['s'])
    poses,members,info=generate(o,raw)
    assert len(poses)>0 and torch.max(torch.abs(poses-torch.eye(4)))<1e-4
    short={k:v[:5] for k,v in raw.items()}
    assert generate(o,short)[0].shape==(0,4,4)
    pose_array=np.stack([np.eye(4,dtype=np.float32)]*2)
    x=features(o,pose_array);r=regional(o,pose_array)
    assert x.shape==(2,40) and r.shape==(2,64)
    _,receipt=select_pose(pose_array,x,r,models,arbiter)
    assert receipt['selected']==1,'Identical candidates must retain reference'
    report=dict(device='cpu',gpu_forward=False,weights_verified=True,geometry_identity=True,
                degenerate_pool_fallback=True,evidence_dimensions=[40,64],identical_pose_fallback=True,cache_checks=[])
    for path in args.cache:
        with np.load(path,allow_pickle=False) as z:
            poses=z['poses'];x=z['features'];r=z['regional']
            _,receipt=select_pose(poses,x,r,models,arbiter)
            ids=np.asarray(receipt['nominees']);u=np.asarray(receipt['utility']);risk=np.asarray(receipt['risk'])
            assert np.array_equal(ids,z['ids'])
            assert np.allclose(u,z['utility'],atol=1e-10,rtol=0)
            assert np.allclose(risk,z['risk'],atol=1e-10,rtol=0)
            base=len(poses)-1;delta=z['utility']-z['utility'][np.flatnonzero(z['ids']==base)[0]]
            keep=(delta>.02)&(z['risk']<.1)&(z['ids']!=base)
            old=int(z['ids'][np.argmax(np.where(keep,delta,-np.inf))]) if keep.any() else base
            assert receipt['selected']==old
            report['cache_checks'].append(dict(cache=str(path),candidates=len(poses),selection=old,max_utility_error=float(abs(u-z['utility']).max()),max_risk_error=float(abs(risk-z['risk']).max())))
    if args.load_encoder:
        from cera_reg.data import seed_all
        seed_all(8351);cfg=configuration();model=create_model(cfg)
        cp=torch.load(ROOT/'weights/encoder.pth.tar',map_location='cpu',weights_only=False)
        missing=load_compatible_model_state(model,cp['model'])
        report['encoder']=dict(parameters=sum(p.numel() for p in model.parameters()),compatible_missing=missing,
                               use_mdb=cfg.dre.use_mdb,use_film=cfg.dre.use_film,robust_density=cfg.dre.robust_density,
                               propagate_density=cfg.dre.propagate_density,structure=cfg.structure.enabled)
    assert not torch.cuda.is_initialized(),'This check must not initialize CUDA'
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
