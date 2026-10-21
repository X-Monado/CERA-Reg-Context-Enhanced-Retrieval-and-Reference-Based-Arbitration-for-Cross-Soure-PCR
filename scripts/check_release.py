"""Small CPU acceptance checks for release CLIs; never replaces manuscript models."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from evaluate_3dcsr import pose_errors
from train_scorers import load_training
from cera_reg.backbone.dataset_3dcsr import d1_scene_split
from cera_reg.pipeline import load_scorers,select_pose,verify_weights


def main():
    result={}
    eye=np.eye(4);m=eye.copy();m[:3,3]=[.1,.2,.2]
    assert pose_errors(eye,eye)==(0.,0.)
    assert np.allclose(pose_errors(m,eye),(0.,.3))
    angle=np.deg2rad(10);m=eye.copy();m[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    assert np.allclose(pose_errors(m,eye),(10.,0.))
    result['pose_metrics_known_transforms']=True
    info,packs=load_training(ROOT/'data/scorer_training/manifest.json')
    result['source_pack_validation']=dict(observations=len(packs),regional=sum(r['regional_train'] for r in info['observations']))
    with tempfile.TemporaryDirectory(prefix='cera_release_') as tmp:
        t=Path(tmp);rows=[];rng=np.random.default_rng(41)
        # Tiny synthetic feature packs exercise OOF fitting, not benchmark performance.
        for i in range(8):
            n=48;f=rng.normal(size=(n,40)).astype('float32');r=rng.normal(size=(n,64)).astype('float32')
            poses=np.tile(eye,(n,1,1)).astype('float32');poses[:,:3,3]=rng.normal(size=(n,3))
            success=np.arange(n)%3!=0;strict10=success&(np.arange(n)%2==0);strict5=strict10&(np.arange(n)%4==0)
            success[-1]=True;strict10[-1]=True;strict5[-1]=True
            p=t/('%03d.npz'%i);np.savez(p,features=f,regional=r,poses=poses,success=success,strict10=strict10,strict5=strict5,quality=rng.uniform(size=n))
            rows.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),group='scene%d'%(i%4),fold=i%4,regional_train=True))
        manifest=dict(dataset='3DMatch',split='train',grouping='synthetic test',observations=rows)
        mp=t/'manifest.json';mp.write_text(json.dumps(manifest))
        subprocess.run([sys.executable,str(ROOT/'scripts/train_scorers.py'),'--manifest',str(mp),'--output',str(t/'fitted')],check=True)
        models,arbiter=load_scorers(t/'fitted')
        pose,receipt=select_pose(poses,f,r,models,arbiter)
        assert pose.shape==(4,4) and np.isfinite(pose).all() and receipt['selected'] in receipt['nominees']
        result['synthetic_oof_fit_reload_and_selection']=True
        # Invalid folds must be rejected before fitting.
        manifest['observations'][4]['fold']=1;mp.write_text(json.dumps(manifest))
        try:load_training(mp)
        except ValueError:result['group_leakage_rejected']=True
        else:raise AssertionError('Cross-fold group leakage accepted')
        for name in ['scene1','scene2','scene10']:(t/'dataset/kinect_lidar'/name).mkdir(parents=True)
        assert d1_scene_split(2,str(t/'dataset'))==(['scene1','scene2'],['scene10'])
        result['portable_dataset_root']=True
        # Dummy payload tests packaging only; real weight loading is checked separately.
        (t/'encoder').write_bytes(b'test encoder payload')
        subprocess.run([sys.executable,str(ROOT/'scripts/package_weights.py'),'--scorers',str(t/'fitted'),'--encoder',str(t/'encoder'),'--output',str(t/'package')],check=True)
        assert len(verify_weights(t/'package'))==4
        result['weight_packaging_checksum']=True
    dest=ROOT/'output/release_checks_20261018';dest.mkdir(parents=True,exist_ok=True)
    (dest/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
