"""Small raw-cloud replay against trusted local result archives (no training).
Run from this repository; data/archive paths are explicit and never imported as code.
"""
from pathlib import Path
import os,sys,argparse,json,time,hashlib,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--archive-root',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--memory-fraction',type=float,default=.25)
 ap.add_argument('--cases',nargs='+',default=['sfm:1','sfm:10'],help='category:index pairs; default is two short SfM cases')
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 faulthandler.dump_traceback_later(60,repeat=True)
 print('Loading dependencies',flush=True)
 import numpy as np
 import torch
 from cera_reg.pipeline import CERAReg,select_pose
 from cera_reg.backbone.dataset_3dcsr import ThreeDCSRPairDataset
 torch.set_num_threads(2)
 def dump(name,obj):(a.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
 start=time.perf_counter();print('Initializing CUDA',flush=True);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
 probe=torch.arange(64,device='cuda').square().sum();assert probe.item()==85344
 print('CUDA arithmetic passed; loading model',flush=True)
 model=CERAReg();load_seconds=time.perf_counter()-start
 # Import origin checks prevent an installed original checkout shadowing the release.
 modules={k:str(Path(m.__file__).resolve()) for k,m in sys.modules.items() if (k=='cera_reg' or k.startswith('cera_reg.') or k=='geotransformer' or k.startswith('geotransformer.')) and getattr(m,'__file__',None)}
 assert all(Path(p).is_relative_to(ROOT) for p in modules.values()) if hasattr(Path('.'),'is_relative_to') else all(p.startswith(str(ROOT)+os.sep) for p in modules.values())
 dump('environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),load_seconds=load_seconds,modules=modules,memory_fraction=a.memory_fraction))
 print('Model loaded',load_seconds,flush=True)
 rows=[];datasets={}
 cases=[(v.split(':')[0],int(v.split(':')[1])) for v in a.cases]
 assert all(c in ['sfm','lidar'] for c,i in cases)
 for category,index in cases:
  key=f'{category}_{index:04d}';archive=a.archive_root/f'{category}_observed_7351';seed=7351+index
  old=json.loads((archive/'predictions'/f'{index:04d}.json').read_text());oldmetric=json.loads((archive/'results'/f'{index:04d}.json').read_text())['metrics']['fusion']
  assert old['seed']==7351
  if category not in datasets:datasets[category]=ThreeDCSRPairDataset(str(a.data_root),category='kinect_'+category,split='all',voxel_size=.025,point_limit=30000)
  ds=datasets[category];pair=ds.pairs[index];ref=ds._load_point_cloud(pair['src_path']);src=ds._load_point_cloud(pair['tgt_path'])
  if category=='sfm' and index==10:np.savez_compressed(a.output/'cli_input.npz',ref_points=ref,src_points=src)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
  try:pose,receipt=model.register(ref,src,route='observed',seed=seed)
  except torch.cuda.OutOfMemoryError:
   dump(key+'_oom.json',dict(status='oom',seed=seed,input=pair));raise
  torch.cuda.synchronize();elapsed=time.perf_counter()-tick
  # Commit prediction before loading GT for evaluation.
  dump(key+'_prediction.json',dict(transform=pose.tolist(),receipt=receipt,source_paths=pair,seconds=elapsed))
  gt=np.loadtxt(pair['gt_path']).astype(np.float32).astype(np.float64);t=pose.astype(np.float64)
  rre=float(np.degrees(np.arccos(np.clip((np.trace(t[:3,:3].T@gt[:3,:3])-1)/2,-1,1))));rte=float(np.linalg.norm(t[:3,3]-gt[:3,3]))
  delta=float(np.max(abs(t-np.array(old['transforms']['fusion']))))
  row=dict(case=key,scene=old['scene'],seed=seed,points=[len(ref),len(src)],seconds=elapsed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,old_RRE=oldmetric['RRE'],new_RRE=rre,old_RTE_cm=100*oldmetric['RTE'],new_RTE_cm=100*rte,old_success=oldmetric['success'],new_success=rre<15 and rte<.30,max_pose_delta=delta,selected_match=receipt['selected']==old['choice']['fusion'])
  row['pass']=bool(delta<=1e-4 and row['old_success']==row['new_success'] and row['selected_match'])
  rows.append(row);dump('results.json',dict(rows=rows,complete=False));print(json.dumps(row),flush=True);torch.cuda.empty_cache()
 dump('results.json',dict(rows=rows,complete=True,all_pass=all(r['pass'] for r in rows),total_seconds=time.perf_counter()-start,scope='Selected CSR pairs; not a full benchmark or a clean-environment installation test.'))
 faulthandler.cancel_dump_traceback_later()
 if not all(r['pass'] for r in rows):raise SystemExit('Replay mismatch: inspect saved results before claiming reproduction.')
if __name__=='__main__':main()
