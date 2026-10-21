"""Register a raw XYZ pair supplied as NPZ; predictions contain no GT fields."""
from pathlib import Path
import sys,os,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='NPZ with ref_points and src_points, in meters')
    parser.add_argument('--output',required=True,help='New JSON destination; never overwrites a result')
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--route',choices=['observed','native'],default='observed')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=7351)
    args=parser.parse_args();dest=Path(args.output)
    if dest.exists():parser.error('Output already exists; choose another path')
    import numpy as np
    import torch
    from cera_reg.pipeline import CERAReg
    torch.set_num_threads(2)
    with np.load(args.input,allow_pickle=False) as data:
        ref=data['ref_points'].copy();src=data['src_points'].copy()
    model=CERAReg(weights=args.weights,device=args.device)
    try:pose,receipt=model.register(ref,src,route=args.route,seed=args.seed)
    except torch.cuda.OutOfMemoryError:
        dest.parent.mkdir(parents=True,exist_ok=True)
        error=dest.with_suffix('.oom.json')
        error.write_text(json.dumps(dict(status='oom',input=args.input,route=args.route,seed=args.seed,retry='Rerun the same command when GPU memory is available; no valid pose was produced.'),indent=2))
        raise
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as stream:json.dump(dict(transform=pose.tolist(),receipt=receipt),stream,indent=2,allow_nan=False)
    print(dest)

if __name__=='__main__':main()
