# CERA-Reg

**Context-Enhanced Retrieval and Reference-Based Arbitration for Cross-Source Point Cloud Registration**

[Project repository](https://github.com/X-Monado/CERA-Reg-Context-Enhanced-Retrieval-and-Reference-Based-Arbitration-for-Cross-Soure-PCR)

CERA-Reg aligns point clouds from different sensors. A frozen hierarchical encoder
provides local descriptors and superpoint context. Bidirectional retrieval expands
correspondences; endpoint-unique compatibility communities generate poses; global
and regional scorers decide whether to replace a reference pose.

## Requirements and scope

- Linux, a C++ compiler, Conda, and an NVIDIA GPU with a driver compatible with the chosen PyTorch CUDA build for practical inference.
- Tested software: Python 3.8.20, PyTorch 2.4.0 + CUDA 12.4, NumPy 1.24.4, scikit-learn 1.3.2. CPU supports geometry checks and scorer fitting.
- Input: finite `N x 3` XYZ arrays in **meters**, with consistent scale and some shared structure. Color and reflectance are not required. Unknown relative scale is not estimated.
- Intended setting: indoor rigid registration. Low overlap, repeated structure, and outdoor generalization remain limitations.
- Frozen weights are required; see [weight status](weights/WEIGHTS.md). This folder includes inference and CPU scorer training from prepared source evidence. Encoder training from scratch and a standalone official 3DMatch/3DLoMatch evaluation launcher are not yet packaged.

## Quick installation

```bash
git clone https://github.com/X-Monado/CERA-Reg-Context-Enhanced-Retrieval-and-Reference-Based-Arbitration-for-Cross-Soure-PCR.git CERA-Reg
cd CERA-Reg
conda env create -f environment.yml
conda activate CERA-Reg
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
MAX_JOBS=2 python setup.py build_ext --inplace
python -m pip check
```

The [PyTorch version archive](https://pytorch.org/get-started/previous-versions/) documents
CUDA wheel alternatives. The local sampling extension is CPU-only and does not require
`nvcc`; neural inference still uses CUDA. Rebuild the extension after changing Python or
PyTorch. These commands specify the tested versions; a clean-machine installation has
not yet been validated. Run all commands from this repository root.

## Model assets and installation check

Inference and the CPU model check require the frozen weights. Download
`CERA-Reg-reproduction-assets-20261018.tar.gz` from the project's
[Releases page](https://github.com/X-Monado/CERA-Reg-Context-Enhanced-Retrieval-and-Reference-Based-Arbitration-for-Cross-Soure-PCR/releases)
**once the maintainer has attached it**. The archive is prepared locally; public
availability is not yet confirmed. It also contains the optional scorer-training
evidence. See [weights/WEIGHTS.md](weights/WEIGHTS.md) for its checksum.

```bash
# Put the downloaded archive in the repository root, then extract it there.
tar --exclude=data/DATASETS.md -xzf CERA-Reg-reproduction-assets-20261018.tar.gz
python scripts/check_core.py --load-encoder --report output/core_check.json
```

The check verifies asset hashes, geometry, evidence dimensions, reference fallback,
and encoder loading on CPU. It does not run a GPU forward pass or reproduce a full
benchmark. Model assets are separate from source code; raw datasets are not included.

## Data

See [download links and directory layout](data/DATASETS.md).
Use `data/3DCSR` and `data/3DMatch`, or set:

```bash
export CERA_3DCSR_ROOT=/path/to/3DCSR
export CERA_3DMATCH_ROOT=/path/to/3DMatch
```
## Pre-training Model

https://pan.baidu.com/s/1wj7bd6W71OpcTP3IOydlnA?pwd=4zj4

## Register and test

Save a pair as `pair.npz` with arrays `ref_points` and `src_points`:

```python
import numpy as np
np.savez("pair.npz", ref_points=reference_xyz, src_points=source_xyz)
```

```bash
python scripts/infer_pair.py --input pair.npz --output output/pair.json --route observed
# A small raw-data check; remove --limit for the full subset.
python scripts/evaluate_3dcsr.py --category kinect_sfm --output output/sfm_smoke --limit 2
python scripts/evaluate_3dcsr.py --category kinect_lidar --output output/lidar
# CPU-only summary of already saved predictions.
python scripts/evaluate_3dcsr.py --category kinect_lidar --output output/lidar --summarize-only
```



## Train the scorers 

Place the prepared source evidence archive under `data/scorer_training/`:

```bash
python scripts/train_scorers.py --manifest data/scorer_training/manifest.json --dry-run
python scripts/train_scorers.py --manifest data/scorer_training/manifest.json --output output/trained_scorers
python scripts/package_weights.py --scorers output/trained_scorers --encoder weights/encoder.pth.tar --output output/retrained_weights
python scripts/infer_pair.py --input pair.npz --output output/retrained_pair.json --weights output/retrained_weights
```


## Recorded  results

| Dataset | Success / RR (%) | RRE (deg) | RTE (cm) | IR (%) | FMR (%) |
|---|---:|---:|---:|---:|---:|
| 3DCSR Kinect-LiDAR | 69.09 | 2.87 | 14.57 | 11.47 | 58.18 |
| 3DCSR Kinect-SfM | 81.08 | 2.27 | 6.83 | 42.44 | 86.49 |
| 3DMatch | 93.27 | 2.04 | 6.65 | 65.86 | 99.13 |
| 3DLoMatch | 70.74 | 3.53 | 9.12 | 41.68 | 92.04 |




## Acknowledgements

The encoder and geometric operations build on [GeoTransformer](https://github.com/qinzheng93/GeoTransformer).
Its MIT notice is retained in `LICENSE`. Cross-PCR informs the density-adaptation design;
this project does not claim to distribute its official implementation. Publication and
citation details will be updated when available.

For maintainers: [release contents and upload checklist](RELEASE.md).
