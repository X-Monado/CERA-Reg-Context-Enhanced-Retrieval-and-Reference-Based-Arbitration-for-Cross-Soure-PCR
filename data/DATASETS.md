# Dataset preparation

Download datasets from their authors and retain their terms of use. No raw dataset is redistributed here.

- **3DCSR:** [official benchmark](https://multimediauts.org/3D_data_for_registration/).
- **3DMatch / 3DLoMatch:** [original 3DMatch project](https://3dmatch.cs.princeton.edu/) and [GeoTransformer preprocessing instructions](https://github.com/qinzheng93/GeoTransformer#3dmatch), which link to the PREDATOR-prepared fragments and metadata.

Expected local layout (names are case-sensitive):

```text
data/
  3DCSR/
    kinect_lidar/scene1/pair1/{kinect.ply,lidar.ply,T_gt.txt}
    kinect_sfm/easy/<case>/{kinect.ply,sfm.ply,T_gt.txt}
    kinect_sfm/hard/<case>/{kinect.ply,sfm.ply,T_gt.txt}
  3DMatch/
    metadata/{train.pkl,val.pkl,3DMatch.pkl,3DLoMatch.pkl}
    data/train/<sequence>/cloud_bin_0.pth
    data/test/<scene>/cloud_bin_0.pth
  3DLoMatch/       # Documentation placeholder: shares 3DMatch fragments.
  scorer_training/
    manifest.json
    000.npz ... 079.npz
```

The 3DCSR loader treats Kinect as reference and LiDAR/SfM as source.
`T_gt.txt` must transform source into reference; coordinates are in meters.
Do not invert transforms based only on archive file names. The loader uses
lexicographically sorted `scene*` and `pair*` directories; seed = 7351 + pair index.
The adopted local sets contain 165 Kinect–LiDAR and 37 Kinect–SfM pairs.

3DLoMatch uses the same processed test fragments as 3DMatch with its own pair metadata;
do not duplicate the point clouds. Official evaluation additionally needs benchmark
GT logs and information matrices. The generic 3DCSR pose threshold cannot replace that evaluator.

Override paths with `CERA_3DCSR_ROOT` and `CERA_3DMATCH_ROOT`, or use `--root` in
`scripts/evaluate_3dcsr.py`. Run commands from the repository root.

Download-page check (2026-10-21): the 3DCSR benchmark page and GeoTransformer
repository were accessible. The original 3DMatch website returned HTTP 403 to
the automated checker; use the upstream GeoTransformer preprocessing links if
that also happens in your browser. Raw archive downloads were not retested.

The optional scorer-training archive contains 80 source-domain evidence packs, not raw
point clouds. The regional branch uses the original 48 observations; the global branch
uses all 80. Each pack stores `features[N,40]`, `regional[N,64]`, `poses[N,4,4]`,
boolean `success`, `strict10`, `strict5`, and continuous `quality[N]`; the final row
is the reference. Labels come from source-domain GT only. `manifest.json` records
SHA-256 checksums, group/fold assignments, and branch membership. The historical
sequence folds are preserved for replay and do not establish physical-scene independence.
