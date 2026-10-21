# Frozen manuscript weights

Inference requires `encoder.pth.tar`, `global_teacher.joblib`,
`regional_teacher.joblib`, `arbiter.joblib`, and `manifest.json` in this directory.
The manifest verifies SHA-256 hashes before loading. Standard GeoTransformer weights
are not a drop-in replacement for the CERA-Reg encoder.

The prepared local folder contains all weights. A public asset download has not yet
been attached to the GitHub repository; do not interpret the repository URL as an
available weight download. The maintainer can upload the prepared reproduction asset
archive to GitHub Releases. Extract its `weights/` and `data/scorer_training/` paths
at the repository root. Raw datasets remain separate.

Prepared asset: `CERA-Reg-reproduction-assets-20261018.tar.gz` (174,361,521 bytes).
Verify the downloaded archive before extraction:

```bash
echo 'ba4b7e2d54503c0f09830d349cf27514dc3e78746945e49ac4fccd713950af67  CERA-Reg-reproduction-assets-20261018.tar.gz' | sha256sum -c -
tar --exclude=data/DATASETS.md -xzf CERA-Reg-reproduction-assets-20261018.tar.gz
```

The archive includes an older dataset note; the exclusion above preserves the
updated dataset documentation supplied with the source release.

Individual file hashes are checked again by `scripts/check_core.py`. Keep this
manifest and the encoder/scorer files together; do not replace only the encoder
with an experimental epoch checkpoint and assume the reported results still apply.

Joblib and PyTorch checkpoint files must come from a trusted source. Match the tested
scikit-learn version when loading the scorers. Freshly trained scorers are written to
an independent output directory; they never overwrite the frozen manuscript weights.
