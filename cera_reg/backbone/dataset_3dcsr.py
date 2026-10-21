import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .structure_features import make_structure_collate
import open3d as o3d

from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def _thin_cloud(points, target, radial_exp, radial_mix):
    """Thin a point cloud to `target` points.

    With probability `radial_mix`, applies lidar-like radial power-law thinning
    (p_i ~ 1/d_i^radial_exp w.r.t. the cloud centroid) so the thin side mimics
    the distance-dependent sparsity of LiDAR sampling; otherwise falls back to
    uniform random thinning. Weighted sampling without replacement is done via
    the Gumbel top-k trick (log-weight + Gumbel noise, take top-k).
    """
    n = points.shape[0]
    if radial_exp > 0 and np.random.rand() < radial_mix:
        centroid = points.mean(axis=0)
        d = np.linalg.norm(points - centroid, axis=1)
        w = 1.0 / np.power(d + 1e-3, radial_exp)
        w = w / w.max()  # stabilize log-weights
        gumbel = np.log(w) + np.random.gumbel(size=n)
        indices = np.argpartition(gumbel, -target)[-target:]
    else:
        indices = np.random.permutation(n)[:target]
    return points[indices]


def _ring_scan_sample(points, target, num_lines=64):
    """Structure ADR operator 1: spinning-LiDAR scan-pattern simulation.

    A synthetic sensor is placed at the cloud centroid. Points are converted
    to spherical coordinates; elevation is quantized into `num_lines` scan
    rings and, within each ring, azimuth is subsampled with a systematic
    stride (random phase per ring), so the surviving set carries the regular
    ring/line structure of a mechanical LiDAR while preserving the cloud
    extent. The result is padded/trimmed uniformly to exactly `target` points.
    """
    n = points.shape[0]
    if target >= n or target < 3:
        return points
    origin = points.mean(axis=0)
    d = points - origin
    r = np.linalg.norm(d, axis=1) + 1e-6
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2] / r, -1.0, 1.0))
    el_span = max(el.max() - el.min(), 1e-6)
    ring = ((el - el.min()) / el_span * num_lines).astype(np.int64)
    ring = np.clip(ring, 0, num_lines - 1)
    keep = np.zeros(n, dtype=bool)
    for rid in np.unique(ring):
        idx = np.where(ring == rid)[0]
        quota = max(1, int(round(target * idx.size / float(n))))
        if quota >= idx.size:
            keep[idx] = True
        else:
            order = np.argsort(az[idx])
            stride = idx.size / float(quota)
            start = np.random.uniform(0, stride)
            sel = (start + np.arange(quota) * stride).astype(np.int64)
            keep[idx[order[np.clip(sel, 0, idx.size - 1)]]] = True
    kept_idx = np.where(keep)[0]
    if kept_idx.size > target:
        kept_idx = np.random.choice(kept_idx, target, replace=False)
    elif kept_idx.size < target:
        pool = np.setdiff1d(np.arange(n), kept_idx)
        pad = np.random.choice(pool, target - kept_idx.size, replace=False)
        kept_idx = np.concatenate([kept_idx, pad])
    return points[kept_idx]


def _grid_scan_sample(points, target, res_w=96, res_h=72, retries=3):
    """Structure ADR operator 2: depth-camera raster simulation.

    A random view direction is drawn; points are orthographically projected
    onto a res_w x res_h image grid, keeping the frontmost point (closest to
    the view plane) inside each cell as a cheap visibility/occlusion proxy.
    The grid is refined until at least `target` points survive, then the set
    is trimmed uniformly so the raster structure stays dominant.
    """
    n = points.shape[0]
    if target >= n or target < 3:
        return points
    v = np.random.randn(3)
    v /= np.linalg.norm(v) + 1e-12
    a = np.random.randn(3)
    a -= a.dot(v) * v
    a /= np.linalg.norm(a) + 1e-12
    b = np.cross(v, a)
    pu, pv, pd = points @ a, points @ b, points @ v
    su, sv = pu.max() - pu.min(), pv.max() - pv.min()
    su, sv = max(su, 1e-6), max(sv, 1e-6)
    w, h = res_w, res_h
    kept_idx = np.arange(n)
    for _ in range(retries + 1):
        iw = np.clip(((pu - pu.min()) / su * w).astype(np.int64), 0, w - 1)
        ih = np.clip(((pv - pv.min()) / sv * h).astype(np.int64), 0, h - 1)
        cell = iw * h + ih
        order = np.argsort(-pd)
        _, first = np.unique(cell[order], return_index=True)
        kept_idx = order[first]
        if kept_idx.size >= target:
            break
        w, h = w * 2, h * 2
    if kept_idx.size > target:
        kept_idx = np.random.choice(kept_idx, target, replace=False)
    elif kept_idx.size < target:
        pool = np.setdiff1d(np.arange(n), kept_idx)
        pad = np.random.choice(pool, target - kept_idx.size, replace=False)
        kept_idx = np.concatenate([kept_idx, pad])
    return points[kept_idx]


def _stripe_sample(points, target, duty=0.35, retries=3):
    """Structure ADR operator 3: line-scanner stripe simulation.

    Points whose projection onto a random unit direction falls inside
    periodic stripe bands (duty-cycle `duty` of the projection span) are kept
    preferentially; the duty cycle is widened and retried if too few points
    fall in bands. Survivors are drawn from the band pool first (preserving
    the stripe structure) and padded from the non-band pool only as a last
    resort, then trimmed uniformly to exactly `target` points.
    """
    n = points.shape[0]
    if target >= n or target < 3:
        return points
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction) + 1e-12
    s = points @ direction
    s_min = s.min()
    span = max(s.max() - s_min, 1e-6)
    period = span * 0.2
    d = duty
    kept_idx = np.array([], dtype=np.int64)
    for _ in range(retries + 1):
        phase = np.random.uniform(0, period)
        in_band = ((s - s_min + phase) % period) < (period * d)
        band_idx = np.where(in_band)[0]
        if band_idx.size >= 4:
            p_keep = min(1.0, target / float(band_idx.size))
            draw = np.random.rand(band_idx.size)
            sel = band_idx[draw < p_keep]
            if sel.size >= min(target, int(band_idx.size * 0.5)):
                kept_idx = sel
                break
        d = min(d * 1.6, 0.95)
    if kept_idx.size == 0:
        kept_idx = np.random.choice(n, target, replace=False)
        return points[kept_idx]
    if kept_idx.size > target:
        kept_idx = np.random.choice(kept_idx, target, replace=False)
    else:
        pool = np.setdiff1d(np.arange(n), kept_idx)
        pad = np.random.choice(pool, target - kept_idx.size, replace=False)
        kept_idx = np.concatenate([kept_idx, pad])
    return points[kept_idx]


def _structure_thin(points, target, mode):
    if mode == 'ring':
        return _ring_scan_sample(points, target)
    if mode == 'grid':
        return _grid_scan_sample(points, target)
    return _stripe_sample(points, target)


def overlap_halfspace_crop(points, keep_min=0.2, keep_max=0.6, min_points=512):
    """OADR axis-2: keep points inside a random half-space to simulate the
    low-visibility/low-overlap regime of cross-source scans.

    The crop is a pure point selection, so the GT transform stays valid. The
    keep fraction is sampled log-uniformly in [keep_min, keep_max]; on failure
    (too few survivors) the crop strength is weakened and retried, falling
    back to the original cloud. Returns the cropped points.
    """
    if points.shape[0] <= min_points:
        return points
    keep_lo, keep_hi = np.log(keep_min), np.log(keep_max)
    for _ in range(5):
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal) + 1e-12
        keep = float(np.exp(np.random.uniform(keep_lo, keep_hi)))
        proj = points @ normal
        threshold = np.quantile(proj, keep)
        kept = points[proj <= threshold]
        if kept.shape[0] >= min_points:
            return kept
        # weaken the crop for the next retry (keep fraction moves toward 1)
        keep_lo = np.log(keep ** 0.5)
        keep_hi = np.log(max(keep ** 0.5, keep_max))
    return points


def _draw_structure_mode(structure_mix):
    mix = np.array(structure_mix, dtype=np.float64)
    return str(np.random.choice(['ring', 'grid', 'stripe'], p=mix / mix.sum()))


def asymmetric_density_resample(ref_points, src_points, ratio_exp=4.0, min_points=512,
                                 radial_exp=1.5, radial_mix=0.5,
                                 structure_prob=0.0, structure_mix=(0.4, 0.3, 0.3),
                                 cross_structure_prob=0.0):
    """Asymmetric density + sampling-structure augmentation (DCR-Former).

    Density axis: samples a ratio r in [2^-ratio_exp, 2^ratio_exp] (log2-uniform)
    and subsamples the corresponding side of the pair:
      r >= 1: src is thinned so that N_src ~= N_ref / r  (src sparser)
      r <  1: ref is thinned so that N_ref ~= N_src * r  (ref sparser)
    Thinning is uniform with probability (1 - radial_mix) and lidar-like radial
    power-law (p ~ 1/d^radial_exp) with probability radial_mix.

    Structure axis (ADR-S): with probability `structure_prob` the thinning is
    replaced by a sensor-pattern simulation (ring/grid/stripe per
    `structure_mix`) so the thinned side mimics real sampling anisotropy
    instead of an isotropic subset. The small-ratio early-exit is bypassed in
    this case (target ~ [0.5, 0.9] x N), creating same-density pairs that
    differ only in sampling structure -- the regime measured by E11
    (kinect_sfm vs kinect_lidar: same n_src, 6x inlier-ratio gap).

    Cross-structure: with probability `cross_structure_prob` the non-thinned
    side is independently resampled by a structure operator at near-unchanged
    density ([0.5, 0.9] x N), so both sides can carry different scan patterns
    in a single training pair.

    Returns:
        (ref_points, src_points)
    """
    ratio = 2.0 ** np.random.uniform(-ratio_exp, ratio_exp)
    do_structure = structure_prob > 0 and np.random.rand() < structure_prob
    if not do_structure and abs(np.log2(ratio)) < 0.25:
        return ref_points, src_points
    n_ref, n_src = ref_points.shape[0], src_points.shape[0]
    mode = _draw_structure_mode(structure_mix) if do_structure else None
    thin_src = ratio >= 1.0
    if abs(np.log2(ratio)) < 0.25:
        # structure-only pass at near-unchanged density (requires do_structure)
        base = n_src if thin_src else n_ref
        target = int(base * np.random.uniform(0.5, 0.9))
    else:
        target = int(n_ref / ratio) if thin_src else int(n_src * ratio)
    if thin_src:
        if min_points <= target < n_src:
            if do_structure:
                src_points = _structure_thin(src_points, target, mode)
            else:
                src_points = _thin_cloud(src_points, target, radial_exp, radial_mix)
        other = ref_points
    else:
        if min_points <= target < n_ref:
            if do_structure:
                ref_points = _structure_thin(ref_points, target, mode)
            else:
                ref_points = _thin_cloud(ref_points, target, radial_exp, radial_mix)
        other = src_points
    if cross_structure_prob > 0 and np.random.rand() < cross_structure_prob:
        n_other = other.shape[0]
        tgt = int(n_other * np.random.uniform(0.5, 0.9))
        if tgt >= min_points and tgt < n_other:
            other_mode = _draw_structure_mode(structure_mix)
            other = _structure_thin(other, tgt, other_mode)
    if thin_src:
        ref_points = other
    else:
        src_points = other
    return ref_points, src_points


class ThreeDCSRPairDataset(Dataset):
    def __init__(self, root_path, category='kinect_lidar', split='train',
                 voxel_size=0.025, point_limit=None,
                 use_augmentation=False, augmentation_noise=0.005, augmentation_rotation=1.0,
                 density_augmentation=False, density_ratio_exp=4.0,
                 density_augmentation_prob=0.8, density_min_points=512,
                 ref_match_voxel=None, scene_filter=None):
        super(ThreeDCSRPairDataset, self).__init__()
        self.root_path = root_path
        self.category = category
        self.split = split
        self.voxel_size = voxel_size
        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_rotation = augmentation_rotation
        self.density_augmentation = density_augmentation and split == 'train'
        self.density_ratio_exp = density_ratio_exp
        self.density_augmentation_prob = density_augmentation_prob
        self.density_min_points = density_min_points
        # M1 density matching: extra voxel downsample on the REF (dense kinect)
        # side so its NN spacing approaches the sparse lidar side's (~0.044 m).
        # Both sides then fall inside the density spectrum covered by ADR
        # training (alpha up to 4), instead of ref being in-distribution (0.84)
        # while src hangs at alpha 2.2 out-of-distribution.
        self.ref_match_voxel = ref_match_voxel
        # D1 scene-level split: when given, only pairs from these scenes are
        # kept (e.g. ['scene1', .., 'scene11'] for train, the rest for eval).
        # Scene-level separation prevents same-scene leakage between the
        # fine-tune set and the held-out evaluation scenes.
        self.scene_filter = set(scene_filter) if scene_filter is not None else None

        self.pairs = self._scan_pairs()
        if self.scene_filter is not None:
            self.pairs = [p for p in self.pairs if self._scene_of(p) in self.scene_filter]
        if len(self.pairs) == 0:
            raise ValueError(f'No pairs found in {root_path}/{category}')

        if split == 'train':
            self.pairs = self.pairs[:int(0.8 * len(self.pairs))]
        elif split == 'val':
            self.pairs = self.pairs[int(0.8 * len(self.pairs)):]

    def _scene_of(self, pair):
        """Dataset case/scene key; SfM difficulty is not a scene identifier."""
        if self.category == 'kinect_sfm':
            return os.path.basename(os.path.dirname(pair['src_path']))
        return os.path.basename(os.path.dirname(os.path.dirname(pair['src_path'])))

    def _scan_pairs(self):
        pairs = []
        category_path = os.path.join(self.root_path, self.category)

        if self.category == 'kinect_lidar':
            scene_dirs = sorted(glob.glob(os.path.join(category_path, 'scene*')))
            for scene_dir in scene_dirs:
                pair_dirs = sorted(glob.glob(os.path.join(scene_dir, 'pair*')))
                for pair_dir in pair_dirs:
                    kinect_path = os.path.join(pair_dir, 'kinect.ply')
                    lidar_path = os.path.join(pair_dir, 'lidar.ply')
                    gt_path = os.path.join(pair_dir, 'T_gt.txt')
                    if os.path.exists(kinect_path) and os.path.exists(lidar_path) and os.path.exists(gt_path):
                        pairs.append({
                            'src_path': kinect_path,
                            'tgt_path': lidar_path,
                            'gt_path': gt_path,
                        })
        elif self.category == 'kinect_sfm':
            for difficulty in ['easy', 'hard']:
                diff_path = os.path.join(category_path, difficulty)
                if not os.path.exists(diff_path):
                    continue
                scene_dirs = sorted(glob.glob(os.path.join(diff_path, '*')))
                for scene_dir in scene_dirs:
                    if not os.path.isdir(scene_dir):
                        continue
                    kinect_path = os.path.join(scene_dir, 'kinect.ply')
                    sfm_path = os.path.join(scene_dir, 'sfm.ply')
                    gt_path = os.path.join(scene_dir, 'T_gt.txt')
                    if os.path.exists(kinect_path) and os.path.exists(sfm_path) and os.path.exists(gt_path):
                        pairs.append({
                            'src_path': kinect_path,
                            'tgt_path': sfm_path,
                            'gt_path': gt_path,
                        })

        return pairs

    def _load_point_cloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float32)
        return points

    def _voxel_downsample(self, points, voxel_size=None):
        if voxel_size is None:
            voxel_size = self.voxel_size
        if voxel_size <= 0:
            return points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points, dtype=np.float32)

    def _random_sample(self, points):
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]
            points = points[indices]
        return points

    def _augment(self, ref_points, src_points, transform):
        if not self.use_augmentation:
            return ref_points, src_points, transform

        noise_ref = np.random.randn(*ref_points.shape).astype(np.float32) * self.augmentation_noise
        noise_src = np.random.randn(*src_points.shape).astype(np.float32) * self.augmentation_noise
        ref_points = ref_points + noise_ref
        src_points = src_points + noise_src

        angle = np.random.uniform(-self.augmentation_rotation, self.augmentation_rotation) * np.pi / 180.0
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + np.sin(angle).astype(np.float32) * K + (1 - np.cos(angle)).astype(np.float32) * (K @ K)
        aug_transform = np.eye(4, dtype=np.float32)
        # src_aug = R @ src, hence T_aug = T @ inverse(R).
        aug_transform[:3, :3] = R.T
        transform = (transform.astype(np.float32) @ aug_transform).astype(np.float32)
        src_points = np.ascontiguousarray((R @ src_points.T.astype(np.float32)).T, dtype=np.float32)

        return ref_points, src_points, transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        ref_points = self._load_point_cloud(pair['src_path'])
        src_points = self._load_point_cloud(pair['tgt_path'])
        transform = np.loadtxt(pair['gt_path']).astype(np.float32)

        ref_points = self._voxel_downsample(ref_points)
        src_points = self._voxel_downsample(src_points)

        # M1 density matching (eval-time preprocessing): push the dense kinect
        # side toward the lidar side's NN spacing so both halves of the input
        # live in the same density regime the backbone was trained on.
        if self.ref_match_voxel is not None and self.ref_match_voxel > self.voxel_size:
            ref_points = self._voxel_downsample(ref_points, self.ref_match_voxel)

        ref_points = self._random_sample(ref_points)
        src_points = self._random_sample(src_points)

        if self.density_augmentation and np.random.rand() < self.density_augmentation_prob:
            ref_points, src_points = asymmetric_density_resample(
                ref_points, src_points, self.density_ratio_exp, self.density_min_points
            )

        ref_points, src_points, transform = self._augment(ref_points, src_points, transform)

        ref_feats = np.ones_like(ref_points[:, :1], dtype=np.float32)
        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)

        overlap = 0.5

        return {
            'scene_name': pair['src_path'],
            'ref_points': ref_points,
            'src_points': src_points,
            'ref_feats': ref_feats,
            'src_feats': src_feats,
            'transform': transform,
            'overlap': overlap,
        }


def d1_scene_split(num_train_scenes=11, root_path=None):
    """D1 probe scene-level split (deterministic, numeric scene order).

    The local kinect_lidar split contains 165 pairs across 19 scenes. Scenes are sorted by
    numeric id: the first `num_train_scenes` are the fine-tune set, the rest
    the held-out evaluation set. Numeric (not lexicographic) ordering keeps
    scene2 next to scene1 instead of scene10.
    """
    category_path = os.path.join(root_path or os.environ.get('CERA_3DCSR_ROOT', 'data/3DCSR'), 'kinect_lidar')
    scene_dirs = sorted(glob.glob(os.path.join(category_path, 'scene*')),
                        key=lambda d: int(os.path.basename(d).replace('scene', '')))
    scene_names = [os.path.basename(d) for d in scene_dirs]
    return scene_names[:num_train_scenes], scene_names[num_train_scenes:]


def train_valid_data_loader_3dcsr(cfg, distributed):
    """D1: loaders for the scene-split 3DCSR fine-tune probe.

    Train = 11 scenes (99 pairs, rotation+noise augmentation only — the raw
    cross-source density/structure contrast IS the training signal here),
    val = 8 held-out scenes (72 pairs). neighbor_limits are calibrated on the
    3DCSR train set itself (calibrate cache would collide with 3DMatch's).
    """
    train_scenes, val_scenes = d1_scene_split(11, cfg.data.threedcsr_root)
    train_dataset = ThreeDCSRPairDataset(
        cfg.data.threedcsr_root,
        category=cfg.data.threedcsr_category,
        split='all',
        voxel_size=cfg.backbone.init_voxel_size,
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        scene_filter=train_scenes,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        make_structure_collate(cfg),
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    val_dataset = ThreeDCSRPairDataset(
        cfg.data.threedcsr_root,
        category=cfg.data.threedcsr_category,
        split='all',
        voxel_size=cfg.backbone.init_voxel_size,
        point_limit=cfg.val.point_limit,
        use_augmentation=False,
        scene_filter=val_scenes,
    )
    val_loader = build_dataloader_stack_mode(
        val_dataset,
        make_structure_collate(cfg),
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, val_loader, neighbor_limits
