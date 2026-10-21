import sys
import os
from functools import partial

import numpy as np


from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from .dataset_3dcsr import ThreeDCSRPairDataset, asymmetric_density_resample, overlap_halfspace_crop
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
from geotransformer.utils.torch import build_dataloader
from .data_adaptive import adaptive_registration_collate_fn_stack_mode
from .structure_features import make_structure_collate


class DensityAugmentedThreeDMatchPairDataset(ThreeDMatchPairDataset):
    """ThreeDMatch pair dataset with asymmetric density augmentation (DCR-Former).

    With probability `density_augmentation_prob`, one side of the pair is randomly
    subsampled so that the ref/src point-count ratio covers [2^-E, 2^E]
    (log2-uniform), teaching the density-robust encoder (MDB + FiLM) to be
    invariant to cross-source density inconsistency. Only applied on 'train'.
    """

    def __init__(self, dataset_root, subset, *args, density_augmentation=False,
                 density_ratio_exp=4.0, density_augmentation_prob=0.8, density_min_points=512,
                 density_radial_exp=1.5, density_radial_mix=0.5,
                 overlap_crop_prob=0.0, overlap_crop_keep_min=0.2, overlap_crop_keep_max=0.6,
                 structure_prob=0.0, structure_mix=(0.4, 0.3, 0.3), cross_structure_prob=0.0,
                 **kwargs):
        super(DensityAugmentedThreeDMatchPairDataset, self).__init__(dataset_root, subset, *args, **kwargs)
        self.density_augmentation = density_augmentation and subset == 'train'
        self.density_ratio_exp = density_ratio_exp
        self.density_augmentation_prob = density_augmentation_prob
        self.density_min_points = density_min_points
        self.density_radial_exp = density_radial_exp
        self.density_radial_mix = density_radial_mix
        self.overlap_crop_prob = overlap_crop_prob
        self.overlap_crop_keep_min = overlap_crop_keep_min
        self.overlap_crop_keep_max = overlap_crop_keep_max
        self.structure_prob = structure_prob
        self.structure_mix = structure_mix
        self.cross_structure_prob = cross_structure_prob

    def _load_point_cloud(self, file_name):
        # PyTorch >= 2.6 defaults weights_only=True which rejects legacy numpy
        # point cloud pickles; the 3DMatch files here are locally generated.
        import torch
        points = torch.load(os.path.join(self.data_root, file_name), weights_only=False)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = super(DensityAugmentedThreeDMatchPairDataset, self).__getitem__(index)
        if self.density_augmentation and np.random.rand() < self.density_augmentation_prob:
            ref_points, src_points = data_dict['ref_points'], data_dict['src_points']
            # OADR axis-2: random half-space crop on one side (low-overlap simulation).
            if np.random.rand() < self.overlap_crop_prob:
                keep_min, keep_max = self.overlap_crop_keep_min, self.overlap_crop_keep_max
                if np.random.rand() < 0.5:
                    ref_points = overlap_halfspace_crop(
                        ref_points, keep_min, keep_max, self.density_min_points)
                else:
                    src_points = overlap_halfspace_crop(
                        src_points, keep_min, keep_max, self.density_min_points)
            ref_points, src_points = asymmetric_density_resample(
                ref_points, src_points,
                self.density_ratio_exp, self.density_min_points,
                radial_exp=self.density_radial_exp,
                radial_mix=self.density_radial_mix,
                structure_prob=self.structure_prob,
                structure_mix=self.structure_mix,
                cross_structure_prob=self.cross_structure_prob,
            )
            data_dict['ref_points'] = ref_points
            data_dict['src_points'] = src_points
            data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
            data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        return data_dict


def train_valid_data_loader(cfg, distributed):
    train_dataset = DensityAugmentedThreeDMatchPairDataset(
        cfg.data.threedmatch_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        density_augmentation=cfg.train.density_augmentation,
        density_ratio_exp=cfg.train.density_ratio_exp,
        density_augmentation_prob=cfg.train.density_augmentation_prob,
        density_min_points=cfg.train.density_min_points,
        density_radial_exp=cfg.train.density_radial_exp,
        density_radial_mix=cfg.train.density_radial_mix,
        overlap_crop_prob=cfg.train.overlap_crop_prob,
        overlap_crop_keep_min=cfg.train.overlap_crop_keep_min,
        overlap_crop_keep_max=cfg.train.overlap_crop_keep_max,
        structure_prob=cfg.train.structure_prob,
        structure_mix=(cfg.train.structure_mix_ring, cfg.train.structure_mix_grid,
                        cfg.train.structure_mix_stripe),
        cross_structure_prob=cfg.train.cross_structure_prob,
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

    val_dataset = ThreeDMatchPairDataset(
        cfg.data.threedmatch_root,
        'val',
        point_limit=cfg.val.point_limit,
        use_augmentation=False,
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


def test_data_loader_3dmatch(cfg, benchmark='3DMatch'):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.threedmatch_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = ThreeDMatchPairDataset(
        cfg.data.threedmatch_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        make_structure_collate(cfg),
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits


def test_data_loader_3dcsr(cfg, split='val', use_da_window=False, ref_match_voxel=None):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.threedmatch_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = ThreeDCSRPairDataset(
        cfg.data.threedcsr_root,
        category=cfg.data.threedcsr_category,
        split=split,
        voxel_size=cfg.backbone.init_voxel_size,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        ref_match_voxel=ref_match_voxel,
    )
    if use_da_window:
        # DA-window pipeline: per-cloud adaptive search radii + cloud_scales.
        collate_fn = partial(
            adaptive_registration_collate_fn_stack_mode,
            num_stages=cfg.backbone.num_stages,
            voxel_size=cfg.backbone.init_voxel_size,
            search_radius=cfg.backbone.init_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=True,
            alpha_max=cfg.dre.da_alpha_max,
            alpha_gate=cfg.dre.da_alpha_gate,
        )
        collate_fn = make_structure_collate(cfg, collate_fn)
        test_loader = build_dataloader(
            test_dataset,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        test_loader = build_dataloader_stack_mode(
            test_dataset,
            make_structure_collate(cfg),
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=False,
        )

    return test_loader, neighbor_limits
