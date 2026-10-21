import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

_C.seed = 7351

_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')


_C.data = edict()
_C.data.threedmatch_root = os.environ.get('CERA_3DMATCH_ROOT', 'data/3DMatch')
_C.data.threedcsr_root = os.environ.get('CERA_3DCSR_ROOT', 'data/3DCSR')
_C.data.threedcsr_category = 'kinect_lidar'
_C.data.cross3dreg_root = os.environ.get('CERA_CROSS3DREG_ROOT', 'data/Cross3DReg')
_C.data.kitti_root = os.environ.get('CERA_KITTI_ROOT', 'data/KITTI')
# Outdoor street scenes (~100 m) rescaled to indoor-trained statistics;
# metrics converted back to original units at evaluation time.
_C.data.cross3dreg_scale = 0.05

_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 4
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0
# Asymmetric density augmentation (DCR-Former innovation: density-invariant training)
_C.train.density_augmentation = True
_C.train.density_ratio_exp = 4.0  # ref/src density ratio covers [2^-4, 2^4], log2-uniform
_C.train.density_augmentation_prob = 0.5  # Step-1 ladder: +ADR uniform only (resumes from try4 snapshot); radial/structure/OADR stay off
_C.train.density_min_points = 512
# Radial (lidar-like) thinning: with prob radial_mix, thinning uses p ~ 1/d^radial_exp
# instead of uniform, teaching the encoder distance-dependent sparsity (E11: 85% of
# cross-source failures are generation failures -> feature-level, not hypothesis-level).
_C.train.density_radial_exp = 1.5
_C.train.density_radial_mix = 0.5  # R2: radial thinning (run8's winning recipe) 50% / uniform 50%
# Overlap-aware augmentation (OADR axis-2): random half-space crop on one side of
# the pair simulates the low-visibility/low-overlap regime of cross-source scans
# (measured 3DCSR kinect-lidar overlap: median 6.3%, all pairs < 30%).
_C.train.overlap_crop_prob = 0.0  # R2: off (run8 recipe = radial ADR only)
_C.train.overlap_crop_keep_min = 0.15  # keep fraction of the cropped side (log-uniform range)
_C.train.overlap_crop_keep_max = 0.6

# ADR-S: sampling-structure simulation (route-2 recipe). E11 measured the
# residual kinect_sfm-vs-kinect_lidar gap in sampling structure, not density
# (same n_src, 6x IR gap). These operators replace isotropic thinning with
# sensor-pattern simulations: spinning-LiDAR rings / depth-camera rasters /
# line-scanner stripes, plus independent cross-structure resampling of the
# non-thinned side (same-density, different-structure pairs).
_C.train.structure_prob = 0.0       # R2: off (run8 recipe = radial ADR only)
_C.train.structure_mix_ring = 0.4
_C.train.structure_mix_grid = 0.3
_C.train.structure_mix_stripe = 0.3
_C.train.cross_structure_prob = 0.0  # Step-2: off (single variable = structure_prob)

_C.val = edict()
_C.val.batch_size = 1
_C.val.num_workers = 4
_C.val.point_limit = None

_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 4
_C.test.point_limit = None

_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3

_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

_C.optim = edict()
# fine-tuning lr for warm-start: 1e-4 drifted, 1e-5 plateaued on val after ep1;
# 3e-5 for the radial-ADR short run to move faster without drift.
_C.optim.lr = 1e-4  # OFFICIAL 3DMatch scratch value (verified in GeoTransformer clone config); 1e-3 in earlier smokes was 10x too high — c_loss stuck at 1.0
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 2  # R2 (run8+radial): resumes from run8 (epoch 1) -> one more epoch with radial ADR
_C.optim.grad_acc_steps = 1

_C.backbone = edict()
_C.backbone.num_stages = 4
# NOTE: must match official 3DMatch config (0.025) for warm-load; radius/sigma follow.
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# DCR-Former density-robust encoder switches (MDB + FiLM)
_C.dre = edict()
_C.dre.use_mdb = True
_C.dre.use_film = True
_C.dre.film_hidden_dim = 32
# Feed density-conditioned P0/P1/P2 features into the NEXT encoder stage.
# The previous DRE used P1/P2 conditioning only in decoder skip connections,
# leaving coarse features largely unaware of those density corrections.
_C.dre.propagate_density = True
# Ignore the self neighbor when estimating spacing; a self-only point is
# assigned the search radius rather than an apparent zero spacing.
_C.dre.robust_density = True
# Density-adaptive KPConv window (DA-window): dilate the sparse side's
# convolution window by alpha = rho_cloud / rho_ref so its KPConv support
# matches the 3DMatch density the pretrained encoder expects. Requires the
# adaptive data pipeline (data_adaptive.py) to supply 'cloud_scales'.
_C.dre.use_da_window = False  # F3b short run negative (see feature_experiments.md); off for mainline
_C.dre.da_alpha_max = 4.0
_C.dre.da_alpha_gate = 1.1  # alpha below this = in-distribution, no dilation

# Experimental P0/P1 geometry residual. Opt in explicitly; old checkpoints
# and evaluation protocols remain reproducible with this disabled.
_C.structure = edict()
_C.structure.enabled = False
_C.structure.levels = (0, 1)
_C.structure.scales = (1.0, 2.0, 4.0)
_C.structure.neighbor_limit = 32
_C.structure.hidden_dim = 32
_C.structure.chunk_size = 1024
_C.structure.density_weighting = False
_C.structure.reliability_gate = True

_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

_C.fine_matching = edict()
_C.fine_matching.topk = 3

_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

_C.otm_matching = edict()
_C.otm_matching.topk = 3
_C.otm_matching.acceptance_radius = 0.1
_C.otm_matching.confidence_threshold = 0.05
_C.otm_matching.correspondence_threshold = 3
_C.otm_matching.correspondence_limit = None
_C.otm_matching.num_refinement_steps = 5
_C.otm_matching.sc_topk = 6
_C.otm_matching.sc2_topk = 6
_C.otm_matching.sc_threshold = 0.5
# DCR-Former innovation 2: joint spectral strict selection (TRIM + SC2 + feature consistency).
_C.otm_matching.use_joint_spectral_filter = True
_C.otm_matching.spectral_sigma = 0.1
_C.otm_matching.spectral_feature_weight = 0.5
_C.otm_matching.spectral_iterations = 50
_C.otm_matching.spectral_threshold = 0.25
# DCR-Former innovation 3: multi-hypothesis pose estimation (m weighted-SVD + chamfer/semantic selection).
_C.otm_matching.use_multi_hypothesis = True
_C.otm_matching.num_hypotheses = 8
_C.otm_matching.subset_ratio = 0.7
_C.otm_matching.chamfer_radius = 0.1
_C.otm_matching.hypothesis_semantic_weight = 0.3

_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
# Overlap-aware coarse supervision (SR-Net-style "宽进严管"): lower the positive
# threshold so that true low-overlap superpoint correspondences (0.05-0.1, the
# un-supervised dead zone that dominates cross-source pairs with measured 6.3%
# median overlap) enter the positive set, while the o^kappa power weighting
# (loss.py) keeps their supervision intensity scaled by overlap quality.
_C.coarse_loss.positive_overlap = 0.1  # Step-0: official scratch value; OACL 0.05 reintroduced at Step 3
_C.coarse_loss.overlap_power = 0.5

_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.1

_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0
# Density consistency regularization (produced by the model on ~50% of training batches)
_C.loss.weight_consistency_loss = 0.1
_C.loss.consistency_prob = 0.5


def make_cfg():
    return _C

