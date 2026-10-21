"""DCR-Former density-robust encoder backbone.

Extends the standard KPConvFPN with two cross-source components:

1. MultiDensityBlock (MDB): density-robust fusion adapted from Cross-PCR (AAAI 2025).
   For features F_l at grid level l, available coarse/fine views are computed
   on the existing multi-scale grids and fused:
     down : F_l --gather--> P_{l+1} --ResBlock--> nearest_upsample --> P_l
     up   : F_l --nearest_upsample--> P_{l-1} --ResBlock--> maxpool --> P_l
     fused: F_l + zero-initialized UnaryBlock(cat[down, up])

2. FiLM density conditioning: a per-point condition c = [log rho, log(1+n)] is
   estimated from the radius neighborhoods, where
     rho : median distance to non-self neighbors within the level radius,
     n   : number of non-self neighbors retained by the radius search.
   Each modulator applies  F' = F * (1 + gamma(c)) + beta(c)  with the last MLP
   layer zero-initialized, so the modulator starts as an identity map.

With propagate_density enabled, P0 FiLM and P1/P2 MDB+FiLM modify features
before the next encoder stage; the legacy mode applies P1/P2 only in decoder
skips. All grid mappings reuse the standard GeoTransformer data pipeline.
"""
import torch
import torch.nn as nn

from geotransformer.modules.kpconv import (
    UnaryBlock,
    LastUnaryBlock,
    nearest_upsample,
)
from geotransformer.modules.kpconv.functional import maxpool
from .da_kpconv import DAConvBlock, DAResidualBlock
from .structure_features import MultiRadiusGeometryAdapter


# (radius_level, query_level) window-scale combinations consumed by the encoder:
#   neighbors_list[l]   -> queries P_l      -> (l, l)
#   subsampling_list[l] -> queries P_{l+1}  -> (l, l+1)
_WINDOW_COMBOS = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
_NULL_WS = {c: None for c in _WINDOW_COMBOS}


class DensityEstimator(nn.Module):
    """Estimate local spacing and support, with a legacy mode for old checkpoints."""

    def __init__(self, robust=True):
        super(DensityEstimator, self).__init__()
        self.robust = robust

    def forward(self, points, neighbors, radius=None):
        """Args:
            points: (N, 3) coordinates of the level.
            neighbors: (N, K) radius-search neighbor indices, padded with N.

        Returns:
            cond: (N, 2) density condition [log rho, log(1+n)] in robust mode.
        """
        num_points = points.shape[0]
        valid = neighbors < num_points  # (N, K)
        padded_points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
        neighbor_points = padded_points[neighbors]  # (N, K, 3)
        dists = torch.norm(points.unsqueeze(1) - neighbor_points, dim=-1)  # (N, K)
        if self.robust:
            # Radius search includes each query itself. In the old descriptor a
            # self-only point had log spacing -13.8, falsely resembling a dense
            # point. A missing non-self neighbor is censored at the search radius.
            if radius is None or radius <= 0:
                raise ValueError('robust density estimation requires a positive search radius')
            valid = valid & (dists > 1e-8)
        counts = valid.sum(dim=1)  # (N,)
        has_neighbor = counts > 0
        dists = dists.masked_fill(~valid, float('inf'))

        sorted_dists, _ = torch.sort(dists, dim=1)
        median_idx = (counts.clamp(min=1) - 1) // 2
        median_dists = sorted_dists.gather(1, median_idx.unsqueeze(1)).squeeze(1)  # (N,)
        fallback = float(radius) if self.robust else 0.0
        median_dists = torch.where(has_neighbor, median_dists, torch.full_like(median_dists, fallback))

        log_rho = torch.log(median_dists + 1e-6)
        if self.robust:
            # Absolute support is stable when radius_search's returned width
            # changes across pairs; dividing by that width is not.
            support_signal = torch.log1p(counts.float())
        else:
            support_signal = valid.float().mean(dim=1)
        return torch.stack([log_rho, support_signal], dim=1)


class FiLMDensityModulator(nn.Module):
    """Feature-wise density conditioning: F' = F * (1 + gamma(c)) + beta(c).

    The final MLP layer is zero-initialized so the modulator is an exact
    identity at the start of training.
    """

    def __init__(self, feat_dim, cond_dim=2, hidden_dim=32):
        super(FiLMDensityModulator, self).__init__()
        self.feat_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats, cond):
        gamma_beta = self.mlp(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return feats * (1.0 + gamma) + beta


class MultiDensityBlock(nn.Module):
    """Zero-initialized multi-density residual block on the precomputed multi-scale grids.

    The main path is the (warm-loaded) encoder itself; this block only computes
    auxiliary density branches and fuses them with a zero-initialized projection,
    so the backbone starts exactly equal to the pretrained KPConvFPN:
        F_l' = F_l + fusion(cat[down(F_l), up(F_l)]),  fusion zero-init.
    """

    def __init__(self, channels, kernel_size, radius, sigma, group_norm, has_coarser=True, has_finer=True):
        """Args:
            channels: feature dimension at this level (in == out).
            radius / sigma: KPConv parameters of the current level.
            has_coarser: whether a coarser grid P_{l+1} exists (enables down branch).
            has_finer: whether a finer grid P_{l-1} exists (enables up branch).
        """
        super(MultiDensityBlock, self).__init__()
        assert has_coarser or has_finer
        self.has_coarser = has_coarser
        self.has_finer = has_finer

        if has_coarser:
            self.branch_down = DAResidualBlock(channels, channels, kernel_size, radius * 2, sigma * 2, group_norm)
        if has_finer:
            self.branch_up = DAResidualBlock(channels, channels, kernel_size, radius / 2, sigma / 2, group_norm)

        num_aux = int(has_coarser) + int(has_finer)
        self.fusion = UnaryBlock(num_aux * channels, channels, group_norm)
        # Zero-init the fusion projection: the whole block outputs 0 at start,
        # so F_l' == F_l and warm-loading stage-4 weights yields an exact identity.
        nn.init.zeros_(self.fusion.mlp.weight)
        nn.init.zeros_(self.fusion.mlp.bias)

    def forward(self, feats, points_list, neighbors_list, subsampling_list, upsampling_list, level, ws=None):
        """Args:
            feats: (N_l, C) features at grid level `level`.
            ws: optional dict {(op_level, q_level): per-point window scale tensor}
                from the DA-window pipeline; None -> standard behaviour.
        Returns:
            aux: (N_l, C) multi-density residual (all zeros at initialization).
        """
        aux_feats = []

        if self.has_coarser:
            # Down branch: gather to P_{l+1}, process, upsample back to P_l.
            scale_down = ws.get((level + 1, level + 1)) if ws is not None else None
            # Sentinel N denotes an empty query neighborhood. Do not index N
            # in the unpadded feature array (can occur on sparse/empty patches).
            padded_feats = torch.cat([feats, torch.zeros_like(feats[:1])], dim=0)
            down_feats = padded_feats[subsampling_list[level][:, 0]]
            down_feats = self.branch_down(
                down_feats, points_list[level + 1], points_list[level + 1], neighbors_list[level + 1],
                window_scale=scale_down
            )
            down_feats = nearest_upsample(down_feats, upsampling_list[level])
            aux_feats.append(down_feats)

        if self.has_finer:
            # Up branch: upsample to P_{l-1}, process, max-pool back to P_l.
            scale_up = ws.get((level - 1, level - 1)) if ws is not None else None
            up_feats = nearest_upsample(feats, upsampling_list[level - 1])
            up_feats = self.branch_up(
                up_feats, points_list[level - 1], points_list[level - 1], neighbors_list[level - 1],
                window_scale=scale_up
            )
            up_feats = maxpool(up_feats, subsampling_list[level - 1])
            aux_feats.append(up_feats)

        return self.fusion(torch.cat(aux_feats, dim=1))


class DREBackbone(nn.Module):
    """KPConvFPN backbone augmented with MDB fusion and FiLM density conditioning.

    Drop-in replacement of KPConvFPN: same constructor args plus DRE switches,
    same forward signature (feats, data_dict) -> feats_list.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        init_dim,
        kernel_size,
        init_radius,
        init_sigma,
        group_norm,
        use_mdb=True,
        use_film=True,
        film_hidden_dim=32,
        use_da_window=False,
        propagate_density=False,
        robust_density=False,
        structure_cfg=None,
    ):
        super(DREBackbone, self).__init__()
        self.use_mdb = use_mdb
        self.use_film = use_film
        self.use_da_window = use_da_window
        self.propagate_density = propagate_density
        self.init_radius = init_radius
        self.use_structure = structure_cfg is not None and structure_cfg.enabled
        if self.use_structure:
            if tuple(structure_cfg.levels) != (0, 1):
                raise ValueError('the structure adapter currently supports exactly P0/P1')
            settings = dict(hidden_dim=structure_cfg.hidden_dim,
                            num_scales=len(structure_cfg.scales),
                            chunk_size=structure_cfg.chunk_size,
                            density_weighting=structure_cfg.density_weighting,
                            reliability_gate=structure_cfg.reliability_gate)
            self.structure0 = MultiRadiusGeometryAdapter(input_dim, init_dim, **settings)
            self.structure1 = MultiRadiusGeometryAdapter(init_dim * 2, init_dim * 2, **settings)

        # Encoder blocks use the density-adaptive KPConv classes: with
        # window_scale=None they reproduce the standard KPConv bit-exactly, and
        # their state-dict layout is identical, so stage4 weights warm-load
        # unchanged whether or not DA windows are enabled.
        conv_block = DAConvBlock
        res_block = DAResidualBlock

        self.encoder1_1 = conv_block(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = res_block(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = res_block(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = res_block(init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm)
        self.encoder2_3 = res_block(init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm)

        self.encoder3_1 = res_block(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = res_block(init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm)
        self.encoder3_3 = res_block(init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm)

        self.encoder4_1 = res_block(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )
        self.encoder4_2 = res_block(init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm)
        self.encoder4_3 = res_block(init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm)

        # DRE components. Level 3 has no coarser grid -> up branch only.
        if use_mdb:
            self.mdb3 = MultiDensityBlock(
                init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm, has_coarser=False
            )
            self.mdb2 = MultiDensityBlock(
                init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
            )
            self.mdb1 = MultiDensityBlock(
                init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
            )

        if use_film:
            if propagate_density:
                self.film0 = FiLMDensityModulator(init_dim * 2, hidden_dim=film_hidden_dim)
            self.film3 = FiLMDensityModulator(init_dim * 16, hidden_dim=film_hidden_dim)
            self.film2 = FiLMDensityModulator(init_dim * 8, hidden_dim=film_hidden_dim)
            self.film1 = FiLMDensityModulator(init_dim * 4, hidden_dim=film_hidden_dim)
            self.film_f = FiLMDensityModulator(output_dim, hidden_dim=film_hidden_dim)
            self.density_estimator = DensityEstimator(robust=robust_density)

        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def _build_window_scales(self, data_dict):
        """Expand per-cloud alpha (num_stages, B) into the per-point tensors the
        encoder consumes, keyed by (radius_level, query_level).

        Neighbour/subsampling lists at level l were searched at radius
        base_l * alpha_{c,l}; upsampling at 2*base_l * alpha_{c,l} (see
        data_adaptive.py). Consumers:
          neighbors_list[l]          -> queries P_l      -> ws[(l, l)]
          subsampling_list[l]        -> queries P_{l+1} -> ws[(l, l+1)]
          MDB branches at level l±1  -> ws[(l±1, l±1)]

        Returns a dict whose missing/None values mean "no dilation" so the
        standard pipeline path stays bit-identical.
        """
        combos = _WINDOW_COMBOS
        cloud_scales = data_dict.get('cloud_scales')
        if cloud_scales is None:
            return dict(_NULL_WS)
        if not torch.is_tensor(cloud_scales):
            cloud_scales = torch.as_tensor(cloud_scales)
        lengths_list = data_dict['lengths']
        ws = {}
        for radius_level, query_level in combos:
            if radius_level >= cloud_scales.shape[0] or query_level >= len(lengths_list):
                ws[(radius_level, query_level)] = None
                continue
            lengths = lengths_list[query_level]
            alpha_row = cloud_scales[radius_level].to(lengths.device)
            ws[(radius_level, query_level)] = torch.repeat_interleave(alpha_row, lengths)
        return ws

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
        if self.use_structure and 'structure' not in data_dict:
            raise ValueError('enabled structure adapter requires make_structure_collate(cfg) '
                             'or prepare_structure_data(...) before moving data to the device')

        # DA-window per-point scales (all-None when the standard pipeline is used).
        ws = self._build_window_scales(data_dict) if self.use_da_window else _NULL_WS

        # Density conditions per level (only needed when FiLM is on).
        if self.use_film:
            if self.propagate_density:
                cond0 = self.density_estimator(points_list[0], neighbors_list[0], self.init_radius)
            cond1 = self.density_estimator(points_list[1], neighbors_list[1], self.init_radius * 2)
            cond2 = self.density_estimator(points_list[2], neighbors_list[2], self.init_radius * 4)
            cond3 = self.density_estimator(points_list[3], neighbors_list[3], self.init_radius * 8)

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0], window_scale=ws[(0, 0)])
        if self.use_structure:
            feats_s1 = feats_s1 + self.structure0(feats, points_list[0], data_dict['structure']['0'])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0], window_scale=ws[(0, 0)])
        if self.propagate_density and self.use_film:
            feats_s1 = self.film0(feats_s1, cond0)

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0], window_scale=ws[(0, 1)])
        if self.use_structure:
            feats_s2 = feats_s2 + self.structure1(feats_s2, points_list[1], data_dict['structure']['1'])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1], window_scale=ws[(1, 1)])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1], window_scale=ws[(1, 1)])
        if self.propagate_density:
            if self.use_mdb:
                feats_s2 = feats_s2 + self.mdb1(feats_s2, points_list, neighbors_list, subsampling_list, upsampling_list, 1, ws=ws)
            if self.use_film:
                feats_s2 = self.film1(feats_s2, cond1)

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1], window_scale=ws[(1, 2)])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2], window_scale=ws[(2, 2)])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2], window_scale=ws[(2, 2)])
        if self.propagate_density:
            if self.use_mdb:
                feats_s3 = feats_s3 + self.mdb2(feats_s3, points_list, neighbors_list, subsampling_list, upsampling_list, 2, ws=ws)
            if self.use_film:
                feats_s3 = self.film2(feats_s3, cond2)

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2], window_scale=ws[(2, 3)])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3], window_scale=ws[(3, 3)])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3], window_scale=ws[(3, 3)])

        # DRE at the bottleneck level P3.
        if self.use_mdb:
            feats_s4 = feats_s4 + self.mdb3(feats_s4, points_list, neighbors_list, subsampling_list, upsampling_list, 3, ws=ws)
        if self.use_film:
            feats_s4 = self.film3(feats_s4, cond3)

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        # DRE at level P2 before the decoder skip.
        if self.use_mdb and not self.propagate_density:
            feats_s3 = feats_s3 + self.mdb2(feats_s3, points_list, neighbors_list, subsampling_list, upsampling_list, 2, ws=ws)
        if self.use_film and not self.propagate_density:
            feats_s3 = self.film2(feats_s3, cond2)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        # DRE at level P1 before the decoder skip.
        if self.use_mdb and not self.propagate_density:
            feats_s2 = feats_s2 + self.mdb1(feats_s2, points_list, neighbors_list, subsampling_list, upsampling_list, 1, ws=ws)
        if self.use_film and not self.propagate_density:
            feats_s2 = self.film1(feats_s2, cond1)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        # FiLM on the dense features before fine matching (reuses the P1 condition).
        if self.use_film:
            latent_s2 = self.film_f(latent_s2, cond1)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list
