"""Density-adaptive KPConv: per-query window dilation for cross-source support.

Motivation (3DCSR kinect-lidar, measured on the val split):
  - lidar-side level-0 spacing is ~2.5x the kinect side (median NN 0.0415 vs
    0.0201), so within the fixed level radius (0.0625) the median lidar point
    sees only ~3 support neighbours vs ~10 on the dense side, and in the worst
    scenes the nearest neighbour falls outside sigma (0.05), where the KPConv
    kernel weight clamp(1 - d / sigma, min=0) vanishes -> starved features.
  - The pretrained encoder expects the 3DMatch spacing distribution
    (rho_ref = [0.0201, 0.0338, 0.0594, 0.1124] per level).

Fix: dilate the convolution window of a cloud by alpha = rho_cloud / rho_ref
(per level, per cloud). Key identity: dividing the relative neighbour offsets
by alpha is EXACTLY equivalent to scaling kernel_points and sigma by alpha,
    |(n - q)/a - kp| / sigma == |n - q - a*kp| / (a*sigma),
so the dilation is a one-line change inside KPConv.forward and the module keeps
an identical state_dict (warm-load compatible).

The neighbour CANDIDATE lists themselves are produced by the adaptive
precompute in data_adaptive.py (search radius alpha * base per cloud); without
that pipeline the dilation alone cannot add new support points.

All block classes mirror geotransformer.modules.kpconv.modules with identical
attribute names, so pretrained checkpoints load unchanged.
"""
import torch
import torch.nn as nn

from geotransformer.modules.kpconv.kpconv import KPConv
from geotransformer.modules.kpconv.modules import GroupNorm, UnaryBlock
from geotransformer.modules.kpconv.functional import maxpool
from geotransformer.modules.ops import index_select


class DensityAdaptiveKPConv(KPConv):
    """KPConv whose convolution window is dilated per query point.

    forward(..., window_scale): (M,) or (M, 1) tensor of per-query dilations.
    window_scale=None (or all ones) reproduces the standard KPConv exactly.
    """

    def forward(self, s_feats, q_points, s_points, neighbor_indices, window_scale=None):
        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N, 3) -> (N+1, 3)
        neighbors = index_select(s_points, neighbor_indices, dim=0)  # (N+1, 3) -> (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # Density-adaptive window dilation (see module docstring for the proof).
        if window_scale is not None:
            scale = window_scale
            if scale.dim() == 2:
                scale = scale.squeeze(1)
            scale = scale.view(-1, 1, 1).to(neighbors.dtype)
            neighbors = neighbors / scale

        # Get Kernel point influences
        neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
        differences = neighbors - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
        sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

        # apply neighbor weights
        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)  # (N+1, C) -> (M, H, C)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

        # apply convolutional weights
        weighted_feats = weighted_feats.permute(1, 0, 2)  # (M, K, C) -> (K, M, C)
        kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, C_out) -> (K, M, C_out)
        output_feats = torch.sum(kernel_outputs, dim=0, keepdim=False)  # (K, M, C_out) -> (M, C_out)

        # normalization
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats


class DAConvBlock(nn.Module):
    """ConvBlock with a density-adaptive KPConv (identical state_dict layout)."""

    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, group_norm,
                 negative_slope=0.1, bias=True, layer_norm=False):
        super(DAConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.KPConv = DensityAdaptiveKPConv(in_channels, out_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, s_feats, q_points, s_points, neighbor_indices, window_scale=None):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices, window_scale=window_scale)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class DAResidualBlock(nn.Module):
    """ResidualBlock with a density-adaptive KPConv (identical state_dict layout)."""

    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, group_norm,
                 strided=False, bias=True, layer_norm=False):
        super(DAResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias, layer_norm=layer_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = DensityAdaptiveKPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm_conv = nn.LayerNorm(mid_channels)
        else:
            self.norm_conv = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(mid_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm)

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(in_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm)
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices, window_scale=None):
        x = self.unary1(s_feats)

        x = self.KPConv(x, q_points, s_points, neighbor_indices, window_scale=window_scale)
        x = self.norm_conv(x)
        x = self.leaky_relu(x)

        x = self.unary2(x)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)

        return x
