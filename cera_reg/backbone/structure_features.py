"""Bounded multi-radius geometry adapter and its CPU collate preprocessing.

Coordinates/grids stay unchanged. Neighbor searches are per cloud, contain no
GT, and run in data workers. Network aggregation is chunked on the input device.
"""
from functools import partial

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from geotransformer.utils.data import registration_collate_fn_stack_mode


def prepare_structure_data(data, radius, scales=(1., 2., 4.), levels=(0, 1), neighbor_limit=32):
    """Attach per-level bounded candidates + uncapped nonzero-distance counts.

    The returned indices use the usual global sentinel N. All queries/sources
    within a segment belong to that same cloud, even for coincident coordinates.
    """
    if neighbor_limit < 1 or not scales or min(scales) <= 0 or radius <= 0:
        raise ValueError('positive radius, scales and neighbor_limit are required')
    if sorted(scales) != list(scales):
        raise ValueError('scales must be sorted so the first branch estimates local support')
    structures = {}
    for level in levels:
        points = data['points'][level].detach().cpu().numpy().astype(np.float64)
        lengths = data['lengths'][level].detach().cpu().tolist()
        total = len(points)
        radii = np.asarray(scales) * radius * 2 ** level
        all_indices, all_counts = [], []
        offset = 0
        for length in lengths:
            cloud = points[offset:offset + length]
            if length == 0:
                all_indices.append(np.empty((0, len(radii), neighbor_limit), dtype=np.int64))
                all_counts.append(np.empty((0, len(radii)), dtype=np.int64))
                continue
            tree = cKDTree(cloud)
            indices = np.full((length, len(radii), neighbor_limit), total, dtype=np.int64)
            counts = np.zeros((length, len(radii)), dtype=np.int64)
            # Per-radius selection is important: taking the same K nearest
            # candidates at every radius collapses large-radius dense branches
            # into a tiny physical patch. Keep close supports plus radial
            # coverage. Stream ball results rather than allocate N x N or all
            # neighborhoods at once. No Cartesian sectors / frame are used.
            for scale_id, r in enumerate(radii):
                for start in range(0, length, 512):
                    neighbors = tree.query_ball_point(cloud[start:start + 512], r, workers=1)
                    for local_id, neighbors_i in enumerate(neighbors):
                        query_id = start + local_id
                        candidates = np.asarray(neighbors_i, dtype=np.int64)
                        squared = np.sum((cloud[candidates] - cloud[query_id]) ** 2, axis=1)
                        good = squared > 1e-16
                        candidates, squared = candidates[good], squared[good]
                        counts[query_id, scale_id] = len(candidates)
                        if len(candidates) > neighbor_limit:
                            order = np.argsort(squared, kind='stable')
                            close = max(1, neighbor_limit // 2)
                            coverage = np.linspace(close, len(order) - 1,
                                                   neighbor_limit - close, dtype=int)
                            chosen = np.concatenate([order[:close], order[coverage]])
                            candidates = candidates[chosen]
                        indices[query_id, scale_id, :len(candidates)] = candidates + offset
            all_indices.append(indices)
            all_counts.append(counts.astype(np.int64))
            offset += length
        if offset != total:
            raise ValueError('points/lengths do not describe the same cloud segments')
        structures[str(level)] = {
            'indices': torch.from_numpy(np.concatenate(all_indices)),
            'counts': torch.from_numpy(np.concatenate(all_counts)),
            'radii': torch.tensor(radii, dtype=torch.float32),
        }
    data['structure'] = structures
    return data


def structure_collate(data_dicts, *args, base_collate=registration_collate_fn_stack_mode,
                      structure_radius, structure_scales, structure_levels, structure_limit, **kwargs):
    data = base_collate(data_dicts, *args, **kwargs)
    if isinstance(data.get('points'), list):
        prepare_structure_data(data, structure_radius, structure_scales, structure_levels, structure_limit)
    return data


def make_structure_collate(cfg, base_collate=registration_collate_fn_stack_mode):
    """Wrap a normal/adaptive collate only when the experimental adapter is on."""
    settings = getattr(cfg, 'structure', None)
    if settings is None or not settings.enabled:
        return base_collate
    return partial(structure_collate, base_collate=base_collate,
                   structure_radius=cfg.backbone.init_radius,
                   structure_scales=settings.scales, structure_levels=settings.levels,
                   structure_limit=settings.neighbor_limit)


class MultiRadiusGeometryAdapter(nn.Module):
    """Rotation-invariant geometry auxiliary path with zero output at init.

    The pretrained KPConv is a separate residual path. Optional inverse support
    weighting and learned reliability gates are independent ablation switches.
    Normalization uses explicit valid masks, never padded or repeated points.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_scales=3,
                 chunk_size=1024, density_weighting=False, reliability_gate=True):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_scales = num_scales
        self.density_weighting = density_weighting
        self.reliability_gate = reliability_gate
        self.feature_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.geometry_mlp = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(7, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_stats = {}

    def forward(self, features, points, structure):
        indices, counts = structure['indices'], structure['counts']
        radii = structure['radii'].to(points.device, points.dtype)
        if len(radii) != self.num_scales:
            raise ValueError('adapter and preprocessing scale counts differ')
        if indices.shape[:2] != (len(points), self.num_scales) or counts.shape != (len(points), self.num_scales):
            raise ValueError('structure data do not align with this grid')
        n = len(points)
        point_pad = torch.cat([points, points.new_zeros(1, 3)])
        projected = self.feature_proj(features)
        feat_pad = torch.cat([projected, projected.new_zeros(1, projected.shape[1])])
        density = (1.0 / counts[:, 0].clamp(min=1).to(features.dtype)).clamp(min=1/32, max=1)
        density_pad = torch.cat([density, density.new_zeros(1)])
        outputs, gate_total = [], features.new_zeros(self.num_scales)
        amplitude = features.new_zeros(())
        empty_count = features.new_zeros(())
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            branch_outputs, reliabilities, available = [], [], []
            for scale_id, r in enumerate(radii):
                ids = indices[start:end, scale_id]
                offsets = point_pad[ids] - points[start:end, None]
                distances = offsets.norm(dim=-1)
                valid = (ids < n) & (distances > 1e-8)
                unit = offsets / distances.clamp(min=1e-8)[..., None]
                neighbor_features = feat_pad[ids]
                mask = valid & (distances <= r)
                support = mask.sum(dim=1)
                weights = mask.to(features.dtype)
                if self.density_weighting:
                    weights = weights * density_pad[ids]
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                # Direction second moment gives mean squared pairwise cosine
                # with O(N*K*3^2), without an N*K*K tensor or a fragile normal.
                with torch.no_grad():
                    uniform = mask.to(points.dtype) / support.clamp(min=1)[:, None]
                    moment = torch.einsum('bk,bki,bkj->bij', uniform, unit, unit)
                    angle = torch.einsum('bki,bij,bkj->bk', unit, moment, unit).clamp(0, 1)
                    radial = distances / r
                    geometry = torch.stack([radial, radial.square(), angle,
                                             radial * angle], dim=-1)
                    geometry = geometry.masked_fill(~mask[..., None], 0)
                    eigen = torch.linalg.eigvalsh(moment).flip(-1).clamp(min=0)
                    eigen_valid = support >= 3
                    linearity = (eigen[:, 0] - eigen[:, 1]) / eigen[:, 0].clamp(min=1e-8)
                    planarity = (eigen[:, 1] - eigen[:, 2]) / eigen[:, 0].clamp(min=1e-8)
                    nearest = distances.masked_fill(~mask, float('inf')).min(dim=1).values
                    nearest = torch.where(support > 0, nearest / r, torch.ones_like(nearest))
                    actual = counts[start:end, scale_id].to(features.dtype)
                    reliability = torch.stack([
                        torch.log1p(actual), torch.log1p(support.to(features.dtype)),
                        (actual > support).to(features.dtype), nearest,
                        linearity * eigen_valid, planarity * eigen_valid,
                        eigen_valid.to(features.dtype)], dim=1)
                values = neighbor_features + self.geometry_mlp(geometry)
                branch_outputs.append((values * weights[..., None]).sum(dim=1))
                reliabilities.append(reliability)
                available.append(support > 0)
            branch_outputs = torch.stack(branch_outputs, dim=1)
            available = torch.stack(available, dim=1)
            if self.reliability_gate:
                logits = self.gate(torch.stack(reliabilities, dim=1)).squeeze(-1)
            else:
                logits = features.new_zeros(end - start, self.num_scales)
            # Softmax all-masked rows is undefined; output exactly zero instead.
            logits = logits.masked_fill(~available, -1e4)
            gates = logits.softmax(dim=1) * available.to(features.dtype)
            gates = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
            fused = (branch_outputs * gates[..., None]).sum(dim=1)
            update = self.output(fused)
            update = update * available.any(dim=1, keepdim=True).to(update.dtype)
            outputs.append(update)
            with torch.no_grad():
                gate_total += gates.sum(dim=0)
                amplitude += update.abs().sum()
                empty_count += (~available.any(dim=1)).sum()
        if not n:
            return features.new_zeros(0, self.output.out_features)
        self.last_stats = {'gate_mean': (gate_total / n).detach(),
                           'update_abs_mean': (amplitude / (n * self.output.out_features)).detach(),
                           'no_support_fraction': (empty_count / n).detach()}
        return torch.cat(outputs)
