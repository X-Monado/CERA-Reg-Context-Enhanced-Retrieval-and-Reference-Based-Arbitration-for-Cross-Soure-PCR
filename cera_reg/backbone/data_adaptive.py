"""Adaptive data pipeline for the density-adaptive KPConv window (DA-window).

Drop-in replacement of geotransformer.utils.data.precompute_data_stack_mode /
registration_collate_fn_stack_mode that computes per-cloud, per-level window
scales alpha = rho_cloud / rho_ref (clamped to [1, alpha_max]) and searches
neighbours at the DILATED radius alpha * base_radius for sparse clouds.

Grid construction (grid_subsample) is identical to the standard pipeline; only
the radius searches adapt. Dense clouds (alpha <= alpha_gate, e.g. the kinect
side with median spacing 0.021 ~ rho_ref) keep the exact standard neighbour
lists, so the DA pipeline is bit-compatible with the pretrained encoder for
in-distribution clouds.

Outputs add one key used by the DRE backbone:
  'cloud_scales': FloatTensor (num_stages, num_clouds) of alpha values.

Measured 3DMatch-train reference spacings (see experiments log):
  rho_ref = [0.0201, 0.0338, 0.0594, 0.1124] at levels [0, 1, 2, 3]
  (voxel 0.025; median NN distance per level, 8 train clouds).
"""
import numpy as np
import torch

from geotransformer.modules.ops import grid_subsample, radius_search


# Measured 3DMatch train per-level median NN spacing (voxel 0.025 pipeline).
RHO_REF_0025 = [0.0201, 0.0338, 0.0594, 0.1124]


def _probe_nn_median(p_c, n_c, r, k=2):
    """Median distance to the nearest non-self neighbour within radius r.

    Returns None when fewer than 50% of points have any neighbour. The ext
    radius_search pads the returned matrix to the ACTUAL max neighbour count,
    so width 1 means self-only (no neighbour at all) and column 1 would be
    out of bounds.
    """
    len_c = torch.LongTensor([n_c])
    nb2 = radius_search(p_c, p_c, len_c, len_c, r, k)
    if nb2.shape[1] < 2:
        return None
    valid = nb2[:, 1] < n_c
    if float(valid.float().mean()) < 0.5:
        return None
    d = torch.norm(p_c[nb2[valid, 1]] - p_c[valid], dim=1)
    return float(d.median())


def _adaptive_radius_neighbors(p_c, n_c, base_radius, limit, rho_ref_level, alpha_max, alpha_gate):
    """Per-cloud neighbour search with adaptive window.

    Steps:
      1. standard search at base_radius (reused when alpha resolves to 1);
      2. spacing probe from column 1 of the neighbour list (nearest non-self
         neighbour), escalated x2 / x4 for starved clouds (which are small);
      3. if rho / rho_ref > alpha_gate: re-search at alpha * base_radius.

    Returns (neighbors (n_c, limit) LOCAL indices padded with n_c, alpha).
    """
    len_c = torch.LongTensor([n_c])
    nb = radius_search(p_c, p_c, len_c, len_c, base_radius, limit)
    if n_c < 4 or limit < 2:
        return nb, 1.0

    rho = None
    valid = nb[:, 1] < n_c if nb.shape[1] >= 2 else torch.zeros(n_c, dtype=torch.bool)
    if nb.shape[1] >= 2 and float(valid.float().mean()) >= 0.5:
        d = torch.norm(p_c[nb[valid, 1]] - p_c[valid], dim=1)
        rho = float(d.median())
    else:
        # starved at base radius (possibly zero neighbours anywhere):
        # escalate the probe (starved clouds are small)
        r = base_radius * 2.0
        for _ in range(2):
            rho = _probe_nn_median(p_c, n_c, r)
            if rho is not None:
                break
            r *= 2.0

    if rho is None:
        # fully isolated cloud at any probe radius: keep the standard list
        return nb, 1.0
    alpha = rho / rho_ref_level
    if alpha <= alpha_gate:
        return nb, 1.0
    alpha = min(alpha, alpha_max)
    nb = radius_search(p_c, p_c, len_c, len_c, base_radius * alpha, limit)
    return nb, alpha


def _adaptive_radius_search_cross(q_c, s_c, radius, limit, n_s_c):
    """Cross radius search for one cloud segment; returns LOCAL support indices."""
    return radius_search(q_c, s_c, torch.LongTensor([q_c.shape[0]]), torch.LongTensor([n_s_c]), radius, limit)


def _restack(per_cloud, support_lengths, support_offsets, n_support_total):
    """Re-stack per-cloud LOCAL neighbour lists into global indices.

    per_cloud[c]: (n_q_c, w_c) with LOCAL support indices padded with n_s_c,
    where w_c <= limit (the ext pads each call to that call's max neighbour
    count, so widths can differ between clouds). Returns (N_q_total, max_w)
    with GLOBAL indices padded with n_support_total.
    """
    max_w = max(nb.shape[1] for nb in per_cloud)
    outs = []
    for nb, n_s, s_off in zip(per_cloud, support_lengths, support_offsets):
        if nb.shape[1] < max_w:
            pad = torch.full((nb.shape[0], max_w - nb.shape[1]), n_support_total, dtype=nb.dtype)
            nb = torch.cat([nb, pad], dim=1)
        valid = nb < n_s
        outs.append(torch.where(valid, nb + s_off, torch.full_like(nb, n_support_total)))
    return torch.cat(outs, dim=0)


def adaptive_precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits,
                                         alpha_max=4.0, alpha_gate=1.1, rho_ref=None):
    """Standard stack-mode precompute with per-cloud adaptive search radii.

    See module docstring. `points` are the already-voxelized level-0 points of
    all clouds (stacked), `lengths` their per-cloud sizes.
    """
    assert num_stages == len(neighbor_limits)
    if rho_ref is None:
        rho_ref = RHO_REF_0025
    assert len(rho_ref) >= num_stages
    num_clouds = int(lengths.shape[0])

    points_list = []
    lengths_list = []
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []
    cloud_scales = torch.ones(num_stages, num_clouds)

    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]
        cur_list = cur_lengths.tolist()
        n_cur_total = cur_points.shape[0]

        # --- neighbours at level i: radius = base * alpha_{c,i} ---
        per_cloud = []
        alphas = []
        offset = 0
        for c in range(num_clouds):
            n_c = int(cur_list[c])
            p_c = cur_points[offset:offset + n_c]
            nb_c, alpha_c = _adaptive_radius_neighbors(
                p_c, n_c, radius, neighbor_limits[i], rho_ref[i], alpha_max, alpha_gate)
            per_cloud.append(nb_c)
            alphas.append(alpha_c)
            offset += n_c
        neighbors_list.append(_restack(per_cloud, cur_list, _cum_offsets(cur_list), n_cur_total))
        cloud_scales[i] = torch.tensor(alphas)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]
            sub_list = sub_lengths.tolist()
            n_sub_total = sub_points.shape[0]

            # --- subsampling: queries = level i+1, support = level i ---
            per_cloud = []
            cur_off = 0
            sub_off = 0
            for c in range(num_clouds):
                n_cur, n_sub = int(cur_list[c]), int(sub_list[c])
                nb = _adaptive_radius_search_cross(
                    sub_points[sub_off:sub_off + n_sub], cur_points[cur_off:cur_off + n_cur],
                    radius * alphas[c], neighbor_limits[i], n_cur)
                per_cloud.append(nb)
                cur_off += n_cur
                sub_off += n_sub
            subsampling_list.append(_restack(per_cloud, cur_list, _cum_offsets(cur_list), n_cur_total))

            # --- upsampling: queries = level i, support = level i+1 ---
            # (upsample radius 2*base_i equals base_{i+1}; scale by alpha_{c,i}:
            # the sparse side's level-(i+1) grid is flat, so alpha_{c,i} already
            # reflects its spacing. nearest_upsample takes column 0, so this is
            # a defensive scaling.)
            per_cloud = []
            cur_off = 0
            sub_off = 0
            for c in range(num_clouds):
                n_cur, n_sub = int(cur_list[c]), int(sub_list[c])
                nb = _adaptive_radius_search_cross(
                    cur_points[cur_off:cur_off + n_cur], sub_points[sub_off:sub_off + n_sub],
                    radius * 2 * alphas[c], neighbor_limits[i + 1], n_sub)
                per_cloud.append(nb)
                cur_off += n_cur
                sub_off += n_sub
            upsampling_list.append(_restack(per_cloud, sub_list, _cum_offsets(sub_list), n_sub_total))

        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
        'cloud_scales': cloud_scales,
    }


def _cum_offsets(lengths_list):
    offs = []
    o = 0
    for n in lengths_list:
        offs.append(o)
        o += n
    return offs


def adaptive_registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True,
    alpha_max=4.0, alpha_gate=1.1, rho_ref=None
):
    """registration_collate_fn_stack_mode with adaptive per-cloud search radii."""
    batch_size = len(data_dicts)
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = adaptive_precompute_data_stack_mode(
            points, lengths, num_stages, voxel_size, search_radius, neighbor_limits,
            alpha_max=alpha_max, alpha_gate=alpha_gate, rho_ref=rho_ref)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict
