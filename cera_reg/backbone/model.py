import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.geotransformer import SuperPointTargetGenerator, LocalGlobalRegistration
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.utils.data import precompute_data_stack_mode
from geotransformer.modules.ops import radius_search

from .semantic_geotransformer import SemanticGeometricTransformer
from .otm_matching import (SemanticOneToManyMatching, SpatialConsistencyFilter,
                          JointSpectralConsistencyFilter, OneToManyRegistration,
                          MultiHypothesisRegistration)
from .dre_backbone import DREBackbone
from .structure_features import prepare_structure_data


class DCRFormer(nn.Module):
    def __init__(self, cfg):
        super(DCRFormer, self).__init__()
        self.cfg = cfg
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = DREBackbone(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
            use_mdb=cfg.dre.use_mdb,
            use_film=cfg.dre.use_film,
            film_hidden_dim=cfg.dre.film_hidden_dim,
            use_da_window=cfg.dre.use_da_window,
            propagate_density=cfg.dre.propagate_density,
            robust_density=cfg.dre.robust_density,
            structure_cfg=getattr(cfg, 'structure', None),
        )

        self.transformer = SemanticGeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            semantic_dim=256,  # projection kept for stage-4 warm-load compatibility; never activated
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SemanticOneToManyMatching(
            cfg.coarse_matching.num_correspondences,
            topk=cfg.otm_matching.topk,
            dual_normalization=cfg.coarse_matching.dual_normalization,
        )

        self.lgr = LocalGlobalRegistration(
            k=cfg.fine_matching.topk,
            acceptance_radius=cfg.otm_matching.acceptance_radius,
            mutual=True,
            confidence_threshold=cfg.otm_matching.confidence_threshold,
            use_dustbin=False,
            use_global_score=False,
            correspondence_threshold=cfg.otm_matching.correspondence_threshold,
            correspondence_limit=cfg.otm_matching.correspondence_limit,
            num_refinement_steps=cfg.otm_matching.num_refinement_steps,
        )

        # DCR-Former innovation 2: joint spectral strict selection (ablatable to the legacy SC/SC2 filter).
        self.use_joint_spectral_filter = cfg.otm_matching.use_joint_spectral_filter
        if self.use_joint_spectral_filter:
            self.spatial_filter = JointSpectralConsistencyFilter(
                spectral_sigma=cfg.otm_matching.spectral_sigma,
                feature_weight=cfg.otm_matching.spectral_feature_weight,
                num_iterations=cfg.otm_matching.spectral_iterations,
                sc2_topk=cfg.otm_matching.sc2_topk,
                threshold=cfg.otm_matching.spectral_threshold,
            )
        else:
            self.spatial_filter = SpatialConsistencyFilter(
                sc_topk=cfg.otm_matching.sc_topk,
                sc2_topk=cfg.otm_matching.sc2_topk,
                sc_threshold=cfg.otm_matching.sc_threshold,
            )

        self.otm_registration = OneToManyRegistration(
            acceptance_radius=cfg.otm_matching.acceptance_radius,
            correspondence_threshold=cfg.otm_matching.correspondence_threshold,
            correspondence_limit=cfg.otm_matching.correspondence_limit,
            num_refinement_steps=cfg.otm_matching.num_refinement_steps,
        )

        # DCR-Former innovation 3: multi-hypothesis pose estimation (ablatable to single-hypothesis).
        self.use_multi_hypothesis = cfg.otm_matching.use_multi_hypothesis
        if self.use_multi_hypothesis:
            self.mh_registration = MultiHypothesisRegistration(
                acceptance_radius=cfg.otm_matching.acceptance_radius,
                num_hypotheses=cfg.otm_matching.num_hypotheses,
                subset_ratio=cfg.otm_matching.subset_ratio,
                chamfer_radius=cfg.otm_matching.chamfer_radius,
                correspondence_threshold=cfg.otm_matching.correspondence_threshold,
                num_refinement_steps=cfg.otm_matching.num_refinement_steps,
                semantic_weight=cfg.otm_matching.hypothesis_semantic_weight,
            )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        # Density consistency regularization (training only).
        self.weight_consistency_loss = cfg.loss.weight_consistency_loss
        self.consistency_prob = cfg.loss.consistency_prob
        self.consistency_min_points = cfg.train.density_min_points

    @staticmethod
    def _move_nested(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return [DCRFormer._move_nested(o, device) for o in obj]
        if isinstance(obj, dict):
            return {k: DCRFormer._move_nested(v, device) for k, v in obj.items()}
        return obj

    @staticmethod
    def _gather_corr_feats(corr_points, points_f, feats_f, block=2048):
        """Feature lookup for correspondence points via chunked nearest-neighbour search.

        LGR correspondences are drawn from the fine grid, so an exact (zero-distance)
        match always exists; chunking keeps the cdist memory bounded.
        """
        num = corr_points.shape[0]
        if num == 0:
            return feats_f.new_zeros(0, feats_f.shape[1])
        indices = []
        for start in range(0, num, block):
            dist = torch.cdist(corr_points[start:start + block], points_f)
            indices.append(dist.argmin(dim=1))
        return feats_f[torch.cat(indices)]

    def _density_consistency(self, feats_list, data_dict):
        """Density-invariance regularization on fine features (DCR-Former).

        Re-encodes a randomly thinned copy of the reference cloud through the
        backbone and pulls its P1-level features towards the original ones via
        nearest-neighbor correspondences. Only one extra backbone forward is
        introduced; both views receive gradients.
        """
        device = data_dict['points'][0].device
        ref_length = data_dict['lengths'][0][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_points0 = data_dict['points'][0][:ref_length]
        ref_points1 = data_dict['points'][1][:ref_length_f]
        ref_feats_f = feats_list[0][:ref_length_f]

        # Randomly thin the reference: density ratio in [1/8, 1].
        ratio = 2.0 ** torch.empty(1).uniform_(-3.0, 0.0).item()
        n_target = int(ref_length * ratio)
        if n_target < self.consistency_min_points:
            return ref_feats_f.new_zeros(())

        indices = torch.randperm(ref_length, device=device)[:n_target]
        q_points0 = ref_points0[indices].cpu()

        # Build the multiscale grid for the thinned cloud (grid ops run on CPU).
        neighbor_limits = [nb.shape[1] for nb in data_dict['neighbors']]
        q_data = precompute_data_stack_mode(
            q_points0,
            torch.tensor([n_target], dtype=torch.long),
            num_stages=len(data_dict['points']),
            voxel_size=self.cfg.backbone.init_voxel_size,
            radius=self.cfg.backbone.init_radius,
            neighbor_limits=neighbor_limits,
        )
        if self.backbone.use_structure:
            settings = self.cfg.structure
            prepare_structure_data(q_data, self.cfg.backbone.init_radius,
                                   settings.scales, settings.levels, settings.neighbor_limit)
        q_data = self._move_nested(q_data, device)

        q_feats = torch.ones(n_target, 1, device=device)
        q_feats_list = self.backbone(q_feats, q_data)
        q_feats_f = q_feats_list[0]
        q_points_f = q_data['points'][1]

        # Nearest-neighbor alignment from the thinned grid back to the original P1 grid.
        nn_indices = radius_search(
            q_points_f.cpu(), ref_points1.cpu(),
            torch.tensor([q_points_f.shape[0]], dtype=torch.long),
            torch.tensor([ref_length_f], dtype=torch.long),
            self.cfg.backbone.init_voxel_size * 2.0,
            1,
        )
        corr = nn_indices[:, 0].to(device)
        valid = corr < ref_length_f
        if valid.sum() < 32:
            return ref_feats_f.new_zeros(())

        q_vec = F.normalize(q_feats_f[valid], p=2, dim=1)
        r_vec = F.normalize(ref_feats_f[corr[valid]], p=2, dim=1)
        loss = (1.0 - (q_vec * r_vec).sum(dim=1)).mean()
        return loss

    def forward(self, data_dict):
        output_dict = {}

        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        ref_point_to_node, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        src_point_to_node, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c, src_points_c,
            ref_node_knn_points, src_node_knn_points,
            transform, self.matching_radius,
            ref_masks=ref_node_masks, src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 1. KPConv Geometric Features
        feats = data_dict['features'].detach()
        feats_list = self.backbone(feats, data_dict)
        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]

        # 1b. Density consistency regularization (training only, ~consistency_prob of batches).
        if self.training and self.weight_consistency_loss > 0 and torch.rand(1).item() < self.consistency_prob:
            output_dict['consistency_loss'] = self._density_consistency(feats_list, data_dict)

        # 2. Geometric Transformer (pure geometric path; semantic bypass interface reserved).
        ref_feats_c_input = ref_feats_c.unsqueeze(0)
        src_feats_c_input = src_feats_c.unsqueeze(0)

        ref_feats_c_enhanced, src_feats_c_enhanced = self.transformer(
            ref_points_c.unsqueeze(0), src_points_c.unsqueeze(0),
            ref_feats_c_input, src_feats_c_input,
        )
        ref_feats_c_norm = F.normalize(ref_feats_c_enhanced.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c_enhanced.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 4. Coarse Matching
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 5. Assemble patch correspondences
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 6. Patch-level Optimal Transport (for training loss)
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        output_dict['matching_scores'] = matching_scores

        # 7. Registration (different for training vs inference)
        with torch.no_grad():
            matching_scores_clean = matching_scores[:, :-1, :-1]
            if self.training:
                ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.lgr(
                    ref_node_corr_knn_points, src_node_corr_knn_points,
                    ref_node_corr_knn_masks, src_node_corr_knn_masks,
                    matching_scores_clean, node_corr_scores,
                )
            else:
                ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.lgr(
                    ref_node_corr_knn_points, src_node_corr_knn_points,
                    ref_node_corr_knn_masks, src_node_corr_knn_masks,
                    matching_scores_clean, node_corr_scores,
                )
                if ref_corr_points.shape[0] > 3:
                    if self.use_joint_spectral_filter:
                        ref_corr_feats = self._gather_corr_feats(ref_corr_points, ref_points_f, ref_feats_f)
                        src_corr_feats = self._gather_corr_feats(src_corr_points, src_points_f, src_feats_f)
                        ref_corr_points, src_corr_points, corr_scores, _ = self.spatial_filter(
                            ref_corr_points, src_corr_points, corr_scores, ref_corr_feats, src_corr_feats
                        )
                    else:
                        ref_corr_points, src_corr_points, corr_scores, _ = self.spatial_filter(
                            ref_corr_points, src_corr_points, corr_scores
                        )
                    if ref_corr_points.shape[0] >= 3:
                        if self.use_multi_hypothesis:
                            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.mh_registration(
                                ref_corr_points, src_corr_points, corr_scores,
                                ref_points_c=ref_points_c, src_points_c=src_points_c,
                            )
                        else:
                            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.otm_registration(
                                ref_corr_points, src_corr_points, corr_scores
                            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(config):
    model = DCRFormer(config)
    return model


def feature_checkpoint_config(cfg):
    """Serializable architecture + neighborhood protocol for new checkpoints."""
    return {'version': 1, 'dre': dict(cfg.dre), 'structure': dict(cfg.structure),
            'backbone': dict(cfg.backbone)}


def restore_feature_checkpoint_config(cfg, checkpoint):
    """Restore saved feature choices before creating the model/data loader."""
    settings = checkpoint.get('feature_config')
    if settings is not None:
        if settings.get('version') != 1:
            raise ValueError('unsupported feature checkpoint protocol version')
        cfg.dre.update(settings['dre'])
        cfg.structure.update(settings['structure'])
        cfg.backbone.update(settings['backbone'])
    elif any(key.startswith('backbone.structure') for key in checkpoint['model']):
        raise ValueError('structure checkpoint is missing feature_config; '
                         'its neighborhood/gating protocol cannot be inferred from tensors')
    return cfg


def load_compatible_model_state(model, state_dict, allow_new_structure=False):
    """Load DCR snapshots across zero-initialized feature additions.

    Only old snapshots missing film0 are accepted. Other missing or unexpected
    parameters still fail loudly, so a mismatched transformer is not hidden.
    """
    result = model.load_state_dict(state_dict, strict=False)
    allowed = {key for key in model.state_dict() if key.startswith('backbone.film0.')}
    if allow_new_structure:
        # Only explicit training initialization may create new zero-output
        # adapters. Evaluation must not silently omit a trained feature module.
        allowed |= {key for key in model.state_dict()
                    if key.startswith(('backbone.structure0.', 'backbone.structure1.'))}
    missing = set(result.missing_keys)
    if missing - allowed or result.unexpected_keys:
        raise RuntimeError(
            'Incompatible checkpoint: missing={}, unexpected={}'.format(
                sorted(missing), result.unexpected_keys
            )
        )
    return sorted(missing)


def load_geotransformer_pretrain(model, state_dict):
    """Warm-load the official GeoTransformer geometric weights into DCR-Former.

    The official transformer has one extra ``transformer.`` nesting level.
    Density modules retain their zero-identity initialization.
    """
    target = model.state_dict()
    loaded = set()
    for key, value in state_dict.items():
        mapped = key
        if key.startswith('transformer.transformer.'):
            mapped = 'transformer.' + key[len('transformer.transformer.'):]
        if mapped in target and target[mapped].shape == value.shape:
            target[mapped] = value
            loaded.add(mapped)
    backbone_count = sum(key.startswith('backbone.') for key in loaded)
    transformer_count = sum(key.startswith('transformer.') for key in loaded)
    if backbone_count < 100 or transformer_count < 80:
        raise RuntimeError('Incomplete GeoTransformer pretrain: backbone={}, transformer={}'.format(
            backbone_count, transformer_count))
    model.load_state_dict(target)
    return {'backbone': backbone_count, 'transformer': transformer_count, 'total': len(loaded)}


def main():
    from .config import make_cfg
    cfg = make_cfg()
    model = create_model(cfg)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')


if __name__ == '__main__':
    main()
