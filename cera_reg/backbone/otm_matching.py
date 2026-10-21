import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.registration.procrustes import weighted_procrustes


class SemanticOneToManyMatching(nn.Module):
    def __init__(self, num_correspondences, topk=3, dual_normalization=True):
        super(SemanticOneToManyMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.topk = topk
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool, device=ref_feats.device)
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool, device=src_feats.device)

        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]

        feat_dists = pairwise_distance(ref_feats, src_feats, normalized=True)
        matching_scores = torch.exp(-feat_dists)

        if self.dual_normalization:
            ref_matching_scores = matching_scores / (matching_scores.sum(dim=1, keepdim=True) + 1e-8)
            src_matching_scores = matching_scores / (matching_scores.sum(dim=0, keepdim=True) + 1e-8)
            matching_scores = ref_matching_scores * src_matching_scores

        k = min(self.topk, matching_scores.shape[1])
        topk_scores, topk_src_local = matching_scores.topk(k=k, dim=1)

        ref_corr_indices = []
        src_corr_indices = []
        corr_scores = []

        for i in range(k):
            ref_sel = torch.arange(topk_scores.shape[0], device=ref_feats.device)
            src_sel = topk_src_local[:, i]
            score_sel = topk_scores[:, i]

            mask = score_sel > 0
            ref_corr_indices.append(ref_sel[mask])
            src_corr_indices.append(src_sel[mask])
            corr_scores.append(score_sel[mask])

        ref_corr_indices = torch.cat(ref_corr_indices)
        src_corr_indices = torch.cat(src_corr_indices)
        corr_scores = torch.cat(corr_scores)

        num_correspondences = min(self.num_correspondences, corr_scores.shape[0])
        topk_idx = corr_scores.topk(k=num_correspondences).indices
        ref_corr_indices = ref_corr_indices[topk_idx]
        src_corr_indices = src_corr_indices[topk_idx]
        corr_scores = corr_scores[topk_idx]

        ref_corr_indices = ref_indices[ref_corr_indices]
        src_corr_indices = src_indices[src_corr_indices]

        return ref_corr_indices, src_corr_indices, corr_scores


class SpatialConsistencyFilter(nn.Module):
    def __init__(self, sc_topk=6, sc2_topk=6, sc_threshold=0.5):
        super(SpatialConsistencyFilter, self).__init__()
        self.sc_topk = sc_topk
        self.sc2_topk = sc2_topk
        self.sc_threshold = sc_threshold

    @torch.no_grad()
    def _compute_sc(self, ref_corr_points, src_corr_points):
        num_corrs = ref_corr_points.shape[0]
        if num_corrs <= self.sc_topk:
            return torch.ones(num_corrs, device=ref_corr_points.device)

        ref_dist = torch.cdist(ref_corr_points, ref_corr_points)
        src_dist = torch.cdist(src_corr_points, src_corr_points)

        _, ref_knn = ref_dist.topk(k=min(self.sc_topk, num_corrs), dim=1, largest=False)
        _, src_knn = src_dist.topk(k=min(self.sc_topk, num_corrs), dim=1, largest=False)

        consistency = torch.zeros(num_corrs, device=ref_corr_points.device)
        for i in range(num_corrs):
            ref_neighbors = ref_knn[i, 1:]
            src_neighbors = src_knn[i, 1:]
            common = torch.tensor(
                list(set(ref_neighbors.tolist()) & set(src_neighbors.tolist())),
                device=ref_corr_points.device, dtype=torch.long,
            )
            if common.shape[0] > 1:
                ref_d = ref_dist[i, common]
                src_d = src_dist[i, common]
                d_ratio = torch.min(ref_d, src_d) / (torch.max(ref_d, src_d) + 1e-8)
                consistency[i] = d_ratio.mean()
            else:
                consistency[i] = 0.0

        return consistency

    @torch.no_grad()
    def _compute_sc2(self, ref_corr_points, src_corr_points, sc_scores):
        num_corrs = ref_corr_points.shape[0]
        if num_corrs <= self.sc2_topk:
            return sc_scores

        ref_dist = torch.cdist(ref_corr_points, ref_corr_points)
        src_dist = torch.cdist(src_corr_points, src_corr_points)

        _, ref_knn = ref_dist.topk(k=min(self.sc2_topk, num_corrs), dim=1, largest=False)
        _, src_knn = src_dist.topk(k=min(self.sc2_topk, num_corrs), dim=1, largest=False)

        sc2_scores = torch.zeros(num_corrs, device=ref_corr_points.device)
        for i in range(num_corrs):
            ref_neighbors = ref_knn[i, 1:]
            src_neighbors = src_knn[i, 1:]
            common = torch.tensor(
                list(set(ref_neighbors.tolist()) & set(src_neighbors.tolist())),
                device=ref_corr_points.device, dtype=torch.long,
            )
            if common.shape[0] > 0:
                sc2_scores[i] = sc_scores[common].mean()
            else:
                sc2_scores[i] = 0.0

        return sc2_scores

    @torch.no_grad()
    def forward(self, ref_corr_points, src_corr_points, corr_scores):
        num_corrs = ref_corr_points.shape[0]
        if num_corrs < 3:
            return ref_corr_points, src_corr_points, corr_scores, torch.arange(num_corrs, device=ref_corr_points.device)

        sc_scores = self._compute_sc(ref_corr_points, src_corr_points)
        sc2_scores = self._compute_sc2(ref_corr_points, src_corr_points, sc_scores)

        final_scores = sc_scores * sc2_scores
        mask = final_scores > self.sc_threshold

        if mask.sum() < 3:
            _, topk_idx = corr_scores.topk(k=min(100, num_corrs))
            mask = torch.zeros(num_corrs, dtype=torch.bool, device=ref_corr_points.device)
            mask[topk_idx] = True

        ref_corr_points = ref_corr_points[mask]
        src_corr_points = src_corr_points[mask]
        corr_scores = corr_scores[mask]
        corr_indices = torch.where(mask)[0]

        return ref_corr_points, src_corr_points, corr_scores, corr_indices


class JointSpectralConsistencyFilter(nn.Module):
    """Spectral strict selection with joint feature-distance consistency (DCR-Former innovation 2).

    A TRIM-style spectral matcher. The joint affinity matrix combines
      (a) pairwise distance consistency between the two clouds (geometric term),
      (b) cross-side feature-neighbourhood consistency (joint feature term),
      (c) per-correspondence feature similarity (diagonal prior).
    The leading eigenvector obtained via power iteration gives the membership of
    the dominant consistent cluster, refined by a vectorised SC2 second-order
    aggregation over reciprocal kNN neighbourhoods. Surviving weights combine the
    OT matching scores with the joint consistency scores.
    """

    def __init__(self, spectral_sigma=0.1, feature_weight=0.5, num_iterations=50,
                 sc2_topk=6, threshold=0.25, topk_fallback=100, min_correspondences=3):
        super(JointSpectralConsistencyFilter, self).__init__()
        self.spectral_sigma = spectral_sigma
        self.feature_weight = feature_weight
        self.num_iterations = num_iterations
        self.sc2_topk = sc2_topk
        self.threshold = threshold
        self.topk_fallback = topk_fallback
        self.min_correspondences = min_correspondences

    @staticmethod
    def _knn_mask(dist, k):
        """[N,N] boolean mask: mask[i, j] = j among the k nearest neighbours of i (self excluded)."""
        num = dist.shape[0]
        mask = torch.zeros_like(dist, dtype=torch.bool)
        k = min(k, num - 1)
        if k <= 0:
            return mask
        _, knn_idx = dist.topk(k=k + 1, dim=1, largest=False)  # +1 to skip the self entry.
        mask.scatter_(1, knn_idx, True)
        mask.fill_diagonal_(False)
        return mask

    @torch.no_grad()
    def forward(self, ref_corr_points, src_corr_points, corr_scores,
                ref_corr_feats=None, src_corr_feats=None):
        num_corrs = ref_corr_points.shape[0]
        if num_corrs < self.min_correspondences:
            return (ref_corr_points, src_corr_points, corr_scores,
                    torch.arange(num_corrs, device=ref_corr_points.device))

        device = ref_corr_points.device
        ref_dist = torch.cdist(ref_corr_points, ref_corr_points)
        src_dist = torch.cdist(src_corr_points, src_corr_points)

        # (a) Geometric term: exp affinity on pairwise distance discrepancy.
        affinity = torch.exp(-(ref_dist - src_dist).abs() / self.spectral_sigma)

        # (b)+(c) Joint feature terms.
        if ref_corr_feats is not None and src_corr_feats is not None and self.feature_weight > 0:
            ref_corr_feats = F.normalize(ref_corr_feats, p=2, dim=1)
            src_corr_feats = F.normalize(src_corr_feats, p=2, dim=1)
            cross_affinity = (torch.relu(ref_corr_feats @ ref_corr_feats.T)
                              * torch.relu(src_corr_feats @ src_corr_feats.T))
            affinity = affinity * (1.0 - self.feature_weight + self.feature_weight * cross_affinity)
            diag_prior = torch.relu((ref_corr_feats * src_corr_feats).sum(dim=1))
        else:
            diag_prior = None
        diag_idx = torch.arange(num_corrs, device=device)
        affinity[diag_idx, diag_idx] = diag_prior if diag_prior is not None else 0.0

        # Leading eigenvector via power iteration (Perron vector of a non-negative matrix).
        spectral_score = torch.ones(num_corrs, device=device)
        for _ in range(self.num_iterations):
            spectral_score = affinity @ spectral_score
            spectral_score = spectral_score / (spectral_score.abs().max() + 1e-12)
        spectral_score = (spectral_score / (spectral_score.max() + 1e-12)).clamp(min=0.0)

        # SC2: second-order aggregation over reciprocal kNN of both sides (vectorised).
        common_mask = self._knn_mask(ref_dist, self.sc2_topk) & self._knn_mask(src_dist, self.sc2_topk)
        neighbor_counts = common_mask.sum(dim=1)
        sc2_score = (common_mask * spectral_score.unsqueeze(0)).sum(dim=1) / neighbor_counts.clamp(min=1)
        isolated = neighbor_counts == 0
        sc2_score[isolated] = spectral_score[isolated]  # isolated correspondences keep their own score.
        sc2_score = sc2_score / (sc2_score.max() + 1e-12)

        # Joint strict selection.
        final_score = spectral_score * sc2_score
        mask = final_score > self.threshold
        if mask.sum() < self.min_correspondences:
            _, topk_idx = corr_scores.topk(k=min(self.topk_fallback, num_corrs))
            mask = torch.zeros(num_corrs, dtype=torch.bool, device=device)
            mask[topk_idx] = True

        joint_scores = corr_scores * final_score
        return (ref_corr_points[mask], src_corr_points[mask], joint_scores[mask], torch.where(mask)[0])


class OneToManyRegistration(nn.Module):
    def __init__(self, acceptance_radius, correspondence_threshold=3,
                 correspondence_limit=None, num_refinement_steps=5):
        super(OneToManyRegistration, self).__init__()
        self.acceptance_radius = acceptance_radius
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps

    @torch.no_grad()
    def forward(self, ref_corr_points, src_corr_points, corr_scores):
        if ref_corr_points.shape[0] < self.correspondence_threshold:
            estimated_transform = torch.eye(4, device=ref_corr_points.device)
            return ref_corr_points, src_corr_points, corr_scores, estimated_transform

        estimated_transform = weighted_procrustes(
            src_corr_points.unsqueeze(0),
            ref_corr_points.unsqueeze(0),
            weights=corr_scores.unsqueeze(0),
            return_transform=True,
        )
        estimated_transform = estimated_transform.squeeze(0)

        for _ in range(self.num_refinement_steps):
            src_corr_points_t = torch.matmul(
                src_corr_points - estimated_transform[:3, 3].unsqueeze(0),
                estimated_transform[:3, :3].T,
            ) + estimated_transform[:3, 3].unsqueeze(0)

            dist = torch.linalg.norm(ref_corr_points - src_corr_points_t, dim=1)
            inlier_masks = torch.lt(dist, self.acceptance_radius)

            if inlier_masks.sum() < self.correspondence_threshold:
                break

            estimated_transform = weighted_procrustes(
                src_corr_points[inlier_masks].unsqueeze(0),
                ref_corr_points[inlier_masks].unsqueeze(0),
                weights=corr_scores[inlier_masks].unsqueeze(0),
                return_transform=True,
            )
            estimated_transform = estimated_transform.squeeze(0)

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform


class MultiHypothesisRegistration(nn.Module):
    """Multi-hypothesis weighted-SVD registration with chamfer/semantic selection (DCR-Former innovation 3).

    Generates m pose hypotheses: one full-set weighted SVD (conservative baseline) plus
    m-1 weighted random subsets with per-hypothesis sampling ratios, each refined by
    inlier iterations on the full correspondence set (reusing OneToManyRegistration).
    Hypotheses are scored by correspondence-level inlier rate and coarse-level chamfer
    coverage (tau0), optionally fused with the semantic consistency of the chamfer-matched
    superpoints, and the best-scoring hypothesis is returned.
    """

    def __init__(self, acceptance_radius, num_hypotheses=8, subset_ratio=0.7,
                 chamfer_radius=0.1, correspondence_threshold=3,
                 num_refinement_steps=5, semantic_weight=0.3, spectral_sigma=0.1):
        super(MultiHypothesisRegistration, self).__init__()
        self.num_hypotheses = num_hypotheses
        self.subset_ratio = subset_ratio
        self.chamfer_radius = chamfer_radius
        self.semantic_weight = semantic_weight
        self.spectral_sigma = spectral_sigma
        self.base_registration = OneToManyRegistration(
            acceptance_radius, correspondence_threshold, None, num_refinement_steps)

    @staticmethod
    def _apply_transform_batch(points, rotations, translations):
        """Apply m rigid transforms to points [N,3] -> [m,N,3] (otm_registration convention)."""
        return ((points.unsqueeze(0) - translations.unsqueeze(1)) @ rotations.transpose(1, 2)
                + translations.unsqueeze(1))

    @torch.no_grad()
    def forward(self, ref_corr_points, src_corr_points, corr_scores,
                ref_points_c=None, src_points_c=None,
                ref_sem_feats_c=None, src_sem_feats_c=None):
        num_corrs = ref_corr_points.shape[0]
        if num_corrs < self.base_registration.correspondence_threshold:
            estimated_transform = torch.eye(4, device=ref_corr_points.device)
            return ref_corr_points, src_corr_points, corr_scores, estimated_transform

        # Hypothesis 0: full-set weighted SVD with refinement.
        transforms = [self.base_registration(ref_corr_points, src_corr_points, corr_scores)[3]]

        # Spectral cluster hypotheses: eigenvectors of the geometric affinity matrix
        # separate rigidly-consistent groups (repeated structures / multi-modal matches).
        cluster_masks = []
        ref_dist = torch.cdist(ref_corr_points, ref_corr_points)
        src_dist = torch.cdist(src_corr_points, src_corr_points)
        affinity = torch.exp(-(ref_dist - src_dist).abs() / self.spectral_sigma)
        affinity.fill_diagonal_(0.0)
        try:
            eigvals, eigvecs = torch.linalg.eigh(affinity.double())
            lam1 = eigvals[-1].clamp(min=1e-9)
            v1 = eigvecs[:, -1].float()
            cluster_masks.append(v1 > v1.median())  # strong members of the dominant cluster
            for k in range(2, min(4, num_corrs) + 1):
                if eigvals[-k] / lam1 > 0.15:
                    vk = eigvecs[:, -k].float()
                    cluster_masks.append(vk > 0)
                    cluster_masks.append(vk < 0)
        except Exception:
            pass  # degenerate matrices fall back to full-set + subsets only
        for cluster in cluster_masks:
            if cluster.sum() >= self.base_registration.correspondence_threshold:
                _, _, _, transform_k = self.base_registration(
                    ref_corr_points[cluster], src_corr_points[cluster], corr_scores[cluster])
                transforms.append(transform_k)

        # Remaining slots: weighted random subsets for extra diversity.
        sample_weights = corr_scores.clamp(min=1e-6)
        while len(transforms) < self.num_hypotheses:
            ratio_k = self.subset_ratio * (0.75 + 0.5 * torch.rand(1).item())
            num_sub = max(self.base_registration.correspondence_threshold,
                         int(num_corrs * ratio_k))
            sub_idx = torch.multinomial(sample_weights, num_sub, replacement=True)
            _, _, _, transform_k = self.base_registration(
                ref_corr_points[sub_idx], src_corr_points[sub_idx], corr_scores[sub_idx])
            transforms.append(transform_k)

        rotations = torch.stack([T[:3, :3] for T in transforms])    # [m,3,3]
        translations = torch.stack([T[:3, 3] for T in transforms])   # [m,3]

        # Score 1: correspondence-level inlier rate.
        src_t = self._apply_transform_batch(src_corr_points, rotations, translations)  # [m,N,3]
        corr_dists = torch.linalg.norm(src_t - ref_corr_points.unsqueeze(0), dim=2)    # [m,N]
        inlier_mask = torch.lt(corr_dists, self.base_registration.acceptance_radius)   # [m,N]
        corr_rate = inlier_mask.float().mean(dim=1)

        # Score 1b: spectral support of the inliers (membership in the dominant
        # geometrically-consistent cluster of the hypothesis-generation affinity).
        # Selected by E12 offline ablation (+0.6pp over baseline on 3DCSR): hypothesis
        # scoring is NOT the dominant bottleneck, but this weighting is the best found.
        spectral_score = torch.ones(num_corrs, device=ref_corr_points.device)
        for _ in range(30):
            spectral_score = affinity @ spectral_score
            spectral_score = spectral_score / (spectral_score.abs().max() + 1e-12)
        spectral_score = (spectral_score / (spectral_score.max() + 1e-12)).clamp(min=0.0)
        spec_support = (inlier_mask.float() * spectral_score.unsqueeze(0)).sum(dim=1) \
            / (inlier_mask.float().sum(dim=1) + 1e-6)

        # Score 2 + 3: coarse-level chamfer coverage and (optional) semantic consistency.
        sem_consistency = None
        chamfer_rate = torch.zeros_like(corr_rate)
        if ref_points_c is not None and src_points_c is not None and ref_points_c.shape[0] > 0:
            src_c_t = self._apply_transform_batch(src_points_c, rotations, translations)  # [m,Ns,3]
            nn_dists, nn_idx = torch.cdist(src_c_t, ref_points_c).min(dim=2)              # [m,Ns]
            chamfer_rate = torch.lt(nn_dists, self.chamfer_radius).float().mean(dim=1)
            if (ref_sem_feats_c is not None and src_sem_feats_c is not None
                    and self.semantic_weight > 0 and src_sem_feats_c.shape[0] == src_points_c.shape[0]):
                ref_sem = F.normalize(ref_sem_feats_c, p=2, dim=1)
                src_sem = F.normalize(src_sem_feats_c, p=2, dim=1)
                sem_sim = (src_sem.unsqueeze(0) * ref_sem[nn_idx]).sum(dim=2)             # [m,Ns]
                sem_consistency = sem_sim.mean(dim=1)

        scores = 0.5 * corr_rate + 0.3 * chamfer_rate + 0.2 * spec_support
        if sem_consistency is not None:
            scores = (1.0 - self.semantic_weight) * scores \
                     + self.semantic_weight * sem_consistency.clamp(min=0.0)

        best = int(scores.argmax().item())
        return ref_corr_points, src_corr_points, corr_scores, transforms[best]
