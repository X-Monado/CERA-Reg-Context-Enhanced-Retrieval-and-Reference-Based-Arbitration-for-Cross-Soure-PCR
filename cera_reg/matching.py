"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
from .retrieval import EarlySpatialMatching, anchor_indices
from .sparse_support import refined_bank


def matcher_banks(output,rm,sm):
    m=EarlySpatialMatching().to(output['ref_points_f'].device)
    m.ref_points=output['ref_points_c'];m.src_points=output['src_points_c']
    m(output['ref_feats_c'],output['src_feats_c'],rm,sm)
    return m, [anchor_indices(m),refined_bank(m)[0]]

