"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np
import torch
import torch.nn.functional as F
import time
import random
from geotransformer.utils.data import registration_collate_fn_stack_mode


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def collate(item, cfg):
    return registration_collate_fn_stack_mode([item], cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size, cfg.backbone.init_radius, [48]*4)

