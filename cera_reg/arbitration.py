"""Adopted CERA-Reg implementation; source mapping is in provenance/."""
import numpy as np


def choose(utility,risk,ids,baseline,anchor):
    # Subtract the reference utility when anchoring; retain the reference if no candidate passes.
    u=utility.copy()
    if anchor:u-=u[np.flatnonzero(ids==baseline)[0]]
    keep=(u>.02)&(risk<.1)&(ids!=baseline)
    return int(ids[np.argmax(np.where(keep,u,-np.inf))]) if keep.any() else baseline

