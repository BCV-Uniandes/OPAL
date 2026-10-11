import numpy as np


def density(s_gt, s_pred):
    return float(np.linalg.norm(s_gt.density - s_pred.density))
