import numpy as np
import warnings
import point_cloud_utils as pcu
from numpy.typing import NDArray
from common.utils import awa_metric
warnings.filterwarnings("ignore")


def hausdorff_metric(a: NDArray[np.float64], b: NDArray[np.float64]):
    hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(b, a, return_index=True)
    assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_dist**2) < 1e-5
    return hausdorff_dist


def hausdorff_awa(s_gt, s_pred, rotation, translation):
    return awa_metric(s_gt, s_pred, rotation, translation, hausdorff_metric)
