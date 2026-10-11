import numpy as np
import warnings
import point_cloud_utils as pcu
from numpy.typing import NDArray
from common.utils import awa_metric
warnings.filterwarnings("ignore")


def chamfer_metric(a: NDArray[np.float64], b: NDArray[np.float64]):
    return pcu.chamfer_distance(a, b)


def chamfer_awa(s_gt, s_pred, rotation, translation):
    return awa_metric(s_gt, s_pred, rotation, translation, chamfer_metric)
