import numpy as np
import warnings
from numpy.typing import NDArray
from common.utils import awa_metric
warnings.filterwarnings("ignore")


def superpose_rmsd_metric(a: NDArray[np.float64], b: NDArray[np.float64]):
    squared_distances = np.sum(np.square(np.asarray(a) - np.asarray(b)), axis=1)
    return np.sqrt(np.mean(squared_distances))


def superpose_rmsd_awa(s_gt, s_pred, rotation, translation):
    return awa_metric(s_gt, s_pred, rotation, translation, superpose_rmsd_metric)
