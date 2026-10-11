import numpy as np
import warnings
from numpy.typing import NDArray
from common.utils import awa_metric
warnings.filterwarnings("ignore")


# pairwise_distances and sinkhorn are adapted from point-cloud-utils (point_cloud_utils/_sinkhorn.py).
# Copyright 2023 Francis Williams. MIT license, see licenses/LICENSE-point-cloud-utils.
def pairwise_distances(a: NDArray[np.float32], b: NDArray[np.float32], p=None):
    squeezed = False
    if len(a.shape) == 2 and len(b.shape) == 2:
        a = a[np.newaxis, :, :]
        b = b[np.newaxis, :, :]
        squeezed = True
    ret = np.linalg.norm(a[:, :, np.newaxis, :] - b[:, np.newaxis, :, :], axis=-1, ord=p)
    if squeezed:
        ret = np.squeeze(ret)
    return ret.astype(np.float32)


def sinkhorn(a: NDArray[np.float32], b: NDArray[np.float32], M: NDArray[np.float32], eps, max_iters=100, stop_thresh=1e-3):
    M = M[np.newaxis, np.newaxis] if M.shape == () else np.squeeze(M)
    a = a if a.shape == (1,) else np.squeeze(a)
    b = b if b.shape == (1,) else np.squeeze(b)
    squeezed = False

    if len(M.shape) == 2 and len(a.shape) == 1 and len(b.shape) == 1:
        M = M[np.newaxis, :, :]
        a = a[np.newaxis, :]
        b = b[np.newaxis, :]
        squeezed = True

    u = np.zeros_like(a)
    v = np.zeros_like(b)
    M_t = np.transpose(M, axes=(0, 2, 1))

    def stabilized_log_sum_exp(x):
        max_x = x.max(2)
        x = x - max_x[:, :, np.newaxis]
        return np.log(np.sum(np.exp(x), axis=2)) + max_x

    for _ in range(max_iters):
        u_prev = u
        v_prev = v
        summand_u = (-M + np.expand_dims(v, 1)) / eps
        u = eps * (np.log(a) - stabilized_log_sum_exp(summand_u))
        summand_v = (-M_t + np.expand_dims(u, 1)) / eps
        v = eps * (np.log(b) - stabilized_log_sum_exp(summand_v))
        if np.sum(np.abs(u_prev - u), axis=1).max() < stop_thresh and \
           np.sum(np.abs(v_prev - v), axis=1).max() < stop_thresh:
            break

    log_P = (-M + np.expand_dims(u, 2) + np.expand_dims(v, 1)) / eps
    P = np.exp(log_P)
    if squeezed:
        P = np.squeeze(P)
    return P


def sinkhorn_metric(a: NDArray[np.float32], b: NDArray[np.float32]):
    M = pairwise_distances(a, b)
    w_a = np.ones(a.shape[0], dtype=np.float32)
    w_b = np.ones(b.shape[0], dtype=np.float32)
    P = sinkhorn(w_a, w_b, M, eps=1e-3)
    return (M * P).sum()


def sinkhorn_awa(s_gt, s_pred, rotation, translation):
    return awa_metric(s_gt, s_pred, rotation, translation, sinkhorn_metric)
