import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn as nn
from tqdm import tqdm


# Adapted from PyTorch3D (pytorch3d/transforms/rotation_conversions.py).
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# BSD-3-Clause license, see licenses/LICENSE-PyTorch3D.
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def masked_linear_sum_assignment(matrixes, lens):
    R = []
    C = []
    for i in range(len(matrixes)):
        matrix = matrixes[i][:lens[i], :lens[i]]
        row, column = linear_sum_assignment(matrix)
        final_row = np.zeros([len(row), 2]) + i
        final_row[:, 1] = row
        final_column = np.zeros([len(column), 2]) + i
        final_column[:, 1] = column
        if len(R) == 0:
            R = final_row
            C = final_column
        else:
            R = np.concatenate([R, final_row], axis=0)
            C = np.concatenate([C, final_column], axis=0)
    return R, C


def Hungarian_Alignment(X, Y, Xtypes, Ytypes, pred_lens, it=4, steps=400):
    device = 'cuda'
    X = X.to(device)
    Y = Y.to(device)
    Xtypes = Xtypes.to(device)
    Ytypes = Ytypes.to(device)
    pred_lens = torch.tensor(pred_lens).to(device)

    final_qs = torch.zeros(len(Xtypes), 4).to(device)
    final_ts = torch.zeros(len(Xtypes), 3).to(device)
    best_losses = torch.zeros(len(Xtypes)).to(device) - 1

    for j in range(it):
        q = torch.randn((len(Xtypes), 4), device=device, requires_grad=True)
        t = torch.zeros((len(Xtypes), 3), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([q, t], lr=0.1)
        types_mask = Xtypes.unsqueeze(-1) == Ytypes.unsqueeze(1).to(device)
        mse = nn.MSELoss(reduction='none')
        for i in tqdm(range(steps)):
            q_norm = F.normalize(q, dim=1)
            R_est = quaternion_to_matrix(q_norm)
            X_rot = torch.bmm(X, R_est.transpose(1, 2)) + t.unsqueeze(1)
            cost = torch.cdist(X_rot, Y, p=2)
            cost[types_mask == False] = 10**9
            cost = cost.detach().cpu().numpy()
            x_indexes, y_indexes = masked_linear_sum_assignment(cost, pred_lens)
            x_indexes = torch.tensor(x_indexes.tolist()).to(torch.int32).to(device)
            y_indexes = torch.tensor(y_indexes.tolist()).to(torch.int32).to(device)
            gen_indexes = x_indexes[:, 0].to(device)
            ind_loss = mse(X_rot[x_indexes[:, 0], x_indexes[:, 1], :], Y[y_indexes[:, 0], y_indexes[:, 1], :])
            ind_loss = torch.sum(ind_loss, dim=1)
            losses = torch.zeros((X.shape[0])).to(device)
            losses.index_add_(0, gen_indexes, ind_loss)
            losses = losses / (pred_lens * 3)
            aux_mask_1 = best_losses < 0
            final_qs[(aux_mask_1).unsqueeze(-1).repeat(1, 4)] = q_norm.detach()[(aux_mask_1).unsqueeze(-1).repeat(1, 4)]
            final_ts[(aux_mask_1).unsqueeze(-1).repeat(1, 3)] = t.detach()[(aux_mask_1).unsqueeze(-1).repeat(1, 3)]
            best_losses[aux_mask_1] = losses[aux_mask_1]
            aux_mask_2 = losses < best_losses
            final_qs[(aux_mask_2).unsqueeze(-1).repeat(1, 4)] = q_norm.detach()[(aux_mask_2).unsqueeze(-1).repeat(1, 4)]
            final_ts[(aux_mask_2).unsqueeze(-1).repeat(1, 3)] = t.detach()[(aux_mask_2).unsqueeze(-1).repeat(1, 3)]
            best_losses[aux_mask_2] = losses[aux_mask_2]
            final_loss = losses.sum()
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
    final_matrixes = quaternion_to_matrix(F.normalize(final_qs, dim=1))
    final_matrixes = final_matrixes.cpu().numpy().tolist()
    final_ts = final_ts.cpu().numpy().tolist()
    return final_matrixes, final_ts


def Hungarian_Matcher(X, Y, Xtypes, Ytypes, rotation, translation):
    Xfinal = X @ rotation.T + translation
    Xmatched, Ymatched = [], []
    Xtypes_matched = []
    for typ in torch.unique(Xtypes):
        x_group = Xfinal[Xtypes == typ]
        y_group = Y[Ytypes == typ]
        cost = torch.cdist(x_group, y_group, p=2).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        Xmatched.append(x_group[row])
        Ymatched.append(y_group[col])
        Xtypes_matched.append(Xtypes[Xtypes == typ][row])
    Xmatched = torch.cat(Xmatched, dim=0)
    Ymatched = torch.cat(Ymatched, dim=0)
    Xtypes_matched = torch.cat(Xtypes_matched, dim=0)
    return Xmatched, Ymatched, Xtypes_matched
