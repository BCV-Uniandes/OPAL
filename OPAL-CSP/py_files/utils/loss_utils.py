import math
import torch
import torch.nn as nn


class UncertaintyWeightingLoss(nn.Module):
    '''
    Uncertainty Weigthing Loss from: https://arxiv.org/abs/1705.07115

    Original Proposed Loss:
    L = 1/(2 * sigma_coords^2) * L_coords + log(sigma_coords) + 1/(2 * sigma_cell^2) * L_cell + log(sigma_cell)
    
    Numerically stable version:
    s_i = log(sigma_i^2)
    L = 1/2 * exp(-s_coords) * L_coords + 1/2 * s_coords + 1/2 * exp(-s_cell) * L_cell + 1/2 * s_cell
    
    Loss Weights Initialization (alpha, beta):
    alpha = 1/2 * exp(-s_coords)
    beta = 1/2 * exp(-s_cell)

    s_coords = -log(2*alpha)
    s_cell = -log(2*beta)
    '''

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        s_coords = -math.log(2.0 * alpha)
        s_cell = -math.log(2.0 * beta)
        self.log_vars = nn.Parameter(torch.tensor([s_coords, s_cell], dtype = torch.float32))

    def forward(self, l_coords, l_cell):
        s_coords, s_cell = self.log_vars[0], self.log_vars[1]
        w_coords, w_cell = torch.exp(-s_coords), torch.exp(-s_cell)
        total_loss = 0.5 * (w_coords * l_coords + s_coords) + \
            0.5 * (w_cell * l_cell + s_cell)
        info = {
            "alpha": 0.5 * w_coords.detach(),
            "beta": 0.5 * w_cell.detach(),
            "sigma_coord": torch.exp(0.5 * s_coords).detach(),
            "sigma_cell": torch.exp(0.5 * s_cell).detach(),
        }
        return total_loss, info
    

class GradNorm(nn.Module):
    def __init__(self, num_tasks, alpha=1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.register_buffer('initial_losses', torch.zeros(num_tasks))

    def compute_loss(self, task_losses, shared_params):
        # Save initial losses (only once; persisted in state_dict via buffer).
        if self.initial_losses.sum() == 0:
            self.initial_losses.copy_(torch.stack([l.detach() for l in task_losses]))

        # Weighted total loss
        weighted_losses = [self.weights[i] * task_losses[i] for i in range(self.num_tasks)]
        total_loss = sum(weighted_losses)

        # Grad norms by task.
        grad_norms = []
        for wl in weighted_losses:
            grads = torch.autograd.grad(wl, shared_params, retain_graph=True, create_graph=True)
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # Loss ratios.
        loss_ratios = torch.stack([task_losses[i] / self.initial_losses[i] for i in range(self.num_tasks)])

        # Training rates.
        mean_loss_ratio = loss_ratios.mean().detach()
        training_rates = loss_ratios / mean_loss_ratio #r_i

        # Target grad norms.
        mean_grad_norm = grad_norms.mean().detach()
        target_grad_norms = (mean_grad_norm * training_rates ** self.alpha).detach()

        # Gradnorm Loss.
        gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()

        return total_loss, gradnorm_loss
