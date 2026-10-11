import torch
import torch.nn.functional as F
from tqdm import tqdm


class CSPFlow():
    def __init__(self, device, num_atoms, generator, lattice_polar_sigma = 0.1): # lattice_polar_sigma default from CrystalFlow
        self.generator = generator
        self.num_atoms = num_atoms
        self.lattice_polar_sigma = lattice_polar_sigma
        # Defaults overwritten by estimate_flow_statistics() before training.
        self.lattice_source_shift = 1.0
        self.coords_loss_0 = 1.0
        self.l_loss_0 = 1.0

    def estimate_flow_statistics(self, dataloader):
        """Estimates lattice_source_shift, coords_loss_0, and l_loss_0 in a
        single pass over the training set.

        lattice_source_shift: mean of k5 over the training set. Shifts the
            source distribution so that E[field_l[:, -1]] ~ 0, eliminating the
            20x MSE imbalance in the last polar component.
        coords_loss_0: expected MSE of a zero-predictor for coordinates
            (empirical mean of |field_coords|^2 over valid atoms).
        l_loss_0: expected MSE of a zero-predictor for the lattice
            (Var(lattice) + sigma^2, averaged over the 6 polar components).
        """
        all_k5 = []
        all_lattice = []
        coords_mse_list = []
        for batch in tqdm(dataloader):
            coords, f_atoms, mask, lengths, angles, num_atoms, lattice_polar, mp_id, spacegroup = batch
            all_k5.append(lattice_polar[:, -1])
            all_lattice.append(lattice_polar)
            mask_sq = mask.squeeze()
            coords_0 = torch.rand_like(coords)
            field_coords = (coords - coords_0 - 0.5) % 1 - 0.5
            valid_field = field_coords[mask_sq]
            coords_mse_list.append(valid_field.pow(2).mean())
        self.lattice_source_shift = torch.cat(all_k5).mean().item()
        all_lat = torch.cat(all_lattice, dim=0)
        self.l_loss_0 = (all_lat.var(dim=0).mean() + self.lattice_polar_sigma ** 2).item()
        self.coords_loss_0 = torch.stack(coords_mse_list).mean().item()
        print(f'Lattice source shift  (mean k5):  {self.lattice_source_shift:.6f}')
        print(f'Loss scale — coords_0: {self.coords_loss_0:.6f},  lattice_0: {self.l_loss_0:.6f}')

    def get_noised_coords(self, coords, t):
        with torch.no_grad():
            coords_0 = torch.rand(coords.shape, generator=self.generator, device=coords.device, dtype=coords.dtype)
            field_coords = (coords - coords_0 - 0.5) % 1 - 0.5
            t = t.unsqueeze(-1).unsqueeze(-1)
            coords_t = coords_0 + t * field_coords
            coords_t = coords_t - torch.floor(coords_t)
        return coords_t, field_coords

    def get_noised_lattice(self, lattice, t, polar = True):
        # Lattice must be in polar representation
        with torch.no_grad():
            t = t.unsqueeze(-1)
            l_0 = torch.randn(lattice.shape, generator=self.generator, device=lattice.device, dtype=lattice.dtype) * self.lattice_polar_sigma
            if polar:
                # Shift the source distribution for k5 to the dataset mean instead of
                # the hardcoded 1.0, so that E[field_l[:, -1]] ~ 0 and all six polar
                # components contribute equally to the lattice loss gradient.
                l_0[:, -1] = l_0[:, -1] + self.lattice_source_shift
            field_l = lattice - l_0
            l_t = l_0 + t * field_l
        return l_t, field_l

    
    def get_anneal_factor(self, t, slope: float = 5.0, offset: float = 0.0):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return 1 + slope * F.relu(t - offset)


    def step(self, step_size, l_t , x_t, l_pred, x_pred, t):
        with torch.no_grad():
            x_pred *= self.get_anneal_factor(t)
            l_t_plus_1 = l_t + step_size* l_pred
            x_t_plus_1 = x_t + step_size* x_pred
        return l_t_plus_1, x_t_plus_1
    