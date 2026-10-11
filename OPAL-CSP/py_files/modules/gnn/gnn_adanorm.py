import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch_scatter import scatter
from torch.nn import functional as F
from torch_geometric.utils import dense_to_sparse
import pytorch_lightning as pl
from py_files.utils.data_utils import lattice_polar_build_torch, lattice_matrix_to_params
from tqdm import tqdm
from PyAstronomy.pyasl import AtomicNo
import Dans_Diffraction as dif


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def lattice_polar_build_torch(k):
    assert k.dim() == 2, "input must be batched k of shape (B,6)"
    S0 = torch.stack([k[:, 3] + k[:, 4] + k[:, 5], k[:, 0], k[:, 1]], dim=1)  # (B, 3)
    S1 = torch.stack([k[:, 0], -k[:, 3] + k[:, 4] + k[:, 5], k[:, 2]], dim=1)  # (B, 3)
    S2 = torch.stack([k[:, 1], k[:, 2], -2 * k[:, 4] + k[:, 5]], dim=1)  # (B, 3)
    S = torch.stack([S0, S1, S2], dim=1)  # (B, 3, 3)
    expS = torch.matrix_exp(S)  # (B, 3, 3)
    return expS


# Reciprocal lattice vectors b_i = 2*pi * (a_j x a_k) / V for each lattice in the batch.
def get_reciprocal_lattice_torch(L):
    V = torch.det(L)[:, None] + 1e-9  # (B, 1)
    a0 = L[:, 0, :]  # (B, 3)
    a1 = L[:, 1, :]
    a2 = L[:, 2, :]
    cross_a12 = torch.cross(a1, a2, dim=1)  # (B, 3)
    cross_a20 = torch.cross(a2, a0, dim=1)
    cross_a01 = torch.cross(a0, a1, dim=1)
    b0 = (2 * math.pi * cross_a12 / V)[:, None, :]  # (B, 1, 3)
    b1 = (2 * math.pi * cross_a20 / V)[:, None, :]  # (B, 1, 3)
    b2 = (2 * math.pi * cross_a01 / V)[:, None, :]  # (B, 1, 3)
    b = torch.cat([b0, b1, b2], dim=1)  # (B, 3, 3)
    return b



class RecSinusoidsEmbedding(nn.Module):
    def __init__(self, n_millers=10):
        super().__init__()
        self.n_millers = n_millers
        self.millerindices = torch.cartesian_prod(
            torch.arange(self.n_millers),
            torch.arange(self.n_millers),
            torch.arange(self.n_millers),
        ).to(torch.get_default_dtype())
        self.dim = 2 * self.millerindices.shape[0]

    def forward(self, frac_diff, lattice):
        cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattice)  # (E,3)
        R = get_reciprocal_lattice_torch(lattice)
        hb = torch.einsum('mi,eij->emj', self.millerindices.to(R.device), R)  # (E, M, 3)
        hbX = torch.einsum("emj,ej->em", hb, cart_diff)  # (E, M)
        emb = torch.cat([hbX.cos(), hbX.sin()], dim=-1)  # (E, 2M)
        return emb


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class CSPLayer(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        latent_dim = 64,
        lattice_dim=9,
        act_fn=nn.SiLU(),
        dis_emb=None,
        rec_emb=None,
    ):
        super().__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.rec_emb = rec_emb

        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if rec_emb is not None:
            self.dis_dim += rec_emb.dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + lattice_dim + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), act_fn, nn.Linear(hidden_dim, hidden_dim), act_fn
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 3 * hidden_dim, bias=True)
        )



    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices_rep,
        edge_index,
        edge2graph,
        frac_diff,
        lattices_mat,
    ):
        
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        inputs = [hi, hj]


        xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
        frac_diff = (xj - xi) % 1.0


        rec_diff = self.rec_emb(frac_diff, lattices_mat[edge2graph])
        inputs.append(rec_diff)


        frac_diff_emb = self.dis_emb(frac_diff)
        inputs.append(frac_diff_emb)


        lattice_ips = lattices_rep
        lattice_ips_flatten = torch.flatten(lattice_ips, start_dim=1)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        inputs.append(lattice_ips_flatten_edges)

        edges_input = torch.cat(inputs, dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features


    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim=0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out


    def forward(self, node_features, frac_coords, lattices_rep, edge_index, edge2graph, frac_diff, lattices_mat, c):
        node_input = node_features
        shift_gnn, scale_gnn, gate_gnn = self.adaLN_modulation(c).chunk(3, dim=1)
        node_features = modulate(self.norm1(node_input), shift_gnn, scale_gnn)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices_rep, edge_index, edge2graph, frac_diff, lattices_mat
        )
        node_output = node_input + gate_gnn * self.node_model(node_features, edge_features, edge_index)
        return node_output



class GNN(pl.LightningModule):
    def __init__(
        self, method,
        generator,
        hidden_dim=512,
        latent_dim=256,
        lattice_dim=6,
        num_layers=6,
        max_atoms=100,
        num_freqs=256,
        num_millers=8,
        alpha = 10, 
        beta = 1,
        lr = 1e-03,
        scheduler_factor = 0.6,
        scheduler_patience = 30,
    ):
        super().__init__()
        self.lr = lr
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.method = method
        self.alpha, self.beta = alpha, beta
        self.generator = generator
        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim, hidden_dim)

        self.act_fn = nn.SiLU()
        self.time_emb = SinusoidalTimeEmbeddings(latent_dim)
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)  # no trainable params
        self.rec_emb = RecSinusoidsEmbedding(n_millers=num_millers)  # no trainable params

        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i,
                CSPLayer(
                    hidden_dim=hidden_dim,
                    latent_dim = latent_dim,
                    lattice_dim=lattice_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    rec_emb=self.rec_emb,
                ),
            )

        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, lattice_dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)


    def gen_edges(self, num_atoms, frac_coords):
        lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0 # wrap fractional differences into [0, 1) (periodic boundary conditions)


    @torch.no_grad()
    def _process_batch(self, batch, t):
        coords_t, f_atoms, mask, lenghts, angles, num_atoms, l_t, mp_id, spacegroup = batch
        device = self.device
        coords_t, f_atoms, mask, num_atoms, l_t = coords_t.to(device), f_atoms.to(device), mask.to(device), num_atoms.to(device), l_t.to(device)
        t = t * torch.ones(coords_t.shape[0], device=self.device)
        mask = mask.squeeze()
        atoms = f_atoms[mask]
        num_atoms = mask.sum(dim = -1)
        ar = torch.arange(0, mask.shape[0]).to(mask.device)
        ar = ar.unsqueeze(-1)
        node2graph = mask + ar
        node2graph = node2graph[mask] - 1
        mask = mask.unsqueeze(-1).repeat(1, 1, 3)
        coords_out_f = torch.zeros(coords_t.shape, device=coords_t.device)
        coords_t = coords_t[mask].view(-1, 3)
        l_t = l_t.to(torch.float32)
        num_atoms = num_atoms.to(torch.int32)
        lattices_mat = lattice_polar_build_torch(l_t)
        l_out, coords_out = self.forward(t, atoms, coords_t, l_t, num_atoms, node2graph, lattices_mat)
        coords_out_f[mask] = coords_out.flatten()
        return l_out, coords_out_f


    def forward(self, t, atom_types, frac_coords, lattices_rep, num_atoms, node2graph, lattices_mat):
        t = self.time_emb(t)

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        c = t_per_atom
        node_features = self.atom_latent_emb(node_features)


        for i in range(0, self.num_layers):

            csp_layer = self._modules["csp_layer_%d" % i]
            node_features = csp_layer(
                node_features,
                frac_coords,
                lattices_rep,
                edges,
                edge2graph,
                frac_diff,
                lattices_mat,
                c
            )

        node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(lattices_rep.shape)
        return lattice_out, coord_out
    

    def _common_step(self, batch, batch_idx):
        coords, f_atoms, mask, lenghts, angles, num_atoms, lattice_polar, mp_id, spacegroup = batch
        t = torch.rand(coords.shape[0], device=self.device, generator=self.generator)
        coords_t, coords_tar = self.method.get_noised_coords(coords, t)
        l_t, l_tar =  self.method.get_noised_lattice(lattice_polar, t)
        mask = mask.squeeze()
        atoms = f_atoms[mask]
        num_atoms = mask.sum(dim = -1)
        ar = torch.arange(0, mask.shape[0]).to(mask.device)
        ar = ar.unsqueeze(-1)
        node2graph = mask + ar
        node2graph = node2graph[mask] - 1
        mask = mask.unsqueeze(-1).repeat(1, 1, 3)
        coords_t = coords_t[mask].view(-1, 3)
        coords_tar = coords_tar[mask].view(-1,3)
        lattices_mat = lattice_polar_build_torch(l_t)
        l_out, coords_out = self.forward(t, atoms, coords_t, l_t, num_atoms, node2graph, lattices_mat)
        l_loss = F.mse_loss(l_out, l_tar)
        coords_loss = F.mse_loss(coords_out, coords_tar)
        loss = self.alpha * coords_loss + self.beta * l_loss
        return loss, coords_loss, l_loss


    def training_step(self, batch, batch_idx):
        loss, coords_loss, l_loss = self._common_step(batch, batch_idx)
        self.log_dict(
            {'train_joint_loss': loss.detach(),
             'train_coords_loss': coords_loss.detach(),
             'train_lattice_loss': l_loss.detach(),
             'train_unweighted_loss': coords_loss.detach() + l_loss.detach()
            }
                       , batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        loss, coords_loss, l_loss = self._common_step(batch, batch_idx)
        self.log_dict(
            {'val_joint_loss': loss.detach(),
             'val_coords_loss': coords_loss.detach(),
             'val_lattice_loss': l_loss.detach(),
             'val_unweighted_loss': coords_loss.detach() + l_loss.detach()
            }
                       , batch_size=len(batch), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.flow_sampler(batch, steps=1000)

 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', self.scheduler_factor, self.scheduler_patience)
        return  {"optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_joint_loss"}}


    def flow_sampler(self, batch, steps=1000):
        generator = self.generator
        an = AtomicNo()
        coords, f_atoms, mask, lenghts, angles, num_atoms, lattice_polar, mp_id, spacegroup = batch
        l_t, x_t = torch.randn(lattice_polar.shape, generator=generator, device=self.device) *self.method.lattice_polar_sigma, torch.rand(coords.shape, generator=generator, device=self.device) 
        l_t = l_t
        ss = 1/steps
        for t in tqdm(range(0, steps)):
            batch = (x_t, f_atoms, mask, None, None, num_atoms, l_t, mp_id, spacegroup)
            l_pred, x_pred = self._process_batch(batch, t*ss)
            l_t, x_t = self.method.step(ss, l_t , x_t, l_pred, x_pred, t*ss)
            x_t = x_t - torch.floor(x_t)
        mask = mask.squeeze().unsqueeze(-1)
        l_t = lattice_polar_build_torch(l_t)
        for j in range(x_t.shape[0]):
            final_mask = mask[j]
            final_atoms = f_atoms[j][final_mask.squeeze()]
            final_atoms = [an.getElSymbol(k) for k in final_atoms]
            final_mask = final_mask.repeat(1, 3)
            final_coords = x_t[j][final_mask].view(-1, 3).cpu().numpy()
            final_lattice = lattice_matrix_to_params(l_t[j].cpu().numpy())
            la, lb, lc, alpha, beta, gamma = final_lattice
            u = final_coords[:, 0]
            v = final_coords[:, 1]
            w = final_coords[:, 2]
            xtl = dif.Crystal()
            xtl.Cell.latt([la, lb, lc, alpha, beta, gamma])
            xtl.new_atoms(u=u, v=v, w=w, type=final_atoms)
            xtl.write_cif(os.path.join(self.save_path, mp_id[j]+'.cif'))