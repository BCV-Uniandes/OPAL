import os
import Dans_Diffraction as dif
from py_files.utils.data_utils import lattice_polar_build_torch, lattice_matrix_to_params
from PyAstronomy.pyasl import AtomicNo
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from torch_geometric.nn import radius_graph
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

try:
    from e3nn import o3
except ImportError:
    pass

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from .transformer_block_adanorm import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from .input_block import EdgeDegreeEmbedding
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from py_files.utils.loss_utils import UncertaintyWeightingLoss, GradNorm






@torch.no_grad()
def lattice_polar_build_torch(k):
    assert k.dim() == 2, "input must be batched k of shape (B,6)"
    S0 = torch.stack([k[:, 3] + k[:, 4] + k[:, 5], k[:, 0], k[:, 1]], dim=1)  # (B, 3)
    S1 = torch.stack([k[:, 0], -k[:, 3] + k[:, 4] + k[:, 5], k[:, 2]], dim=1)  # (B, 3)
    S2 = torch.stack([k[:, 1], k[:, 2], -2 * k[:, 4] + k[:, 5]], dim=1)  # (B, 3)
    S = torch.stack([S0, S1, S2], dim=1)  # (B, 3, 3)
    expS = torch.matrix_exp(S)  # (B, 3, 3)
    return expS



def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


class SinusoidalTimeEmbeddings(nn.Module):

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


class CommonLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x



class EquiformerV2(pl.LightningModule):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self, method, 
        lparams, 
        generator,
        use_pbc=False,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=12.0,
        max_num_elements=100,

        num_layers=6,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        
        norm_type='layer_norm_sh',
        
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=18, 

        num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False,
        use_grid_mlp=True, 
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='uniform',
        time_dim=128, 
        cell_dim=128,
        sg_emb_dim=128,
        use_cfg_sg = False,
        use_cfg_atoms = False,
        cfg_eta = 0.1,
        cfg_w = 3.0,
        use_frac_coords = False,
        loss_type = 'manual_w',
        lr = 1e-03,
        scheduler_factor = 0.6,
        scheduler_patience = 30,
        avg_degree = None,
        weight_decay = 0.0,
        scheduler_monitor = 'val_normalized_loss'
    ):
        super().__init__()

        self.save_path = None

        self.use_cfg_sg = use_cfg_sg
        self.use_cfg_atoms = use_cfg_atoms
        self.use_frac_coords = use_frac_coords
        self.loss_type = loss_type
        # GradNorm requires two separate optimizer steps per batch
        self.automatic_optimization = (loss_type != 'gradnorm')
        
        # Optimizer parametrers
        self.cfg_eta = cfg_eta
        self.cfg_w = cfg_w
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_monitor = scheduler_monitor

        # Method & Loss hyperparameters
        self.method = method
        self.alpha, self.beta = lparams
        self.generator = generator

        # avg_degree
        self.avg_degree = avg_degree

        # Embedding Dimensions
        self.time_dim = time_dim
        self.cell_dim = cell_dim
        self.sg_emb_dim = sg_emb_dim

        # EquiformerV2 parameters
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        self.t_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        
        if self.use_cfg_sg:
            self.sg_embedding = nn.Embedding(230, self.sg_emb_dim)

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)
        
        self.edge_degree_embedding = None

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        
        # Final norm. energy_block and force_block are unused leftovers from EquiformerV2's
        # OC20 heads, kept so that existing checkpoints still load; the outputs are
        # coord_out and lattice_out below.
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads, 
                self.attn_alpha_channels,
                self.attn_value_channels, 
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation, 
                self.mappingReduced, 
                self.SO3_grid, 
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding, 
                self.use_m_share_rad,
                self.attn_activation, 
                self.use_s2_act_attn, 
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        self.lt_pol_emb = CommonLayer(6, self.cell_dim) 
        self.time_cat_proj = nn.Linear(self.sphere_channels + self.cell_dim, self.sphere_channels)
        self.num_coefficients = [int((self.lmax_list[i] + 1) ** 2) for i in range(len(self.lmax_list))][0]
        self.coord_out = nn.Linear(self.sphere_channels * self.num_coefficients, 3, bias = False)
        self.lattice_out = nn.Linear(self.sphere_channels * self.num_coefficients, 6, bias = False)
        if self.loss_type in ('uw_loss', 'uw_loss_norm'):
            self.criterion = UncertaintyWeightingLoss(self.alpha, self.beta)
        elif self.loss_type == 'gradnorm':
            self.criterion = GradNorm(num_tasks=2, alpha=1.5)
            
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)


    @conditional_grad(torch.enable_grad())
    def forward(self, t, atom_types, frac_coords, l_t, l_t_polar, num_atoms, node2graph, spacegroup=None):
        if self.avg_degree is None or self.edge_degree_embedding is None:
            raise RuntimeError("Call estimate_avg_degree() first...")

        t = self.t_embedding(t)
        if self.use_cfg_sg:
            if spacegroup is not None:
                t += self.sg_embedding(spacegroup)
        l_t_polar = self.lt_pol_emb(l_t_polar)

        cart_coords = torch.einsum('ni,nij->nj', frac_coords, l_t[node2graph])

        if self.use_frac_coords:
            data = Data(pos=frac_coords, cell=l_t, atomic_numbers=atom_types, natoms=num_atoms, batch=node2graph, time_emb=t)
        else:
            data = Data(pos=cart_coords, cell=l_t, atomic_numbers=atom_types, natoms=num_atoms, batch=node2graph, time_emb=t)
        self.batch_size = len(data.natoms)
        self.t = data.time_emb

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos
        
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        self.t = self.t.repeat_interleave(data.natoms, dim=0)
        self.t = self.t.unsqueeze(1)
        self.t = self.t.repeat(1, self.num_coefficients, 1)
        l_t_polar = l_t_polar.repeat_interleave(data.natoms, dim=0)
        l_t_polar = l_t_polar.unsqueeze(1)
        l_t_polar = l_t_polar.repeat(1, self.num_coefficients, 1)
        x.embedding = torch.cat([x.embedding, l_t_polar], dim=2)
        x.embedding = self.time_cat_proj(x.embedding)
        

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  
                self.t,                # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        x.embedding = x.embedding.view(x.embedding.shape[0], -1)

        ###############################################################
        # Coordinates estimation
        ###############################################################
        coord_out = self.coord_out(x.embedding) # velocity field for fractional coordinates

        ###############################################################
        # Lattice estimation
        ###############################################################
        graph_features = scatter(x.embedding, data.batch, dim = 0, reduce = 'mean') # mean-pool node features per crystal
        lattice_out = self.lattice_out(graph_features) # velocity field for the 6D polar lattice

        return lattice_out, coord_out


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
    

    @torch.no_grad()
    def estimate_avg_degree(self, dataloader, model, n_batches=50):
        tot_deg, tot_nodes = 0.0, 0
        for i, batch in tqdm(enumerate(dataloader)):
            if i == n_batches: break
            coords, f_atoms, mask, lenghts, angles, num_atoms, lattice_polar, mp_id, spacegroup = batch

            mask = mask.squeeze()
            pos = coords[mask].view(-1, 3)
            atoms = f_atoms[mask]
            num_atoms = mask.sum(dim = -1)
            ar = torch.arange(0, mask.shape[0]).to(mask.device)
            ar = ar.unsqueeze(-1)
            node2graph = mask + ar
            node2graph = node2graph[mask] - 1

            lattice = lattice_polar_build_torch(lattice_polar) #matrix rep of polar lt
            
            cart_coords = torch.einsum('ni,nij->nj', pos, lattice[node2graph])

            pre_data = Data(pos=cart_coords, cell=lattice, atomic_numbers=atoms, natoms=num_atoms, batch=node2graph)

            ei, _, _, _, _, _ = model.generate_graph(
                pre_data,
                cutoff=model.cutoff,
                max_neighbors=model.max_neighbors,
                use_pbc=model.use_pbc,
                otf_graph=True,
            )
            
            deg = torch.bincount(ei[1], minlength=pre_data.pos.size(0))  # in-degree of targets
            tot_deg += deg.sum().item()
            tot_nodes += deg.numel()

        self.avg_degree = tot_deg / tot_nodes

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree
        )

        print(f'Using PBC: {model.use_pbc}')
        print(f'The Average Degree is : {self.avg_degree}')



    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, cutoff, max_neighbors
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )
    

    @torch.no_grad()
    def _process_batch(self, batch, t):
        coords_t, f_atoms, mask, lenghts, angles, num_atoms, l_t, mp_id, spacegroup = batch
        device = self.device
        coords_t, f_atoms, mask, num_atoms, l_t = coords_t.to(device), f_atoms.to(device), mask.to(device), num_atoms.to(device), l_t.to(device)
        t = t * torch.ones(coords_t.shape[0], device=self.device)
        lattice_non_polar_t = lattice_polar_build_torch(l_t)
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
        l_cond, coords_cond = self.forward(t, atoms, coords_t, lattice_non_polar_t, l_t, num_atoms, node2graph, spacegroup)
        if self.use_cfg_sg:
            l_uncond, coords_uncond = self.forward(t, atoms, coords_t, lattice_non_polar_t, l_t, num_atoms, node2graph)
            l_out = torch.lerp(l_uncond, l_cond, self.cfg_w) # Guided lattice vector field output
            coords_out = torch.lerp(coords_uncond, coords_cond, self.cfg_w) # Guided coordinates vector field output
        elif self.use_cfg_atoms:
            atoms[atoms!= 0] = 99
            l_uncond, coords_uncond = self.forward(t, atoms, coords_t, lattice_non_polar_t, l_t, num_atoms, node2graph)
            l_out = torch.lerp(l_uncond, l_cond, self.cfg_w) # Guided lattice vector field output
            coords_out = torch.lerp(coords_uncond, coords_cond, self.cfg_w) # Guided coordinates vector field output
        else:
            coords_out, l_out = coords_cond, l_cond
        coords_out_f[mask] = coords_out.flatten()
        return l_out, coords_out_f


    def _common_step(self, batch, batch_idx):
        coords, f_atoms, mask, lenghts, angles, num_atoms, lattice_polar, mp_id, spacegroup = batch
        t = torch.rand(coords.shape[0], device=self.device, generator=self.generator)
        coords_t, coords_tar = self.method.get_noised_coords(coords, t)
        l_t, l_tar =  self.method.get_noised_lattice(lattice_polar, t) # l_t and l_tar in polar rep
        lattice_non_polar_t = lattice_polar_build_torch(l_t) #matrix rep of polar lt
        l_tar = l_tar.to(torch.float32)
        mask = mask.squeeze()
        cfg_flag = torch.rand(1, device=self.device, generator=self.generator).item()
        if cfg_flag < self.cfg_eta:
            if self.use_cfg_sg:
                spacegroup = None
            elif self.use_cfg_atoms:
                f_atoms[f_atoms!= 0] = 99 #99 is used to represent any atom. 
        atoms = f_atoms[mask]
        num_atoms = mask.sum(dim = -1)
        ar = torch.arange(0, mask.shape[0]).to(mask.device)
        ar = ar.unsqueeze(-1)
        node2graph = mask + ar
        node2graph = node2graph[mask] - 1
        mask = mask.unsqueeze(-1).repeat(1, 1, 3)
        coords_t = coords_t[mask].view(-1, 3)
        coords_tar = coords_tar[mask].view(-1,3)
        l_t = l_t.to(torch.float32)
        num_atoms = num_atoms.to(torch.int32)
        l_out, coords_out = self.forward(t, atoms, coords_t, lattice_non_polar_t, l_t, num_atoms, node2graph, spacegroup)
        l_loss = F.mse_loss(l_out, l_tar)
        coords_loss = F.mse_loss(coords_out, coords_tar)
        if self.loss_type == 'manual_w':
            loss = self.alpha * coords_loss + self.beta * l_loss
        elif self.loss_type == 'uw_loss':
            loss, info = self.criterion(coords_loss, l_loss)
            self.log_dict(info, batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.loss_type == 'uw_loss_norm':
            loss, info = self.criterion(coords_loss/self.method.coords_loss_0, l_loss/self.method.l_loss_0)
            self.log_dict(info, batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.loss_type == 'gradnorm':
            # Use current normalized weights (detached) for monitoring only.
            # The actual GradNorm optimization happens in training_step.
            w = self.criterion.weights.detach()
            w = w * self.criterion.num_tasks / w.sum()
            loss = w[0] * coords_loss + w[1] * l_loss
        return loss, coords_loss, l_loss
    

    def training_step(self, batch, batch_idx):
        loss, coords_loss, l_loss = self._common_step(batch, batch_idx)

        if self.loss_type == 'gradnorm':
            opt_main, opt_gn = self.optimizers()

            # shared_params: last shared layer before the two output heads
            shared_params = list(self.norm.parameters())

            _, gradnorm_loss = self.criterion.compute_loss(
                [coords_loss, l_loss], shared_params
            )

            # Step 1: update GradNorm weights w_i via gradnorm_loss
            # retain_graph=True preserves the graph from coords_loss/l_loss to model params
            opt_gn.zero_grad()
            self.manual_backward(gradnorm_loss, retain_graph=True)
            opt_gn.step()

            # Renormalize weights so they always sum to num_tasks (and stay positive)
            with torch.no_grad():
                self.criterion.weights.data.clamp_(min=0)
                self.criterion.weights.data *= (
                    self.criterion.num_tasks / self.criterion.weights.data.sum()
                )

            # Step 2: recompute total_loss with updated detached weights to avoid
            # in-place version conflict (opt_gn.step() modified weights in-place)
            w = self.criterion.weights.detach()
            total_loss = w[0] * coords_loss + w[1] * l_loss
            opt_main.zero_grad()
            self.manual_backward(total_loss)
            opt_main.step()

            loss = total_loss.detach()
            self.log_dict(
                {'gradnorm_w_coords': self.criterion.weights[0].detach(),
                 'gradnorm_w_lattice': self.criterion.weights[1].detach(),
                 'gradnorm_loss': gradnorm_loss.detach()},
                batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
            )

        self.log_dict(
            {'train_joint_loss': loss.detach(),
             'train_coords_loss': coords_loss.detach(),
             'train_lattice_loss': l_loss.detach(),
             'train_unweighted_loss': coords_loss.detach() + l_loss.detach()
            },
            batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        loss, coords_loss, l_loss = self._common_step(batch, batch_idx)
        # Normalized loss: each task divided by its zero-predictor baseline
        # (estimated from the training set), so both tasks contribute equally
        # to the scheduler signal regardless of their natural scale difference.
        val_normalized_loss = (
            coords_loss / self.method.coords_loss_0
            + l_loss    / self.method.l_loss_0
        )
        self.log_dict(
            {'val_joint_loss':      loss.detach(),
             'val_coords_loss':     coords_loss.detach(),
             'val_lattice_loss':    l_loss.detach(),
             'val_unweighted_loss': coords_loss.detach() + l_loss.detach(),
             'val_normalized_loss': val_normalized_loss.detach(),
            },
            batch_size=len(batch), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.flow_sampler(batch, steps=1000)

    def on_save_checkpoint(self, checkpoint):
        if self.generator is not None:
            checkpoint['generator_state'] = self.generator.get_state()

    def on_load_checkpoint(self, checkpoint):
        if self.generator is not None and 'generator_state' in checkpoint:
            self.generator.set_state(checkpoint['generator_state'])

    def configure_optimizers(self):
        scheduler_cfg = {"monitor": self.scheduler_monitor}
        if self.loss_type == 'gradnorm':
            # Exclude criterion.weights from the main optimizer — they have their own.
            model_params = [p for name, p in self.named_parameters()
                            if not name.startswith('criterion.')]
            opt_main = optim.Adam(model_params, lr=self.lr)
            opt_gn = optim.Adam([self.criterion.weights], lr=1e-2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt_main, 'min', self.scheduler_factor, self.scheduler_patience)
            return (
                [opt_main, opt_gn],
                [{"scheduler": scheduler, **scheduler_cfg}],
            )
        if self.weight_decay > 0.0:
            decay_params = [p for name, p in self.named_parameters()
                            if not name.startswith('criterion.')]
            no_decay_params = [p for name, p in self.named_parameters()
                               if name.startswith('criterion.')]
            param_groups = [
                {'params': decay_params, 'weight_decay': self.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ]
            optimizer = optim.AdamW(param_groups, lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', self.scheduler_factor, self.scheduler_patience)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, **scheduler_cfg}}


    def flow_sampler(self, batch, steps=1000):
        generator = self.generator
        an = AtomicNo()
        coords, f_atoms, mask, lenghts, angles, num_atoms, lattice_polar, mp_id, spacegroup = batch
        l_t = torch.randn(lattice_polar.shape, generator=generator, device=self.device) * self.method.lattice_polar_sigma
        l_t[:, -1] = l_t[:, -1] + self.method.lattice_source_shift
        x_t = torch.rand(coords.shape, generator=generator, device=self.device)
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


