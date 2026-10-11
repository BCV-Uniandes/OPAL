import os
import glob
import json
import scipy
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_umap
from pyxtal import pyxtal
from collections import Counter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def write_cifs_from_csvs(csv_path, split):
    df = pd.read_csv(csv_path)
    cifs = [cif for cif in df['cif']]
    mat_ids = [id for id in df['material_id']]
    split_path = csv_path.split('/')[:-1]
    split_path.append(split)
    split_path = '/'.join(split_path)
    os.makedirs(split_path, exist_ok=True)
    for cif, m_id in tqdm(zip(cifs, mat_ids)):
        with open(os.path.join(split_path, f'{m_id}.cif'), 'w') as F:
            F.write(cif)
        F.close()
    return glob.glob(os.path.join(split_path, '*.cif'))


def lattice_polar_decompose(lattice):
    assert lattice.ndim == 2
    A, U = np.linalg.eigh(lattice @ lattice.T)
    A, U = np.real(A), np.real(U)
    A = np.diag(np.log(A)) / 2
    S = U @ A @ U.T
    k = np.array(
        [
            S[0, 1],
            S[0, 2],
            S[1, 2],
            (S[0, 0] - S[1, 1]) / 2,
            (S[0, 0] + S[1, 1] - 2 * S[2, 2]) / 6,
            (S[0, 0] + S[1, 1] + S[2, 2]) / 3,
        ]
    )
    return k


def lattice_polar_build(k):
    assert k.ndim == 1
    S = np.array(
        [
            [k[3] + k[4] + k[5], k[0], k[1]],
            [k[0], -k[3] + k[4] + k[5], k[2]],
            [k[1], k[2], -2 * k[4] + k[5]],
        ]
    )
    expS = scipy.linalg.expm(S)
    return expS


def build_crystal(crystal_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        crystal = Structure.from_file(crystal_path)
    crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal


def build_crystal_arrays(crystal): 
    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    lattice_polar = lattice_polar_decompose(crystal.lattice.matrix)
    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    num_atoms = atom_types.shape[0]
    return frac_coords, atom_types, lengths, angles, num_atoms, lattice_polar


def find_symm_map(frac_coords, general_wyckoff_ops):
    frac_coords = frac_coords % 1
    transformed = np.einsum('pij,nj->pni', general_wyckoff_ops[:, :3, :3], frac_coords)
    transformed += general_wyckoff_ops[:, :3, 3][:, None, :]
    diff = frac_coords[None, None, :, :] - transformed[:, :, None, :]
    diff -= np.round(diff)
    norms = np.linalg.norm(diff, axis=-1)
    symm_map = np.argmin(norms, axis=-1)
    return symm_map


def get_symmetry_info_consistent(crystal, tol=0.01):
    original = crystal.copy()
    n0 = len(original)
    species0 = [s.specie for s in original.sites]
    multiset0 = Counter(species0)
    spga0 = SpacegroupAnalyzer(original, symprec=tol)
    ds0 = spga0.get_symmetry_dataset()
    sg_num0 = int(ds0["number"])
    general_ops0 = np.array([op.affine_matrix for op in spga0.get_symmetry_operations()])
    try:
        spga = SpacegroupAnalyzer(crystal, symprec=tol)
        crystal_refined = spga.get_refined_structure()
        c = pyxtal()
        try:
            c.from_seed(crystal_refined, tol=0.01)
        except Exception:
            c.from_seed(crystal_refined, tol=0.0001)
        space_group = c.group.number
        species, anchors, matrices, coords = [], [], [], []
        for site in c.atom_sites:
            specie = site.specie
            anchor = len(matrices)
            coord = site.position
            for syms in site.wp:
                species.append(specie)
                matrices.append(syms.affine_matrix)
                coords.append(syms.operate(coord))
                anchors.append(anchor)
        anchors = np.array(anchors)
        matrices = np.array(matrices)
        coords = np.array(coords) % 1.0
        general_wyckoff_ops = np.array([op.affine_matrix for op in c.group[0].ops])
        symm_map = find_symm_map(coords, general_wyckoff_ops)
        sym_info_full = {
            "anchors": anchors,
            "wyckoff_ops": matrices,
            "spacegroup": space_group,
            "general_wyckoff_ops": general_wyckoff_ops,
            "symm_map": symm_map,
            "source": "symmetrized",
        }
        crystal_sym = Structure(
            lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
            species=species,
            coords=coords,
            coords_are_cartesian=False,
        )
        same_count = (len(crystal_sym) == n0)
        same_comp  = (Counter([s.specie for s in crystal_sym.sites]) == multiset0)
        if same_count and same_comp:
            print(f'Both count & composition are: {same_count and same_comp}')
            return crystal_sym, sym_info_full
        print(f'Both count & composition are: {same_count and same_comp}')
    except Exception:
        pass
    sym_info_minimal = {
        "anchors": None,                 
        "wyckoff_ops": None,             
        "spacegroup": sg_num0,           
        "general_wyckoff_ops": general_ops0,
        "symm_map": None,
        "source": "original_fallback",   
    }
    return original, sym_info_minimal


def process_one(path, use_space_group):
    crystal = build_crystal(path)
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info_consistent(crystal)
        result_dict.update(sym_info)
    else:
        result_dict.update({'spacegroup': 1})
    crist_arrays = build_crystal_arrays(crystal)
    crist_id = os.path.split(path)[-1][:-4]
    result_dict.update({
        'mp_id': crist_id,
        'crist_arrays': crist_arrays 
    })
    return result_dict


def preprocess_files(paths, num_workers, use_space_group = False):
    results = p_umap(
        process_one,
        [paths[idx] for idx in range(len(paths))],
        [use_space_group] * len(paths),
        num_cpus=num_workers)
    return results


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


def abs_cap(val, max_abs_val=1):
    return max(min(val, max_abs_val), -max_abs_val)



def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix ** 2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) /
                            (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma

