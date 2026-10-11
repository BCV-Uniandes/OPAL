import torch
import numpy as np
import random
from math import gcd
import Dans_Diffraction as dif
from pymatgen.core import Structure, Lattice
import os
from tqdm import tqdm
from common.alignment import Hungarian_Alignment, Hungarian_Matcher


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pymatgen_structure_from_cif(path):
    try:
        pymatgen_structure = Structure.from_file(path)
    except:
        xtl = dif.Crystal(path)
        a, b, c = xtl.Cell.a, xtl.Cell.b, xtl.Cell.c
        alpha, beta, gamma = xtl.Cell.alpha, xtl.Cell.beta, xtl.Cell.gamma
        atoms = xtl.Atoms.type
        u, v, w = xtl.Atoms.u, xtl.Atoms.v, xtl.Atoms.w
        coords = np.vstack((u, v, w)).T
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        pymatgen_structure = Structure(
            lattice=lattice,
            species=atoms,
            coords=coords,
            coords_are_cartesian=False
        )
    return pymatgen_structure


def get_coords_and_atoms(structure):
    coords, atoms = [], []
    for x in structure.sites:
        x_symbol = x.specie.Z
        atoms.append(x_symbol)
        coords.append(x.coords.tolist())
    coords, atoms = torch.Tensor(coords), torch.Tensor(atoms)
    return coords, atoms


def expand_structures_to_lcm(s_pred, s_gt, max_lcm=0):
    n_pred, n_gt = len(s_pred), len(s_gt)
    if n_pred == n_gt:
        return s_pred, s_gt, True
    l = n_pred * n_gt // gcd(n_pred, n_gt)
    if max_lcm > 0 and l > max_lcm:
        return s_pred, s_gt, False
    factor_pred = l // n_pred
    factor_gt = l // n_gt
    try:
        if factor_pred > 1:
            s_pred = s_pred.copy()
            s_pred.make_supercell([factor_pred, 1, 1])
        if factor_gt > 1:
            s_gt = s_gt.copy()
            s_gt.make_supercell([factor_gt, 1, 1])
    except Exception:
        return s_pred, s_gt, False
    return s_pred, s_gt, True


def get_polymorphs(cif_paths):
    formulas = {}
    for path in tqdm(cif_paths):
        xtl = dif.Crystal(path)
        formula = xtl.Atoms.type
        formula.sort()
        formula = ''.join(formula)
        path_id = os.path.split(path)[-1]
        if formula not in formulas:
            formulas[formula] = [path_id]
        else:
            formulas[formula].append(path_id)
    return formulas


def get_alignment_inputs(poly_data, gt_path, pred_path, use_lcm=False, max_lcm=0):
    id2chem = {}
    prediction_ids = []
    annotation_ids = []
    pred_coords = []
    annotation_coords = []
    pred_atoms = []
    annotation_atoms = []
    skipped_lcm = 0
    for chem_form in tqdm(list(poly_data.keys())):
        poly_ids = poly_data[chem_form]
        pred_id = random.choice(poly_ids)
        prediction_path = os.path.join(pred_path, pred_id)
        try:
            pred_struct = pymatgen_structure_from_cif(prediction_path)
        except:
            print(f'The path for the predicted structure cif {pred_id} does not exist.')
            continue
        predicted_coords, predicted_atoms = get_coords_and_atoms(pred_struct)
        if torch.isnan(predicted_coords).any():
            print(f'Skipping {pred_id}: degenerate structure with NaN coordinates.')
            continue
        id2chem[pred_id] = chem_form
        for st_id in poly_ids:
            annot_path = os.path.join(gt_path, st_id)
            assert os.path.exists(annot_path), f'The path for the ground truth structure cif {st_id} does not exist.'
            gt_struct = pymatgen_structure_from_cif(annot_path)
            if use_lcm:
                pred_exp, gt_exp, ok = expand_structures_to_lcm(pred_struct, gt_struct, max_lcm=max_lcm)
                if not ok:
                    skipped_lcm += 1
                    continue
                p_coords, p_atoms = get_coords_and_atoms(pred_exp)
                g_coords, g_atoms = get_coords_and_atoms(gt_exp)
            else:
                p_coords, p_atoms = predicted_coords, predicted_atoms
                g_coords, g_atoms = get_coords_and_atoms(gt_struct)
            pred_coords.append(p_coords)
            pred_atoms.append(p_atoms)
            annotation_coords.append(g_coords)
            annotation_atoms.append(g_atoms)
            prediction_ids.append(pred_id)
            annotation_ids.append(st_id)
    if skipped_lcm > 0:
        print(f'LCM alignment: skipped {skipped_lcm} pairs where LCM > {max_lcm} atoms.')
    return id2chem, prediction_ids, annotation_ids, pred_coords, annotation_coords, pred_atoms, annotation_atoms


def preprocess_alignment_inputs(coords, atoms, total_positions=52):
    processed_coords = torch.zeros((len(coords), total_positions, 3)) - 1
    processed_atoms = torch.zeros((len(atoms), total_positions)) - 1
    lens = []
    for i in range(len(coords)):
        coords_i = coords[i]
        atoms_i = atoms[i]
        lens.append(len(coords_i))
        processed_coords[i, :len(coords_i), :] = coords_i
        processed_atoms[i, :len(atoms_i)] = atoms_i
    return processed_coords, processed_atoms, lens


def get_alignment_outputs(inputs, total_positions=None, batch_size=64):
    id2chem, prediction_ids, annotation_ids, pred_coords, annotation_coords, pred_atoms, annotation_atoms = inputs
    rotations_dict = {}
    n = len(pred_coords)

    order = sorted(range(n), key=lambda i: len(pred_coords[i]), reverse=True)

    all_rotations = [None] * n
    all_ts = [None] * n

    for start in tqdm(range(0, n, batch_size), desc="Alignment batches"):
        batch_idx = order[start:start + batch_size]
        b_pred_coords = [pred_coords[i] for i in batch_idx]
        b_ann_coords = [annotation_coords[i] for i in batch_idx]
        b_pred_atoms = [pred_atoms[i] for i in batch_idx]
        b_ann_atoms = [annotation_atoms[i] for i in batch_idx]

        if total_positions is None:
            batch_total = max(
                max(len(c) for c in b_pred_coords),
                max(len(c) for c in b_ann_coords),
            )
        else:
            batch_total = total_positions

        b_pred_coords, b_pred_atoms, b_pred_lens = preprocess_alignment_inputs(
            b_pred_coords, b_pred_atoms, total_positions=batch_total)
        b_ann_coords, b_ann_atoms, _ = preprocess_alignment_inputs(
            b_ann_coords, b_ann_atoms, total_positions=batch_total)

        rots, ts = Hungarian_Alignment(
            b_pred_coords, b_ann_coords, b_pred_atoms, b_ann_atoms, b_pred_lens)

        for local_i, global_i in enumerate(batch_idx):
            all_rotations[global_i] = rots[local_i]
            all_ts[global_i] = ts[local_i]

    for i in range(n):
        pred_id = prediction_ids[i]
        annot_id = annotation_ids[i]
        chemical_formula = id2chem[pred_id]
        aux_dict = {'rotation': all_rotations[i], 'translation': all_ts[i]}
        if chemical_formula not in rotations_dict:
            rotations_dict[chemical_formula] = {'pred': pred_id, 'annots': {annot_id: aux_dict}}
        else:
            rotations_dict[chemical_formula]['annots'][annot_id] = aux_dict
    return rotations_dict


def awa_metric(s_gt, s_pred, rotation, translation, metric):
    assert isinstance(s_gt, Structure), "The input ground truth structure must be a pymatgen Structure."
    assert isinstance(s_pred, Structure), "The input predicted structure must be a pymatgen Structure."
    total_natoms = len(s_gt.sites)
    pred_coords, pred_atoms = get_coords_and_atoms(s_pred)
    gt_coords, gt_atoms = get_coords_and_atoms(s_gt)
    X, Y, Z = Hungarian_Matcher(pred_coords, gt_coords, pred_atoms, gt_atoms, rotation, translation)
    gt, pred = {}, {}
    for x, y, z in zip(X.tolist(), Y.tolist(), Z.tolist()):
        z = str(int(z))
        if z not in gt.keys():
            gt[z] = []
        if z not in pred.keys():
            pred[z] = []
        pred[z].append(x)
        gt[z].append(y)

    if not gt:
        return np.nan

    weights = {}
    metric_dict = {}
    for atom in gt.keys():
        weights[atom] = len(gt[atom]) / total_natoms
        a = np.array(gt[atom])
        b = np.array(pred[atom])
        metric_dict[atom] = metric(a, b)
    w_metric = [weights[atom] * metric_dict[atom] for atom in gt.keys()]
    return np.sum(w_metric)
