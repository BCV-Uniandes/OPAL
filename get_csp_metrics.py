import warnings
import logging

# material_hasher/hasher/__init__.py calls warnings.filterwarnings("always") at module level,
# which overrides any suppression we set beforehand. We intercept that specific call.
_orig_filterwarnings = warnings.filterwarnings
def _guarded_filterwarnings(action, *args, **kwargs):
    if action == "always" and not args and not kwargs:
        return
    _orig_filterwarnings(action, *args, **kwargs)
warnings.filterwarnings = _guarded_filterwarnings
warnings.filterwarnings("ignore")
logging.raiseExceptions = False
logging.getLogger("material_hasher").setLevel(logging.CRITICAL)
logging.getLogger("lemat_genbench").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from args import get_args
from common.utils import pymatgen_structure_from_cif, expand_structures_to_lcm
from common.energy_utils import get_gt_energies, build_preprocessor, calculate_energies_batch
import Dans_Diffraction as dif
from metrics.sinkhorn import sinkhorn_awa
from metrics.chamfer import chamfer_awa
from metrics.hausdorff import hausdorff_awa
from metrics.superpose_rmsd import superpose_rmsd_awa
from metrics.density import density
from metrics.xrd_dist import xrd_metric
from metrics.soap_dist import soap
from metrics.energy_mlips_dist import energy_dist

warnings.filterwarnings = _orig_filterwarnings

ENERGY_METRICS = ('formation_energy', 'e_above_hull',
                  'relaxed_formation_energy', 'relaxed_e_above_hull')
ENERGY_FLAG = 'energies'
ALIGNMENT_METRICS = {'chamfer', 'sinkhorn', 'hausdorff', 'superpose_rmsd'}


def _worker_compute_energies(worker_args):
    chunk_ids, chunk_paths, device = worker_args
    from common.energy_utils import build_preprocessor, calculate_energies_batch
    from common.utils import pymatgen_structure_from_cif

    preprocessor = build_preprocessor(device=device, num_workers=1)

    valid_ids, valid_structs = [], []
    for pid, path in zip(chunk_ids, chunk_paths):
        try:
            valid_structs.append(pymatgen_structure_from_cif(path))
            valid_ids.append(pid)
        except Exception as e:
            print(f'Error reading {path}: {e}')

    if not valid_structs:
        return []
    try:
        results = calculate_energies_batch(valid_structs, preprocessor)
        return [(pid, res) for pid, res in zip(valid_ids, results)]
    except Exception as e:
        print(f'Worker error: {e}')
        return []


def compute_selected_metric(gt_cif, pred_cif, rotation, translation, metric, use_lcm=False, max_lcm=0):
    s_gt = pymatgen_structure_from_cif(gt_cif)
    s_pred = pymatgen_structure_from_cif(pred_cif)
    if use_lcm and metric in ALIGNMENT_METRICS:
        s_pred, s_gt, ok = expand_structures_to_lcm(s_pred, s_gt, max_lcm=max_lcm)
        if not ok:
            return 10**9
    rotation = torch.tensor(rotation)
    translation = torch.tensor(translation)

    if metric == 'sinkhorn':
        return sinkhorn_awa(s_gt, s_pred, rotation, translation)
    elif metric == 'chamfer':
        return chamfer_awa(s_gt, s_pred, rotation, translation)
    elif metric == 'hausdorff':
        return hausdorff_awa(s_gt, s_pred, rotation, translation)
    elif metric == 'superpose_rmsd':
        return superpose_rmsd_awa(s_gt, s_pred, rotation, translation)
    elif metric == 'density':
        return density(s_gt, s_pred)
    elif metric == 'xrd_dist':
        s_gt = dif.Crystal(gt_cif)
        s_pred = dif.Crystal(pred_cif)
        return xrd_metric(s_gt, s_pred)
    elif metric == 'soap_dist':
        return soap(s_gt, s_pred)
    else:
        raise Exception(f'Invalid --metric argument: {metric}')


def _precompute_pred_energies(poly_rotation_data, args, preprocessor):
    pred_id_to_path = {}
    for chem_form in poly_rotation_data:
        pred_id = poly_rotation_data[chem_form]['pred']
        pred_path = os.path.join(args.pred_path, args.method_name, 'Cifs', args.dataset_name, pred_id)
        pred_id_to_path[pred_id] = pred_path

    all_ids = list(pred_id_to_path.keys())
    all_paths = [pred_id_to_path[pid] for pid in all_ids]
    pred_energies_cache = {}

    if args.num_workers <= 1:
        valid_ids, valid_structs = [], []
        for pid, path in zip(all_ids, all_paths):
            try:
                valid_structs.append(pymatgen_structure_from_cif(path))
                valid_ids.append(pid)
            except Exception as e:
                print(f'Error reading {path}: {e}')

        for i in tqdm(range(0, len(valid_structs), args.batch_size),
                      desc="Computing pred energies", position=0, leave=True):
            batch_ids = valid_ids[i:i + args.batch_size]
            batch_structs = valid_structs[i:i + args.batch_size]
            try:
                results = calculate_energies_batch(batch_structs, preprocessor)
                for pid, res in zip(batch_ids, results):
                    pred_energies_cache[pid] = res
            except Exception as e:
                print(f'Error in batch {i // args.batch_size}: {e}')

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n = args.num_workers
        chunk_size = max(1, (len(all_ids) + n - 1) // n)
        chunks = [
            (all_ids[i:i + chunk_size], all_paths[i:i + chunk_size], device)
            for i in range(0, len(all_ids), chunk_size)
        ]

        import torch.multiprocessing as tmp
        import multiprocessing
        WORKER_TIMEOUT = 4 * 3600
        with tmp.get_context('spawn').Pool(processes=len(chunks)) as pool:
            async_results = [pool.apply_async(_worker_compute_energies, (chunk,)) for chunk in chunks]
            for ar in tqdm(async_results, total=len(chunks),
                           desc=f"Computing pred energies ({len(chunks)} workers)"):
                try:
                    chunk_results = ar.get(timeout=WORKER_TIMEOUT)
                    for pid, res in chunk_results:
                        pred_energies_cache[pid] = res
                except multiprocessing.TimeoutError:
                    print(f"Worker timed out after {WORKER_TIMEOUT}s, skipping chunk.")
                except Exception as e:
                    print(f"Worker error: {e}")

    return pred_energies_cache


def compute_metrics(args):
    poly_rotation_data = json.load(open(os.path.join(
        'results', 'alignment', args.dataset_name, args.method_name + '_alignment.json'), 'r'))

    is_energy = args.metric in ENERGY_METRICS or args.metric == ENERGY_FLAG

    if is_energy:
        gt_energies_file = os.path.join('results', 'gt_energies', f'{args.dataset_name}.json')
        if os.path.exists(gt_energies_file):
            with open(gt_energies_file) as f:
                gt_energies = json.load(f)
        else:
            gt_energies = get_gt_energies(args)

        preprocessor = build_preprocessor(num_workers=1) if args.num_workers <= 1 else None
        pred_energies_cache = _precompute_pred_energies(poly_rotation_data, args, preprocessor)
        active_metrics = list(ENERGY_METRICS)
    else:
        gt_energies = None
        pred_energies_cache = None
        active_metrics = [args.metric]

    results = {m: {'distances': [], 'errors': [], 'pred_ids': [], 'annots_ids': []}
               for m in active_metrics}

    for chem_form in tqdm(list(poly_rotation_data.keys()), desc="Computing metrics"):
        pred_id = poly_rotation_data[chem_form]['pred']
        pred_path = os.path.join(args.pred_path, args.method_name, 'Cifs', args.dataset_name, pred_id)
        poly_ids = list(poly_rotation_data[chem_form]['annots'].keys())

        best = {m: (None, None) for m in active_metrics}

        for st_id in poly_ids:
            gt_path = os.path.join(args.gt_path, args.dataset_name, st_id)
            rotation = poly_rotation_data[chem_form]['annots'][st_id]['rotation']
            translation = poly_rotation_data[chem_form]['annots'][st_id]['translation']

            if is_energy:
                try:
                    gt_energy = gt_energies[st_id]
                    pred_energy = pred_energies_cache[pred_id]
                    dists = energy_dist(gt_energy, pred_energy)
                    for m in active_metrics:
                        val = dists['ensemble'][f'{m}_mean']
                        if val is None:
                            val = 10**9
                        bval, bid = best[m]
                        if bval is None or val < bval:
                            best[m] = (val, st_id)
                except Exception:
                    for m in active_metrics:
                        bval, _ = best[m]
                        if bval is None:
                            best[m] = (10**9, st_id)
            else:
                m = args.metric
                try:
                    val = compute_selected_metric(gt_path, pred_path, rotation, translation, m,
                                                  use_lcm=args.use_lcm, max_lcm=args.max_lcm)
                except Exception:
                    val = 10**9
                bval, bid = best[m]
                if bval is None or val < bval:
                    best[m] = (val, st_id)

        for m in active_metrics:
            bval, bid = best[m]
            r = results[m]
            r['pred_ids'].append(pred_id)
            r['annots_ids'].append(bid)
            if bval is not None and bval != 10**9:
                r['distances'].append(float(bval))
            else:
                r['errors'].append(pred_id)
                r['distances'].append(-1)

    os.makedirs(os.path.join('results', 'metrics'), exist_ok=True)
    for m in active_metrics:
        r = results[m]
        metrics_dict = {
            m:            r['distances'],
            'errors':     r['errors'],
            'pred_ids':   r['pred_ids'],
            'annots_ids': r['annots_ids'],
        }
        json_name = f'{m}_{args.dataset_name}_{args.method_name}'
        with open(os.path.join('results', 'metrics', json_name + '.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)


if __name__ == '__main__':
    args = get_args()
    compute_metrics(args)
