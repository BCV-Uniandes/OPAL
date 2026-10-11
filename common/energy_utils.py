import os
import glob
import json
import multiprocessing
import torch
from tqdm import tqdm
from pymatgen.core import Structure
from common.utils import pymatgen_structure_from_cif
from lemat_genbench.preprocess.multi_mlip_preprocess import MultiMLIPStabilityPreprocessor


def build_preprocessor(device: str = None, num_workers: int = None) -> MultiMLIPStabilityPreprocessor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if num_workers is None:
        num_workers = 1 if device == "cuda" else multiprocessing.cpu_count()
    return MultiMLIPStabilityPreprocessor(
        mlip_names=["orb", "mace", "uma"],
        mlip_configs={
            "orb":  {"model_type": "orb_v3_conservative_inf_omat", "device": device},
            "mace": {"model_type": "mp", "device": device},
            "uma":  {"task": "omat", "device": device},
        },
        relax_structures=True,
        relaxation_config={"fmax": 0.02, "steps": 50},
        calculate_formation_energy=True,
        calculate_energy_above_hull=True,
        extract_embeddings=False,
        timeout=300,
        n_jobs=num_workers,
    )


def calculate_energies_batch(structures: list, preprocessor: MultiMLIPStabilityPreprocessor) -> list:
    result = preprocessor(structures)
    mlips = ["orb", "mace", "uma"]
    all_results = []
    for processed in result.processed_structures:
        props = processed.properties
        res = {}
        for mlip in mlips:
            res[mlip] = {
                "formation_energy":         props.get(f"formation_energy_{mlip}"),
                "e_above_hull":             props.get(f"e_above_hull_{mlip}"),
                "relaxed_formation_energy": props.get(f"relaxed_formation_energy_{mlip}"),
                "relaxed_e_above_hull":     props.get(f"relaxed_e_above_hull_{mlip}"),
            }
        res["ensemble"] = {
            "formation_energy_mean":         props.get("formation_energy_mean"),
            "formation_energy_std":          props.get("formation_energy_std"),
            "e_above_hull_mean":             props.get("e_above_hull_mean"),
            "e_above_hull_std":              props.get("e_above_hull_std"),
            "relaxed_formation_energy_mean": props.get("relaxed_formation_energy_mean"),
            "relaxed_formation_energy_std":  props.get("relaxed_formation_energy_std"),
            "relaxed_e_above_hull_mean":     props.get("relaxed_e_above_hull_mean"),
            "relaxed_e_above_hull_std":      props.get("relaxed_e_above_hull_std"),
        }
        all_results.append(res)
    return all_results


def _worker_gt_energies(worker_args):
    chunk_paths, device = worker_args
    from common.energy_utils import build_preprocessor, calculate_energies_batch
    from common.utils import pymatgen_structure_from_cif

    preprocessor = build_preprocessor(device=device, num_workers=1)

    valid_paths, valid_structs = [], []
    for path in chunk_paths:
        try:
            valid_structs.append(pymatgen_structure_from_cif(path))
            valid_paths.append(path)
        except Exception as e:
            print(f'Error reading {path}: {e}')

    if not valid_structs:
        return {}
    try:
        results = calculate_energies_batch(valid_structs, preprocessor)
        return {os.path.split(p)[-1]: res for p, res in zip(valid_paths, results)}
    except Exception as e:
        print(f'GT worker error: {e}')
        return {}


def get_gt_energies(args) -> dict:
    print(f'Calculating gt energies for {args.dataset_name}...')
    gt_energies_path = os.path.join('results', 'gt_energies')
    os.makedirs(gt_energies_path, exist_ok=True)
    cif_paths = glob.glob(os.path.join(args.gt_path, args.dataset_name, '*.cif'))

    valid_paths = []
    for path in tqdm(cif_paths, desc="Reading CIFs"):
        try:
            pymatgen_structure_from_cif(path)
            valid_paths.append(path)
        except Exception as e:
            print(f'Error reading {path}: {e}')

    num_workers = getattr(args, 'num_workers', 1)
    batch_size = getattr(args, 'batch_size', 32)
    gt_energies_dict = {}
    out_file = os.path.join(gt_energies_path, args.dataset_name + '.json')

    if num_workers <= 1:
        preprocessor = build_preprocessor(num_workers=1)
        batches = [(i, valid_paths[i:i + batch_size])
                   for i in range(0, len(valid_paths), batch_size)]

        for batch_idx, batch_paths in tqdm(batches, desc="Computing GT energies",
                                           position=0, leave=True):
            batch_structs = [pymatgen_structure_from_cif(p) for p in batch_paths]
            try:
                batch_results = calculate_energies_batch(batch_structs, preprocessor)
            except Exception as e:
                print(f'Error in batch {batch_idx}: {e}')
                continue
            for path, res in zip(batch_paths, batch_results):
                gt_energies_dict[os.path.split(path)[-1]] = res
            with open(out_file, 'w') as f:
                json.dump(gt_energies_dict, f, indent=2)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chunk_size = max(1, (len(valid_paths) + num_workers - 1) // num_workers)
        chunks = [
            (valid_paths[i:i + chunk_size], device)
            for i in range(0, len(valid_paths), chunk_size)
        ]

        import torch.multiprocessing as tmp
        WORKER_TIMEOUT = 4 * 3600
        with tmp.get_context('spawn').Pool(processes=len(chunks)) as pool:
            async_results = [pool.apply_async(_worker_gt_energies, (chunk,)) for chunk in chunks]
            for ar in tqdm(async_results, total=len(chunks),
                           desc=f"Computing GT energies ({len(chunks)} workers)"):
                try:
                    gt_energies_dict.update(ar.get(timeout=WORKER_TIMEOUT))
                except multiprocessing.TimeoutError:
                    print(f"Worker timed out after {WORKER_TIMEOUT}s, skipping chunk.")
                except Exception as e:
                    print(f"GT worker error: {e}")

        with open(out_file, 'w') as f:
            json.dump(gt_energies_dict, f, indent=2)

    return gt_energies_dict
