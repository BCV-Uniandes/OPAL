import os
import json

METRICS_DIR       = "results/metrics"
ALIGNMENT_DIR     = "results/alignment"
INTERSECTIONS_DIR = "results/intersections"

ALIGNMENT_METRICS  = ["chamfer", "sinkhorn", "hausdorff", "superpose_rmsd"]
ENERGY_METRICS     = ["formation_energy", "e_above_hull", "relaxed_formation_energy", "relaxed_e_above_hull"]
STRUCTURAL_METRICS = ["density", "soap_dist", "xrd_dist"]
ALL_METRICS        = ALIGNMENT_METRICS + ENERGY_METRICS + STRUCTURAL_METRICS

DATASETS = ["carbon_24", "mp_20", "perov_5", "mpts_52"]

CSP_METHODS   = ["CDVAE", "CrystalFlow", "DiffCSP", "FlowLLM", "FlowMM", "OPAL-CSP", "MatterGen"]
SICSP_METHODS = ["DiffCSP++GT", "DiffCSP++CSPML", "SGEquiDiff-GT", "SGEquiDiff-CSPML"]
ALL_METHODS   = CSP_METHODS + SICSP_METHODS


def _load_predid_to_formula(dataset, methods):
    predid_to_formula = {}
    for method in methods:
        apath = os.path.join(ALIGNMENT_DIR, dataset, f"{method}_alignment.json")
        if not os.path.exists(apath):
            predid_to_formula[method] = {}
            continue
        with open(apath) as f:
            aln = json.load(f)
        predid_to_formula[method] = {d["pred"]: formula for formula, d in aln.items()}
    return predid_to_formula


def _valid_formulas(method, dataset, metric, predid_to_formula):
    fpath = os.path.join(METRICS_DIR, f"{metric}_{dataset}_{method}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        data = json.load(f)
    pred_ids  = data["pred_ids"]
    distances = data[metric]
    p2f = predid_to_formula.get(method, {})
    if p2f:
        return {p2f[pid] for pid, d in zip(pred_ids, distances) if d != -1 and pid in p2f}
    return {pid for pid, d in zip(pred_ids, distances) if d != -1}


def _filter_and_write(method, dataset, metric, intersection, predid_to_formula):
    fpath = os.path.join(METRICS_DIR, f"{metric}_{dataset}_{method}.json")
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        data = json.load(f)

    pred_ids   = data["pred_ids"]
    distances  = data[metric]
    annots_ids = data["annots_ids"]
    p2f        = predid_to_formula.get(method, {})

    if p2f:
        keep = [i for i, pid in enumerate(pred_ids) if p2f.get(pid) in intersection]
    else:
        keep = [i for i, pid in enumerate(pred_ids) if pid in intersection]

    filtered = {
        metric:       [distances[i] for i in keep],
        "errors":     [],
        "pred_ids":   [pred_ids[i]   for i in keep],
        "annots_ids": [annots_ids[i] for i in keep],
    }
    with open(fpath, "w") as f:
        json.dump(filtered, f, indent=2)


def _load_poly_total(dataset):
    with open(os.path.join("poly_files", f"{dataset}.json")) as f:
        return len(json.load(f))


def run_filter():
    os.makedirs(INTERSECTIONS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    for dataset in DATASETS:
        total = _load_poly_total(dataset)

        precomputed_path = os.path.join(INTERSECTIONS_DIR, f"{dataset}.json")
        if os.path.exists(precomputed_path):
            with open(precomputed_path) as f:
                precomputed = json.load(f)
        else:
            precomputed = None

        present_methods = [
            m for m in ALL_METHODS
            if any(os.path.exists(os.path.join(METRICS_DIR, f"{metric}_{dataset}_{m}.json"))
                   for metric in ALL_METRICS)
        ]
        predid_to_formula = _load_predid_to_formula(dataset, present_methods)

        dataset_intersections = {}

        for metric in ALL_METRICS:
            present_csp   = [m for m in present_methods if m in CSP_METHODS]
            present_sicsp = [m for m in present_methods if m in SICSP_METHODS]

            if precomputed and metric in precomputed:
                csp_isect   = set(precomputed[metric]["csp"])
                sicsp_isect = set(precomputed[metric]["sicsp"])
            else:
                csp_valid   = {m: _valid_formulas(m, dataset, metric, predid_to_formula)
                               for m in present_csp}
                sicsp_valid = {m: _valid_formulas(m, dataset, metric, predid_to_formula)
                               for m in present_sicsp}
                csp_valid   = {m: s for m, s in csp_valid.items()   if s is not None}
                sicsp_valid = {m: s for m, s in sicsp_valid.items() if s is not None}
                csp_isect   = set.intersection(*csp_valid.values())   if csp_valid   else set()
                sicsp_isect = set.intersection(*sicsp_valid.values()) if sicsp_valid else set()

            dataset_intersections[metric] = {
                "csp":   sorted(csp_isect),
                "sicsp": sorted(sicsp_isect),
            }

            for method in present_csp:
                _filter_and_write(method, dataset, metric, csp_isect, predid_to_formula)
            for method in present_sicsp:
                _filter_and_write(method, dataset, metric, sicsp_isect, predid_to_formula)

            print(f"  [{dataset}] {metric:<30} CSP {len(csp_isect)}/{total}  SICSP {len(sicsp_isect)}/{total}")

        if not precomputed:
            with open(precomputed_path, "w") as f:
                json.dump(dataset_intersections, f, indent=2)

    print("Intersection filtering complete.")


if __name__ == "__main__":
    run_filter()
