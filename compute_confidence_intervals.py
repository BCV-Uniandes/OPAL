import json
import os
import numpy as np

METRICS_DIR = "results/metrics"
OUTPUT_PATH = "results/main_results/confidence_intervals.json"

DATASETS = ["perov_5", "carbon_24", "mp_20", "mpts_52"]
METRICS  = ["chamfer", "sinkhorn", "hausdorff", "superpose_rmsd", "density",
            "soap_dist", "xrd_dist", "formation_energy", "e_above_hull",
            "relaxed_formation_energy", "relaxed_e_above_hull"]
METHODS  = ["CDVAE", "DiffCSP", "MatterGen", "FlowLLM", "CrystalFlow",
            "FlowMM", "OPAL-CSP", "DiffCSP++GT", "DiffCSP++CSPML",
            "SGEquiDiff-GT", "SGEquiDiff-CSPML"]

N_BOOTSTRAP = 10_000
CI_LEVEL    = 0.95
ALPHA       = 1 - CI_LEVEL
RNG         = np.random.default_rng(42)


def bootstrap_median_ci(values):
    medians = np.array([
        np.median(RNG.choice(values, size=len(values), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    lo = np.percentile(medians, 100 * ALPHA / 2)
    hi = np.percentile(medians, 100 * (1 - ALPHA / 2))
    return lo, hi


def load_values(metric, dataset, model):
    fpath = os.path.join(METRICS_DIR, f"{metric}_{dataset}_{model}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        d = json.load(f)
    key = list(d.keys())[0]
    values = np.array([v for v in d[key] if v is not None and not np.isnan(v)])
    values = values[values != -1]
    return values if len(values) > 0 else None


def compute_confidence_intervals():
    results = {}
    total = len(DATASETS) * len(METRICS) * len(METHODS)
    done  = 0

    for dataset in DATASETS:
        results[dataset] = {}
        for metric in METRICS:
            results[dataset][metric] = {}
            for model in METHODS:
                done += 1
                print(f"[{done}/{total}] {dataset} · {metric} · {model}", end="\r")

                values = load_values(metric, dataset, model)
                if values is None:
                    results[dataset][metric][model] = None
                    continue

                median = float(np.median(values))
                ci_lo, ci_hi = bootstrap_median_ci(values)

                results[dataset][metric][model] = {
                    "median":   round(median, 6),
                    "ci_lo":    round(float(ci_lo), 6),
                    "ci_hi":    round(float(ci_hi), 6),
                    "ci_level": CI_LEVEL,
                    "n":        len(values),
                }

    print()
    return results


def save_confidence_intervals(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return OUTPUT_PATH


if __name__ == "__main__":
    results = compute_confidence_intervals()
    path = save_confidence_intervals(results)
    print(f"Saved to {path}")
