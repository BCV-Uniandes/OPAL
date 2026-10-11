import os
import sys
import json
import subprocess

OSF_URL = "https://osf.io/9jhm4/"

DATA_DIRS = [
    'gt_files', 'pred_files', 'poly_files',
    'results/alignment', 'results/intersections', 'results/gt_energies',
]

def check_data():
    missing = [d for d in DATA_DIRS if not any(f for f in os.listdir(d) if f != '.gitkeep')]
    if missing:
        print("Error: missing data in:", ', '.join(missing))
        print(f"Download the data from {OSF_URL}")
        print("See the README for download instructions.")
        sys.exit(1)

DATASETS = ['perov_5', 'carbon_24', 'mp_20', 'mpts_52']

CSP_METHODS   = ['CDVAE', 'CrystalFlow', 'DiffCSP', 'FlowLLM', 'FlowMM', 'MatterGen', 'OPAL-CSP']
SICSP_METHODS = ['DiffCSP++GT', 'DiffCSP++CSPML', 'SGEquiDiff-GT', 'SGEquiDiff-CSPML']
ALL_METHODS   = CSP_METHODS + SICSP_METHODS

METRICS = ['chamfer', 'sinkhorn', 'hausdorff', 'superpose_rmsd',
           'density', 'soap_dist', 'xrd_dist', 'energies']

LCM_METHODS       = {'SGEquiDiff-GT', 'SGEquiDiff-CSPML', 'DiffCSP++GT', 'DiffCSP++CSPML'}
ALIGNMENT_METRICS = {'chamfer', 'sinkhorn', 'hausdorff', 'superpose_rmsd'}
MAX_LCM = {
    'carbon_24': 264,
    'perov_5':     5,
    'mp_20':    1360,
    'mpts_52':  1360,
}

NUM_WORKERS = 5
BATCH_SIZE  = 128


def run(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nCommand failed: {cmd}")
        sys.exit(1)


def step1_filter_invalid():
    print("\n" + "="*60)
    print("Step 1: Filtering invalid structures")
    print("="*60)
    for dataset in DATASETS:
        for method in ALL_METHODS:
            run(f"python filter_invalid_structures.py "
                f"--dataset {dataset} --method {method} --pred_only --move")


def step2_compute_metrics():
    print("\n" + "="*60)
    print("Step 2: Computing metrics")
    print("="*60)
    for dataset in DATASETS:
        for method in ALL_METHODS:
            for metric in METRICS:
                lcm_flag = ""
                if method in LCM_METHODS and metric in ALIGNMENT_METRICS:
                    lcm_flag = f"--use_lcm --max_lcm {MAX_LCM.get(dataset, 48)}"
                run(f"python get_csp_metrics.py "
                    f"--dataset_name {dataset} --method_name {method} "
                    f"--metric {metric} --pred_path pred_files --gt_path gt_files "
                    f"--num_workers {NUM_WORKERS} --batch_size {BATCH_SIZE} {lcm_flag}")


def step3_filter_intersection():
    print("\n" + "="*60)
    print("Step 3: Filtering by group intersection")
    print("="*60)
    run("python filter_intersection.py")


def step4_compute_medians():
    print("\n" + "="*60)
    print("Step 4: Computing medians")
    print("="*60)
    run("python get_main_results.py")


def step5_confidence_intervals():
    print("\n" + "="*60)
    print("Step 5: Computing bootstrap confidence intervals")
    print("="*60)
    run("python compute_confidence_intervals.py")


DISPLAY_NAMES = {
    'superpose_rmsd': 'rmse',
}


def print_results():
    medians_path = os.path.join("results", "main_results", "medians.json")
    ci_path      = os.path.join("results", "main_results", "confidence_intervals.json")

    with open(medians_path) as f:
        medians = json.load(f)
    with open(ci_path) as f:
        cis = json.load(f)

    print("\n" + "="*60)
    print("RESULTS — Median  [95% CI]")
    print("="*60)

    for dataset in DATASETS:
        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset}")
        print(f"{'─'*60}")
        for metric in medians.get(dataset, {}):
            display = DISPLAY_NAMES.get(metric, metric)
            print(f"\n  {display}")
            for method, value in sorted(medians[dataset][metric].items()):
                ci = cis.get(dataset, {}).get(metric, {}).get(method)
                if ci:
                    print(f"    {method:<24} {value:.6f}  [{ci['ci_lo']:.6f}, {ci['ci_hi']:.6f}]  n={ci['n']}")
                else:
                    print(f"    {method:<24} {value:.6f}")


if __name__ == "__main__":
    check_data()
    step1_filter_invalid()
    step2_compute_metrics()
    step3_filter_intersection()
    step4_compute_medians()
    step5_confidence_intervals()
    print_results()

    print("\n" + "="*60)
    print("Done. Results saved to results/main_results/")
    print("="*60)
