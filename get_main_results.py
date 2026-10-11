import os
import json
import numpy as np

METRICS  = ['chamfer', 'sinkhorn', 'hausdorff', 'superpose_rmsd', 'density',
            'soap_dist', 'xrd_dist', 'formation_energy', 'e_above_hull',
            'relaxed_formation_energy', 'relaxed_e_above_hull']
DATASETS = ['perov_5', 'carbon_24', 'mp_20', 'mpts_52']
METHODS  = ['CDVAE', 'DiffCSP', 'MatterGen', 'FlowLLM', 'CrystalFlow', 'FlowMM',
            'OPAL-CSP', 'DiffCSP++GT', 'DiffCSP++CSPML', 'SGEquiDiff-GT', 'SGEquiDiff-CSPML']


def compute_medians():
    medians = {}
    for metric in METRICS:
        for dataset in DATASETS:
            for method in METHODS:
                path = os.path.join('results', 'metrics', f'{metric}_{dataset}_{method}.json')
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    data = json.load(f)
                key = next(k for k in data if k not in ('errors', 'pred_ids', 'annots_ids'))
                values = np.array(data[key])
                values = values[values != -1]
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    continue
                medians.setdefault(dataset, {}).setdefault(metric, {})[method] = float(np.median(values))
    return medians


def save_medians(medians):
    os.makedirs(os.path.join('results', 'main_results'), exist_ok=True)
    path = os.path.join('results', 'main_results', 'medians.json')
    with open(path, 'w') as f:
        json.dump(medians, f, indent=2)
    return path


if __name__ == '__main__':
    medians = compute_medians()
    path = save_medians(medians)
    print(f"Medians saved to {path}")
