import numpy as np


def energy_dist(res_gt: dict, res_pred: dict) -> dict:
    mlips = ["orb", "mace", "uma"]
    metrics = ["formation_energy", "e_above_hull",
               "relaxed_formation_energy", "relaxed_e_above_hull"]
    ensemble_metrics = [
        "formation_energy_mean", "formation_energy_std",
        "e_above_hull_mean", "e_above_hull_std",
        "relaxed_formation_energy_mean", "relaxed_formation_energy_std",
        "relaxed_e_above_hull_mean", "relaxed_e_above_hull_std",
    ]

    distances = {}
    for mlip in mlips:
        distances[mlip] = {}
        for metric in metrics:
            gt_val, pred_val = res_gt[mlip][metric], res_pred[mlip][metric]
            distances[mlip][metric] = float(abs(gt_val - pred_val)) if gt_val is not None and pred_val is not None else None

    distances["ensemble"] = {}
    for metric in ensemble_metrics:
        gt_val, pred_val = res_gt["ensemble"][metric], res_pred["ensemble"][metric]
        distances["ensemble"][metric] = float(abs(gt_val - pred_val)) if gt_val is not None and pred_val is not None else None

    return distances
