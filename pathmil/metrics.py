"""Evaluation metrics (ported from train_mil_loo.py)."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall MSE/R2 plus per-output Pearson correlation (PCC)."""
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    pcc_per_output = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) > 1e-8:
            pcc, _ = pearsonr(y_pred[:, j], y_true[:, j])
            pcc_per_output.append(pcc)
        else:
            pcc_per_output.append(np.nan)

    pcc_per_output = np.array(pcc_per_output)
    metrics["pcc_per_gene"] = pcc_per_output  # name kept for backwards compat
    metrics["mean_pcc"] = np.nanmean(pcc_per_output)
    metrics["median_pcc"] = np.nanmedian(pcc_per_output)
    metrics["std_pcc"] = np.nanstd(pcc_per_output)
    return metrics
