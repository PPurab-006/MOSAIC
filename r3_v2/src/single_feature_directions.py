#!/usr/bin/env python3
"""
Research 3 (MOSAIC) — V2 Single-Feature Directionality Analysis.

Evaluates the univariate discriminative capacity (AUROC) and directionality (+ / -)
of:
  1. Optical flow velocity magnitude (feature_vel_mean_lag0)
  2. Platform yaw rate (eis_yaw_rate_deg_lag0)
  3. Mean gradient magnitude (mean_grad_mag_lag0)
  4. Laplacian variance (laplacian_var_lag0)
  5. Pixel standard deviation (pixel_std_lag0)
  6. Shi-Tomasi corners found (shi_tomasi_corners_lag0)

Across:
  - R2 Motion Protected Test Partition (14 flights)
  - R3 Visual Degradation Protected Test Partition (15 trials)
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R3_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = R3_ROOT / "r3_v2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FEATURES = [
    ("flow_magnitude", "feature_vel_mean_lag0"),
    ("yaw_rate", "eis_yaw_rate_deg_lag0"),
    ("mean_gradient_magnitude", "mean_grad_mag_lag0"),
    ("laplacian_variance", "laplacian_var_lag0"),
    ("pixel_std", "pixel_std_lag0"),
    ("shi_tomasi_corners", "shi_tomasi_corners_lag0"),
]


def compute_feature_auroc_and_direction(y_true: np.ndarray, x: np.ndarray) -> Dict:
    """Computes raw AUROC, correlation direction, and effective inverted AUROC."""
    if len(np.unique(y_true)) < 2:
        return {"raw_auroc": float("nan"), "direction": "N/A", "effective_auroc": float("nan")}

    raw_auc = float(roc_auc_score(y_true, x))
    direction = "+" if raw_auc >= 0.50 else "-"
    effective_auc = raw_auc if raw_auc >= 0.50 else 1.0 - raw_auc

    return {
        "raw_auroc": round(raw_auc, 4),
        "direction": direction,
        "effective_auroc": round(effective_auc, 4),
    }


def analyze_single_feature_directions(df_test: pd.DataFrame) -> pd.DataFrame:
    """Computes univariate directionality table comparing R2 motion vs R3 visual test sets."""
    df_r2 = df_test[df_test["domain"] == "R2_MOTION"].copy()
    df_r3 = df_test[df_test["domain"] == "R3_VISUAL"].copy()

    y_r2 = df_r2["is_failure_within_next_k"].values
    y_r3 = df_r3["is_failure_within_next_k"].values

    rows = []
    for feat_label, feat_col in TARGET_FEATURES:
        r2_stats = compute_feature_auroc_and_direction(y_r2, df_r2[feat_col].values)
        r3_stats = compute_feature_auroc_and_direction(y_r3, df_r3[feat_col].values)

        concordant = (
            "CONCORDANT"
            if r2_stats["direction"] == r3_stats["direction"]
            else "DISCORDANT (INVERTED)"
        )

        rows.append({
            "feature_name": feat_label,
            "raw_column": feat_col,
            "r2_motion_test_auroc": r2_stats["raw_auroc"],
            "r2_motion_direction": r2_stats["direction"],
            "r2_motion_effective_auroc": r2_stats["effective_auroc"],
            "r3_visual_test_auroc": r3_stats["raw_auroc"],
            "r3_visual_direction": r3_stats["direction"],
            "r3_visual_effective_auroc": r3_stats["effective_auroc"],
            "cross_domain_alignment": concordant,
        })

    out_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "single_feature_directions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"-> Single-feature direction table saved: {out_csv}")
    return out_df
