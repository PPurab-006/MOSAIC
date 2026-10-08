#!/usr/bin/env python3
"""
Research 3 Refinement — Step 6: Targeted Ablation & Factor Attribution.

Ablates the components of the refinement:
1. Baseline: Frozen R2 (Frozen Weights, Frozen Scaler, theta = 0.50) [Model A]
2. Ablation A: Threshold Calibration Only (Frozen Weights, Frozen Scaler, theta = 0.46) [Model C]
3. Ablation B: Refined Weights Only (Refined Weights, Refined Scaler, fixed theta = 0.50)
4. Full Refinement: Refined Weights + Unified Scaler + Validation Threshold theta = 0.37 [Model B]

Outputs:
  - r3_refinement/results/metrics/ablation_results.json
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn

R3_ROOT = Path(__file__).resolve().parent.parent.parent
REFINEMENT_DIR = R3_ROOT / "r3_refinement"
METRICS_DIR = REFINEMENT_DIR / "results" / "metrics"

MANIFEST_PATH = REFINEMENT_DIR / "manifests" / "refinement_split_manifest.json"
LOCKED_THRESHOLDS_PATH = REFINEMENT_DIR / "models" / "locked_thresholds.json"
DATASET_R2 = Path("/home/purab/Purab/Projects/Research2/data/processed/expanded_frames_v2_k5.csv")
DATASET_R3 = R3_ROOT / "data" / "processed" / "r3_dataset_k5.csv"

OUT_ABLATION_JSON = METRICS_DIR / "ablation_results.json"


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int = 15, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    auroc = float(roc_auc_score(y_true, probs))
    auprc = float(average_precision_score(y_true, probs))
    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def main():
    print("=" * 70)
    print("RESEARCH 3 REFINEMENT: TARGETED ABLATION & ATTRIBUTION")
    print("=" * 70)

    with open(MANIFEST_PATH, "r") as f:
        split_manifest = json.load(f)
    with open(LOCKED_THRESHOLDS_PATH, "r") as f:
        locked_th = json.load(f)

    with open(R3_ROOT / "models" / "frozen_r2_manifest.json") as f:
        feat_cols = json.load(f)["feature_specification"]["feature_names_ordered"]

    # Load models & scalers
    scaler_r2 = joblib.load(locked_th["model_a_frozen_r2"]["scaler_path"])
    model_r2 = SmallMLP(15, 32, 16)
    model_r2.load_state_dict(torch.load(locked_th["model_a_frozen_r2"]["model_path"], map_location="cpu"))
    model_r2.eval()

    scaler_ref = joblib.load(locked_th["model_b_refined_mlp"]["scaler_path"])
    model_ref = SmallMLP(15, 32, 16)
    model_ref.load_state_dict(torch.load(locked_th["model_b_refined_mlp"]["model_path"], map_location="cpu"))
    model_ref.eval()

    # Load protected test sets
    r2_test_flights = split_manifest["r2_motion_mechanism"]["protected_test_flights"]
    df_r2_all = pd.read_csv(DATASET_R2)
    df_r2_test = df_r2_all[df_r2_all["run_dir"].isin(r2_test_flights)].copy().reset_index(drop=True)

    r3_test_trials = split_manifest["r3_visual_mechanism"]["protected_test_trials"]
    df_r3_all = pd.read_csv(DATASET_R3)
    df_r3_test = df_r3_all[df_r3_all["trial_id"].isin(r3_test_trials)].copy().reset_index(drop=True)

    # Predictions
    # Frozen model (R2 scaler + R2 weights)
    X_r2_test_r2sc = scaler_r2.transform(df_r2_test[feat_cols].values.astype(float))
    X_r3_test_r2sc = scaler_r2.transform(df_r3_test[feat_cols].values.astype(float))
    with torch.no_grad():
        p_r2_test_frozen = torch.sigmoid(model_r2(torch.tensor(X_r2_test_r2sc, dtype=torch.float32))).numpy()
        p_r3_test_frozen = torch.sigmoid(model_r2(torch.tensor(X_r3_test_r2sc, dtype=torch.float32))).numpy()

    # Refined model (Refined scaler + Refined weights)
    X_r2_test_refsc = scaler_ref.transform(df_r2_test[feat_cols].values.astype(float))
    X_r3_test_refsc = scaler_ref.transform(df_r3_test[feat_cols].values.astype(float))
    with torch.no_grad():
        p_r2_test_ref = torch.sigmoid(model_ref(torch.tensor(X_r2_test_refsc, dtype=torch.float32))).numpy()
        p_r3_test_ref = torch.sigmoid(model_ref(torch.tensor(X_r3_test_refsc, dtype=torch.float32))).numpy()

    y_r2 = df_r2_test["is_failure_within_next_k"].values.astype(int)
    y_r3 = df_r3_test["is_failure_within_next_k"].values.astype(int)

    # Configurations to evaluate:
    # 1. Model A: Frozen R2 (Weights: R2, Scaler: R2, Theta: 0.50)
    # 2. Ablation A: Threshold Only (Weights: R2, Scaler: R2, Theta: 0.46)
    # 3. Ablation B: Refined Rep Only (Weights: Ref, Scaler: Ref, Theta: 0.50)
    # 4. Model B: Full Refinement (Weights: Ref, Scaler: Ref, Theta: 0.37)

    configs = {
        "1_frozen_r2_baseline": {
            "desc": "Frozen R2 (Frozen Weights, Frozen Scaler, theta=0.50)",
            "visual_metrics": compute_metrics(y_r3, p_r3_test_frozen, 0.50),
            "motion_metrics": compute_metrics(y_r2, p_r2_test_frozen, 0.50),
        },
        "2_threshold_calibration_only": {
            "desc": "Calibration Only (Frozen Weights, Frozen Scaler, theta=0.46)",
            "visual_metrics": compute_metrics(y_r3, p_r3_test_frozen, 0.46),
            "motion_metrics": compute_metrics(y_r2, p_r2_test_frozen, 0.46),
        },
        "3_refined_representation_fixed_threshold": {
            "desc": "Refined Rep Only (Refined Weights, Refined Scaler, fixed theta=0.50)",
            "visual_metrics": compute_metrics(y_r3, p_r3_test_ref, 0.50),
            "motion_metrics": compute_metrics(y_r2, p_r2_test_ref, 0.50),
        },
        "4_full_cross_mechanism_refinement": {
            "desc": "Full Refinement (Refined Weights, Refined Scaler, val theta=0.37)",
            "visual_metrics": compute_metrics(y_r3, p_r3_test_ref, 0.37),
            "motion_metrics": compute_metrics(y_r2, p_r2_test_ref, 0.37),
        }
    }

    with open(OUT_ABLATION_JSON, "w") as f:
        json.dump(configs, f, indent=2)

    print("\nABLATION & ATTRIBUTION MATRIX:")
    print("-" * 80)
    print(f"{'Configuration':<35} | {'Visual AUROC':<12} | {'Visual Rec':<10} | {'Visual F1':<10} | {'Motion Rec':<10} | {'Motion F1':<10}")
    print("-" * 80)
    for k, v in configs.items():
        vm = v["visual_metrics"]
        mm = v["motion_metrics"]
        print(f"{v['desc']:<35} | {vm['auroc']:<12.4f} | {vm['recall']*100:<9.1f}% | {vm['f1']:<10.4f} | {mm['recall']*100:<9.1f}% | {mm['f1']:<10.4f}")
    print("=" * 80)
    print(f"-> Ablation results saved to: {OUT_ABLATION_JSON}")


if __name__ == "__main__":
    main()
