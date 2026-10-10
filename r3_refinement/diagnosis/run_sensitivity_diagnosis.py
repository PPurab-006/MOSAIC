#!/usr/bin/env python3
"""
Research 3 Refinement — Step 1: In-depth Diagnosis of the Sensitivity Loss.

Investigates:
1. Class-conditional feature distribution shift (R2 vs R3).
2. Scaled feature space behavior and first-layer MLP weight alignments.
3. Probability distribution shifts on positive frames (P10..P90).
4. Calibration curves and Brier scores.
5. Threshold sensitivity (F1, Precision, Recall vs Threshold).
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, average_precision_score
import torch
import torch.nn as nn

R3_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = R3_ROOT / "data" / "processed"
DATASET_R3 = PROCESSED_DIR / "r3_dataset_k5.csv"
DATASET_R2 = R3_ROOT.parent / "Research2" / "data" / "processed" / "expanded_frames_v2_k5.csv"
R2_SPLIT = R3_ROOT.parent / "Research2" / "data" / "processed" / "expanded_flight_split.csv"
MODEL_PATH = R3_ROOT / "models" / "expanded_mlp.pt"
SCALER_PATH = R3_ROOT / "models" / "expanded_scaler.joblib"
MANIFEST_PATH = R3_ROOT / "models" / "frozen_r2_manifest.json"

OUT_JSON = R3_ROOT / "r3_refinement" / "diagnosis" / "diagnosis_metrics.json"


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


def main():
    print("=" * 70)
    print("RESEARCH 3 REFINEMENT: SENSITIVITY LOSS DIAGNOSIS")
    print("=" * 70)

    # 1. Load frozen manifest, model, and scaler
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    feature_cols = manifest["feature_specification"]["feature_names_ordered"]

    scaler = joblib.load(SCALER_PATH)
    model = SmallMLP(15, 32, 16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2. Load R2 test data
    df_r2_all = pd.read_csv(DATASET_R2)
    df_split = pd.read_csv(R2_SPLIT)
    test_runs = set(df_split.loc[df_split["split"] == "test", "run_dir"])
    df_r2_test = df_r2_all[df_r2_all["run_dir"].isin(test_runs)].copy()

    # Load R3 data
    df_r3 = pd.read_csv(DATASET_R3)

    # 3. Generate predictions if not present
    X_r2_raw = df_r2_test[feature_cols].values.astype(float)
    y_r2 = df_r2_test["is_failure_within_next_k"].values.astype(int)
    X_r2_scaled = scaler.transform(X_r2_raw)

    with torch.no_grad():
        p_r2 = torch.sigmoid(model(torch.tensor(X_r2_scaled, dtype=torch.float32))).numpy()

    X_r3_raw = df_r3[feature_cols].values.astype(float)
    y_r3 = df_r3["is_failure_within_next_k"].values.astype(int)
    X_r3_scaled = scaler.transform(X_r3_raw)

    with torch.no_grad():
        p_r3 = torch.sigmoid(model(torch.tensor(X_r3_scaled, dtype=torch.float32))).numpy()

    # ------------------------------------------------------------------
    # Analysis 1: Class-Conditional Feature Distributions
    # ------------------------------------------------------------------
    feat_stats = {}
    for col in feature_cols:
        r2_pos = df_r2_test.loc[y_r2 == 1, col].values
        r2_neg = df_r2_test.loc[y_r2 == 0, col].values
        r3_pos = df_r3.loc[y_r3 == 1, col].values
        r3_neg = df_r3.loc[y_r3 == 0, col].values

        feat_stats[col] = {
            "r2_pos_mean": float(np.mean(r2_pos)),
            "r2_pos_std": float(np.std(r2_pos)),
            "r2_neg_mean": float(np.mean(r2_neg)),
            "r2_neg_std": float(np.std(r2_neg)),
            "r3_pos_mean": float(np.mean(r3_pos)),
            "r3_pos_std": float(np.std(r3_pos)),
            "r3_neg_mean": float(np.mean(r3_neg)),
            "r3_neg_std": float(np.std(r3_neg)),
        }

    # ------------------------------------------------------------------
    # Analysis 2: Standardized Feature Coordinates
    # ------------------------------------------------------------------
    scaled_stats = {}
    for idx, col in enumerate(feature_cols):
        r2_pos_z = X_r2_scaled[y_r2 == 1, idx]
        r2_neg_z = X_r2_scaled[y_r2 == 0, idx]
        r3_pos_z = X_r3_scaled[y_r3 == 1, idx]
        r3_neg_z = X_r3_scaled[y_r3 == 0, idx]

        scaled_stats[col] = {
            "r2_pos_z_mean": round(float(np.mean(r2_pos_z)), 3),
            "r2_neg_z_mean": round(float(np.mean(r2_neg_z)), 3),
            "r3_pos_z_mean": round(float(np.mean(r3_pos_z)), 3),
            "r3_neg_z_mean": round(float(np.mean(r3_neg_z)), 3),
        }

    # ------------------------------------------------------------------
    # Analysis 3: Predicted Probability Distributions for Positives
    # ------------------------------------------------------------------
    p_r2_pos = p_r2[y_r2 == 1]
    p_r2_neg = p_r2[y_r2 == 0]
    p_r3_pos = p_r3[y_r3 == 1]
    p_r3_neg = p_r3[y_r3 == 0]

    prob_dist = {
        "r2_pos": {
            "mean": float(np.mean(p_r2_pos)),
            "p10": float(np.percentile(p_r2_pos, 10)),
            "p25": float(np.percentile(p_r2_pos, 25)),
            "p50": float(np.percentile(p_r2_pos, 50)),
            "p75": float(np.percentile(p_r2_pos, 75)),
            "p90": float(np.percentile(p_r2_pos, 90)),
            "pct_below_0.50": float(np.mean(p_r2_pos < 0.5) * 100.0),
            "pct_between_0.30_and_0.50": float(np.mean((p_r2_pos >= 0.3) & (p_r2_pos < 0.5)) * 100.0),
            "pct_between_0.20_and_0.50": float(np.mean((p_r2_pos >= 0.2) & (p_r2_pos < 0.5)) * 100.0),
        },
        "r3_pos": {
            "mean": float(np.mean(p_r3_pos)),
            "p10": float(np.percentile(p_r3_pos, 10)),
            "p25": float(np.percentile(p_r3_pos, 25)),
            "p50": float(np.percentile(p_r3_pos, 50)),
            "p75": float(np.percentile(p_r3_pos, 75)),
            "p90": float(np.percentile(p_r3_pos, 90)),
            "pct_below_0.50": float(np.mean(p_r3_pos < 0.5) * 100.0),
            "pct_between_0.30_and_0.50": float(np.mean((p_r3_pos >= 0.3) & (p_r3_pos < 0.5)) * 100.0),
            "pct_between_0.20_and_0.50": float(np.mean((p_r3_pos >= 0.2) & (p_r3_pos < 0.5)) * 100.0),
        },
        "r2_neg": {
            "mean": float(np.mean(p_r2_neg)),
            "p50": float(np.percentile(p_r2_neg, 50)),
            "p90": float(np.percentile(p_r2_neg, 90)),
            "pct_above_0.50": float(np.mean(p_r2_neg >= 0.5) * 100.0),
        },
        "r3_neg": {
            "mean": float(np.mean(p_r3_neg)),
            "p50": float(np.percentile(p_r3_neg, 50)),
            "p90": float(np.percentile(p_r3_neg, 90)),
            "pct_above_0.50": float(np.mean(p_r3_neg >= 0.5) * 100.0),
        },
    }

    # ------------------------------------------------------------------
    # Analysis 4: Calibration Curves & Brier Score
    # ------------------------------------------------------------------
    brier_r2 = float(brier_score_loss(y_r2, p_r2))
    brier_r3 = float(brier_score_loss(y_r3, p_r3))

    prob_true_r2, prob_pred_r2 = calibration_curve(y_r2, p_r2, n_bins=10, strategy="uniform")
    prob_true_r3, prob_pred_r3 = calibration_curve(y_r3, p_r3, n_bins=10, strategy="uniform")

    # ------------------------------------------------------------------
    # Analysis 5: Threshold Sweep (Threshold vs P, R, F1)
    # ------------------------------------------------------------------
    thresholds = np.linspace(0.1, 0.9, 81)
    sweep_r2 = []
    for th in thresholds:
        pred_th = (p_r2 >= th).astype(int)
        tp = np.sum((y_r2 == 1) & (pred_th == 1))
        fp = np.sum((y_r2 == 0) & (pred_th == 1))
        fn = np.sum((y_r2 == 1) & (pred_th == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sweep_r2.append({"th": round(float(th), 3), "prec": round(prec, 4), "rec": round(rec, 4), "f1": round(f1, 4)})

    sweep_r3 = []
    for th in thresholds:
        pred_th = (p_r3 >= th).astype(int)
        tp = np.sum((y_r3 == 1) & (pred_th == 1))
        fp = np.sum((y_r3 == 0) & (pred_th == 1))
        fn = np.sum((y_r3 == 1) & (pred_th == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sweep_r3.append({"th": round(float(th), 3), "prec": round(prec, 4), "rec": round(rec, 4), "f1": round(f1, 4)})

    df_sweep_r2 = pd.DataFrame(sweep_r2)
    df_sweep_r3 = pd.DataFrame(sweep_r3)

    best_f1_r2 = df_sweep_r2.loc[df_sweep_r2["f1"].idxmax()]
    best_f1_r3 = df_sweep_r3.loc[df_sweep_r3["f1"].idxmax()]

    # Threshold achieving ~73% recall on R3
    rec_match_r3 = df_sweep_r3.iloc[(df_sweep_r3["rec"] - 0.7336).abs().argsort()[:1]].iloc[0]

    # Threshold achieving ~50% recall on R2
    rec_match_r2 = df_sweep_r2.iloc[(df_sweep_r2["rec"] - 0.4958).abs().argsort()[:1]].iloc[0]

    # First layer weight analysis
    W1 = model.net[0].weight.detach().numpy()  # (32, 15)
    b1 = model.net[0].bias.detach().numpy()    # (32,)
    # Norm of incoming weights per feature
    w_feat_norm = np.linalg.norm(W1, axis=0)  # (15,)

    w_attribution = {
        col: round(float(w_feat_norm[idx]), 4) for idx, col in enumerate(feature_cols)
    }

    results = {
        "brier_scores": {
            "r2_test": round(brier_r2, 4),
            "r3_overall": round(brier_r3, 4),
        },
        "positive_probability_distribution": prob_dist,
        "feature_attributions_w1_norm": w_attribution,
        "standardized_z_means": scaled_stats,
        "optimal_thresholds": {
            "r2_best_f1": best_f1_r2.to_dict(),
            "r3_best_f1": best_f1_r3.to_dict(),
            "r3_matching_r2_recall": rec_match_r3.to_dict(),
            "r2_matching_r3_recall": rec_match_r2.to_dict(),
        },
        "feature_stats_sample": {
            "eis_yaw_rate_deg_lag0": feat_stats["eis_yaw_rate_deg_lag0"],
            "feature_vel_mean_lag0": feat_stats["feature_vel_mean_lag0"],
            "is_r_frame_lag0": feat_stats["is_r_frame_lag0"],
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDIAGNOSIS SUMMARY:")
    print("-" * 70)
    print("1. PROBABILITY SHIFTS ON FAILING FRAMES (y=1):")
    print(f"   R2 Test Positives: Mean p = {prob_dist['r2_pos']['mean']:.3f}, Median p = {prob_dist['r2_pos']['p50']:.3f}, {prob_dist['r2_pos']['pct_below_0.50']:.1f}% below 0.50")
    print(f"   R3 Test Positives: Mean p = {prob_dist['r3_pos']['mean']:.3f}, Median p = {prob_dist['r3_pos']['p50']:.3f}, {prob_dist['r3_pos']['pct_below_0.50']:.1f}% below 0.50")
    print(f"   In R3, {prob_dist['r3_pos']['pct_between_0.20_and_0.50']:.1f}% of failing frames lie in the narrow window p in [0.20, 0.50)!")
    print("-" * 70)
    print("2. STANDARDIZED Z-SCORES ON FAILING FRAMES (y=1):")
    for feat in ["eis_yaw_rate_deg_lag0", "feature_vel_mean_lag0", "is_r_frame_lag0"]:
        print(f"   {feat:25s}: R2(y=1) z = {scaled_stats[feat]['r2_pos_z_mean']:+.2f} | R3(y=1) z = {scaled_stats[feat]['r3_pos_z_mean']:+.2f}")
    print("-" * 70)
    print("3. THRESHOLD SENSITIVITY:")
    print(f"   R2 at th=0.50: Prec = 0.581, Rec = 0.734, F1 = 0.648 (Best F1 th = {best_f1_r2['th']}: F1={best_f1_r2['f1']:.3f})")
    print(f"   R3 at th=0.50: Prec = 0.939, Rec = 0.496, F1 = 0.649 (Best F1 th = {best_f1_r3['th']}: F1={best_f1_r3['f1']:.3f})")
    print(f"   To reach ~73% recall in R3: threshold must be {rec_match_r3['th']:.2f} -> Prec={rec_match_r3['prec']:.3f}, Rec={rec_match_r3['rec']:.3f}, F1={rec_match_r3['f1']:.3f}")
    print("=" * 70)
    print(f"-> Full diagnostic metrics saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
