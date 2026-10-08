#!/usr/bin/env python3
"""
Research 3 — Step 4: Primary Evaluation of the Frozen R2 Predictor on R3 Dataset.

CRITICAL DISCIPLINE:
  - Zero model retraining, adaptation, recalibration, or threshold tuning.
  - Frozen model: models/expanded_mlp.pt (SHA-256 verified)
  - Frozen scaler: models/expanded_scaler.joblib (SHA-256 verified)
  - Decision threshold: locked at 0.5
  - Target: is_failure_within_next_k (K=5 frames, ~165 ms)

Computes:
  - Overall aggregate transfer metrics (AUROC, AUPRC, F1, Precision, Recall, Confusion Matrix)
  - Per-regime breakdown (Nominal, Low, Medium, High, Transition)
  - Per-flight / trial metrics (all 40 trials)
  - Direct delta comparisons against authoritative R2 test baseline

Outputs:
  - data/processed/r3_frozen_r2_evaluation_metrics.json
  - data/processed/r3_per_flight_metrics.csv
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn

R3_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = R3_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5.csv"
FROZEN_MANIFEST_PATH = R3_ROOT / "models" / "frozen_r2_manifest.json"
MODEL_PATH = R3_ROOT / "models" / "expanded_mlp.pt"
SCALER_PATH = R3_ROOT / "models" / "expanded_scaler.joblib"

METRICS_JSON_PATH = PROCESSED_DIR / "r3_frozen_r2_evaluation_metrics.json"
PER_FLIGHT_CSV_PATH = PROCESSED_DIR / "r3_per_flight_metrics.csv"


class SmallMLP(nn.Module):
    """Authoritative frozen 2-hidden-layer MLP architecture."""

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


def compute_metrics_dict(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    n_pos = int(np.sum(y_true))
    n_tot = len(y_true)
    pos_rate = float(n_pos / n_tot * 100.0) if n_tot > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, probs))
        auprc = float(average_precision_score(y_true, probs))
    else:
        auroc = float("nan")
        auprc = float("nan")

    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    warning_rate = float((tp + fp) / n_tot * 100.0) if n_tot > 0 else 0.0

    return {
        "total_samples": n_tot,
        "positive_samples": n_pos,
        "positive_rate_pct": round(pos_rate, 2),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "warning_rate_pct": round(warning_rate, 2),
    }


def main():
    print("=" * 70)
    print("Research 3 — Step 4: Primary Evaluation of Frozen R2 Predictor")
    print("=" * 70)

    # 1. Load frozen manifest and baseline metrics
    with open(FROZEN_MANIFEST_PATH, "r", encoding="utf-8") as f:
        frozen_manifest = json.load(f)

    r2_baseline = frozen_manifest["r2_test_baseline_performance"]
    feature_cols = frozen_manifest["feature_specification"]["feature_names_ordered"]
    threshold = frozen_manifest["prediction_specification"]["decision_threshold"]

    # 2. Load dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded R3 Dataset: {len(df)} total samples across {df['trial_id'].nunique()} trials.")

    # 3. Load frozen scaler and model
    scaler = joblib.load(SCALER_PATH)
    model = SmallMLP(15, 32, 16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 4. Generate predictions across the full dataset
    X_raw = df[feature_cols].values.astype(float)
    y_all = df["is_failure_within_next_k"].values.astype(int)

    X_scaled = scaler.transform(X_raw)
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        probs_all = torch.sigmoid(logits).numpy()

    df["pred_prob"] = probs_all
    df["pred_label"] = (probs_all >= threshold).astype(int)
    df.to_csv(DATASET_PATH, index=False)

    # 5. Compute Aggregate Overall Metrics
    overall_metrics = compute_metrics_dict(y_all, probs_all, threshold)

    # 6. Compute Per-Regime Metrics
    regime_metrics = {}
    for regime in ["NOMINAL", "LOW", "MEDIUM", "HIGH", "TRANSITION"]:
        r_mask = df["regime"] == regime
        if r_mask.any():
            y_r = y_all[r_mask]
            p_r = probs_all[r_mask]
            regime_metrics[regime] = compute_metrics_dict(y_r, p_r, threshold)

    # 7. Compute Per-Trial / Flight Metrics
    per_flight_rows = []
    for trial_id, group in df.groupby("trial_id", sort=False):
        y_t = group["is_failure_within_next_k"].values.astype(int)
        p_t = group["pred_prob"].values.astype(float)
        t_metrics = compute_metrics_dict(y_t, p_t, threshold)
        t_metrics["trial_id"] = trial_id
        t_metrics["base_flight"] = group["base_flight"].iloc[0]
        t_metrics["trajectory"] = group["trajectory"].iloc[0]
        t_metrics["regime"] = group["regime"].iloc[0]
        per_flight_rows.append(t_metrics)

    df_per_flight = pd.DataFrame(per_flight_rows)
    df_per_flight.to_csv(PER_FLIGHT_CSV_PATH, index=False)

    # 8. Compute Delta against R2 Baseline
    deltas = {
        "auroc_delta": round(overall_metrics["auroc"] - r2_baseline["auroc"], 4),
        "auprc_delta": round(overall_metrics["auprc"] - r2_baseline["auprc"], 4),
        "f1_delta": round(overall_metrics["f1"] - r2_baseline["f1"], 4),
        "precision_delta": round(overall_metrics["precision"] - r2_baseline["precision"], 4),
        "recall_delta": round(overall_metrics["recall"] - r2_baseline["recall"], 4),
    }

    # Regime-specific deltas
    regime_deltas = {}
    for reg, rm in regime_metrics.items():
        if not np.isnan(rm["auroc"]):
            regime_deltas[reg] = {
                "auroc_delta": round(rm["auroc"] - r2_baseline["auroc"], 4),
                "auprc_delta": round(rm["auprc"] - r2_baseline["auprc"], 4),
                "f1_delta": round(rm["f1"] - r2_baseline["f1"], 4),
                "recall_delta": round(rm["recall"] - r2_baseline["recall"], 4),
            }

    # Summary payload
    results_payload = {
        "r2_test_baseline": r2_baseline,
        "r3_overall_transfer_metrics": overall_metrics,
        "deltas_vs_r2": deltas,
        "regime_breakdown": regime_metrics,
        "regime_deltas_vs_r2": regime_deltas,
        "per_flight_summary": {
            "total_flights": len(df_per_flight),
            "mean_auroc": round(float(df_per_flight["auroc"].dropna().mean()), 4),
            "std_auroc": round(float(df_per_flight["auroc"].dropna().std()), 4),
            "min_auroc": round(float(df_per_flight["auroc"].dropna().min()), 4),
            "max_auroc": round(float(df_per_flight["auroc"].dropna().max()), 4),
            "mean_f1": round(float(df_per_flight["f1"].mean()), 4),
            "mean_recall": round(float(df_per_flight["recall"].mean()), 4),
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("=" * 70)
    print("PRIMARY EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"R2 Motion-Induced Baseline Test AUROC: {r2_baseline['auroc']:.4f} | F1: {r2_baseline['f1']:.4f} | Recall: {r2_baseline['recall']:.4f}")
    print(f"R3 Visual-Degradation Overall AUROC:   {overall_metrics['auroc']:.4f} | F1: {overall_metrics['f1']:.4f} | Recall: {overall_metrics['recall']:.4f}")
    print(f"Delta AUROC: {deltas['auroc_delta']:+.4f} | Delta F1: {deltas['f1_delta']:+.4f} | Delta Recall: {deltas['recall_delta']:+.4f}")
    print("-" * 70)
    print("Breakdown by Degradation Regime:")
    for reg, rm in regime_metrics.items():
        print(f"  {reg:12s} | PosRate: {rm['positive_rate_pct']:5.1f}% | AUROC: {rm['auroc']:.4f} | AUPRC: {rm['auprc']:.4f} | F1: {rm['f1']:.4f} | Rec: {rm['recall']:.4f}")
    print("=" * 70)
    print(f"-> Full metrics saved: {METRICS_JSON_PATH}")
    print(f"-> Per-flight metrics saved: {PER_FLIGHT_CSV_PATH}")


if __name__ == "__main__":
    main()
