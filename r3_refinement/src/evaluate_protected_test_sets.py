#!/usr/bin/env python3
"""
Research 3 Refinement — Step 4: Side-by-Side Protected Test Evaluation.

Evaluates Model A (Frozen R2 Baseline), Model B (Cross-Mechanism Refined MLP),
and Model C (Validation-Calibrated Frozen R2) on:
  1. Protected R2 Motion-Induced Test Set (14 held-out test flights).
  2. Protected R3 Visual-Degradation Test Set (15 held-out trials across 5 regimes).
  3. Combined Cross-Mechanism Protected Test Set.
  4. Complete 40-Flight R3 Primary Dataset (for direct comparison against R3 Phase 1 report).

Outputs:
  - r3_refinement/results/metrics/protected_evaluation_metrics.json
  - r3_refinement/results/metrics/per_flight_comparison.csv
  - r3_refinement/results/metrics/episode_early_warning_comparison.json
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

R3_ROOT = Path(__file__).resolve().parent.parent.parent
REFINEMENT_DIR = R3_ROOT / "r3_refinement"
METRICS_DIR = REFINEMENT_DIR / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = REFINEMENT_DIR / "manifests" / "refinement_split_manifest.json"
LOCKED_THRESHOLDS_PATH = REFINEMENT_DIR / "models" / "locked_thresholds.json"

DATASET_R2 = R3_ROOT.parent / "Research2" / "data" / "processed" / "expanded_frames_v2_k5.csv"
DATASET_R3 = R3_ROOT / "data" / "processed" / "r3_dataset_k5.csv"

OUT_METRICS_JSON = METRICS_DIR / "protected_evaluation_metrics.json"
OUT_PER_FLIGHT_CSV = METRICS_DIR / "per_flight_comparison.csv"
OUT_EPISODES_JSON = METRICS_DIR / "episode_early_warning_comparison.json"


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


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
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
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    warn_rate = float((tp + fp) / n_tot * 100.0) if n_tot > 0 else 0.0

    return {
        "samples": n_tot,
        "positives": n_pos,
        "positive_rate_pct": round(pos_rate, 2),
        "threshold": round(threshold, 3),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fpr": round(fpr, 4),
        "warning_rate_pct": round(warn_rate, 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def analyze_episodes(df_trials: pd.DataFrame, probs: np.ndarray, threshold: float, max_horizon_s: float = 0.165) -> dict:
    df_eval = df_trials.copy()
    df_eval["prob"] = probs
    df_eval["warn"] = (probs >= threshold).astype(int)

    all_episodes = []
    # Identify flight identifier column
    flight_col = "run_dir" if "run_dir" in df_eval.columns else "trial_id"

    for f_id, group in df_eval.groupby(flight_col, sort=False):
        if "timestamp_total_sec" in group.columns:
            t = group["timestamp_total_sec"].values.astype(float)
        else:
            t = np.arange(len(group)) * 0.0333
        # Failure column
        y_curr = group["is_failure_current_frame"].values.astype(int) if "is_failure_current_frame" in group.columns else (group["num_inliers_pose"] < 8).astype(int).values
        warn = group["warn"].values.astype(int)

        n = len(group)
        in_ep = False
        start_idx = 0
        episodes = []
        for i in range(n):
            if y_curr[i] == 1 and not in_ep:
                in_ep = True
                start_idx = i
            elif y_curr[i] == 0 and in_ep:
                in_ep = False
                episodes.append((start_idx, i - 1))
        if in_ep:
            episodes.append((start_idx, n - 1))

        for ep_idx, (s_idx, e_idx) in enumerate(episodes):
            t_onset = t[s_idx]
            t_end = t[e_idx]
            pre_mask = (t >= (t_onset - max_horizon_s - 1e-4)) & (t < t_onset)
            pre_warns = np.where(pre_mask & (warn == 1))[0]
            intra_mask = (t >= t_onset) & (t <= t_end)
            intra_warns = np.where(intra_mask & (warn == 1))[0]

            if len(pre_warns) > 0:
                t_w = t[pre_warns[0]]
                all_episodes.append({"status": "EARLY", "lead_s": t_onset - t_w})
            elif len(intra_warns) > 0:
                t_w = t[intra_warns[0]]
                all_episodes.append({"status": "POST", "lead_s": -(t_w - t_onset)})
            else:
                all_episodes.append({"status": "MISSED", "lead_s": np.nan})

    total_ep = len(all_episodes)
    if total_ep == 0:
        return {"total_episodes": 0, "early_rate_pct": 0.0, "missed_rate_pct": 0.0, "median_lead_s": 0.0}

    early_ep = [e for e in all_episodes if e["status"] == "EARLY"]
    post_ep = [e for e in all_episodes if e["status"] == "POST"]
    missed_ep = [e for e in all_episodes if e["status"] == "MISSED"]

    early_rate = len(early_ep) / total_ep * 100.0
    post_rate = len(post_ep) / total_ep * 100.0
    missed_rate = len(missed_ep) / total_ep * 100.0

    leads = [e["lead_s"] for e in early_ep]
    med_lead = float(np.median(leads)) if len(leads) > 0 else 0.0
    mean_lead = float(np.mean(leads)) if len(leads) > 0 else 0.0

    return {
        "total_episodes": total_ep,
        "early_warnings": len(early_ep),
        "early_rate_pct": round(early_rate, 2),
        "post_detections": len(post_ep),
        "post_rate_pct": round(post_rate, 2),
        "missed_failures": len(missed_ep),
        "missed_rate_pct": round(missed_rate, 2),
        "median_lead_s": round(med_lead, 4),
        "mean_lead_s": round(mean_lead, 4),
    }


def main():
    print("=" * 70)
    print("RESEARCH 3 REFINEMENT: PROTECTED TEST EVALUATION")
    print("=" * 70)

    # 1. Load manifest and locked thresholds
    with open(MANIFEST_PATH, "r") as f:
        split_manifest = json.load(f)
    with open(LOCKED_THRESHOLDS_PATH, "r") as f:
        locked_th = json.load(f)

    with open(R3_ROOT / "models" / "frozen_r2_manifest.json") as f:
        feat_cols = json.load(f)["feature_specification"]["feature_names_ordered"]

    th_a = locked_th["model_a_frozen_r2"]["threshold"]           # 0.50
    th_b = locked_th["model_b_refined_mlp"]["threshold"]         # 0.37
    th_c = locked_th["model_c_calibrated_frozen_r2"]["threshold"] # 0.46

    print(f"Locked Decision Thresholds:")
    print(f"  Model A (Frozen R2 Baseline):            theta = {th_a:.2f}")
    print(f"  Model B (Cross-Mechanism Refined MLP):  theta = {th_b:.2f}")
    print(f"  Model C (Validation-Calibrated Frozen):  theta = {th_c:.2f}")

    # 2. Load models and scalers
    scaler_r2 = joblib.load(locked_th["model_a_frozen_r2"]["scaler_path"])
    model_r2 = SmallMLP(15, 32, 16)
    model_r2.load_state_dict(torch.load(locked_th["model_a_frozen_r2"]["model_path"], map_location="cpu"))
    model_r2.eval()

    scaler_ref = joblib.load(locked_th["model_b_refined_mlp"]["scaler_path"])
    model_ref = SmallMLP(15, 32, 16)
    model_ref.load_state_dict(torch.load(locked_th["model_b_refined_mlp"]["model_path"], map_location="cpu"))
    model_ref.eval()

    # 3. Load Protected Datasets
    # A. Protected R2 Test Flights (14 flights)
    r2_test_flights = split_manifest["r2_motion_mechanism"]["protected_test_flights"]
    df_r2_all = pd.read_csv(DATASET_R2)
    df_r2_test = df_r2_all[df_r2_all["run_dir"].isin(r2_test_flights)].copy().reset_index(drop=True)

    # B. Protected R3 Test Trials (15 trials)
    r3_test_trials = split_manifest["r3_visual_mechanism"]["protected_test_trials"]
    df_r3_all = pd.read_csv(DATASET_R3)
    df_r3_test = df_r3_all[df_r3_all["trial_id"].isin(r3_test_trials)].copy().reset_index(drop=True)

    # C. Full 40-flight R3 primary dataset (for complete Phase 1 comparison)
    df_r3_full = df_r3_all.copy()

    # 4. Generate Predictions for all datasets
    def predict_all(df_subset):
        X_raw = df_subset[feat_cols].values.astype(float)
        y = df_subset["is_failure_within_next_k"].values.astype(int)

        # Frozen Model (used by Model A and Model C)
        X_sc_r2 = scaler_r2.transform(X_raw)
        with torch.no_grad():
            p_r2 = torch.sigmoid(model_r2(torch.tensor(X_sc_r2, dtype=torch.float32))).numpy()

        # Refined Model (used by Model B)
        X_sc_ref = scaler_ref.transform(X_raw)
        with torch.no_grad():
            p_ref = torch.sigmoid(model_ref(torch.tensor(X_sc_ref, dtype=torch.float32))).numpy()

        return y, p_r2, p_ref

    y_r2_test, p_r2_test_frozen, p_r2_test_ref = predict_all(df_r2_test)
    y_r3_test, p_r3_test_frozen, p_r3_test_ref = predict_all(df_r3_test)
    y_r3_full, p_r3_full_frozen, p_r3_full_ref = predict_all(df_r3_full)

    # Combined Protected Test Set
    y_comb_test = np.concatenate([y_r2_test, y_r3_test])
    p_comb_frozen = np.concatenate([p_r2_test_frozen, p_r3_test_frozen])
    p_comb_ref = np.concatenate([p_r2_test_ref, p_r3_test_ref])

    # 5. Evaluate Metrics Across Test Sets
    evaluation_results = {
        "protected_r2_motion_test": {
            "model_a_frozen_r2": compute_metrics(y_r2_test, p_r2_test_frozen, th_a),
            "model_b_refined_mlp": compute_metrics(y_r2_test, p_r2_test_ref, th_b),
            "model_c_calibrated_frozen_r2": compute_metrics(y_r2_test, p_r2_test_frozen, th_c),
        },
        "protected_r3_visual_test": {
            "model_a_frozen_r2": compute_metrics(y_r3_test, p_r3_test_frozen, th_a),
            "model_b_refined_mlp": compute_metrics(y_r3_test, p_r3_test_ref, th_b),
            "model_c_calibrated_frozen_r2": compute_metrics(y_r3_test, p_r3_test_frozen, th_c),
        },
        "combined_protected_cross_mechanism_test": {
            "model_a_frozen_r2": compute_metrics(y_comb_test, p_comb_frozen, th_a),
            "model_b_refined_mlp": compute_metrics(y_comb_test, p_comb_ref, th_b),
            "model_c_calibrated_frozen_r2": compute_metrics(y_comb_test, p_comb_frozen, th_c),
        },
        "full_40_flight_r3_dataset": {
            "model_a_frozen_r2": compute_metrics(y_r3_full, p_r3_full_frozen, th_a),
            "model_b_refined_mlp": compute_metrics(y_r3_full, p_r3_full_ref, th_b),
            "model_c_calibrated_frozen_r2": compute_metrics(y_r3_full, p_r3_full_frozen, th_c),
        }
    }

    # 6. Episode-Level Early-Warning Analysis
    episode_results = {
        "protected_r2_motion_test": {
            "model_a_frozen_r2": analyze_episodes(df_r2_test, p_r2_test_frozen, th_a),
            "model_b_refined_mlp": analyze_episodes(df_r2_test, p_r2_test_ref, th_b),
            "model_c_calibrated_frozen_r2": analyze_episodes(df_r2_test, p_r2_test_frozen, th_c),
        },
        "protected_r3_visual_test": {
            "model_a_frozen_r2": analyze_episodes(df_r3_test, p_r3_test_frozen, th_a),
            "model_b_refined_mlp": analyze_episodes(df_r3_test, p_r3_test_ref, th_b),
            "model_c_calibrated_frozen_r2": analyze_episodes(df_r3_test, p_r3_test_frozen, th_c),
        },
        "full_40_flight_r3_dataset": {
            "model_a_frozen_r2": analyze_episodes(df_r3_full, p_r3_full_frozen, th_a),
            "model_b_refined_mlp": analyze_episodes(df_r3_full, p_r3_full_ref, th_b),
            "model_c_calibrated_frozen_r2": analyze_episodes(df_r3_full, p_r3_full_frozen, th_c),
        }
    }

    # 7. Per-Flight Metrics Table
    per_flight_rows = []

    # Per-flight R2 test flights
    for f_id, grp in df_r2_test.groupby("run_dir", sort=False):
        y_f, p_f_frozen, p_f_ref = predict_all(grp)
        mA = compute_metrics(y_f, p_f_frozen, th_a)
        mB = compute_metrics(y_f, p_f_ref, th_b)
        mC = compute_metrics(y_f, p_f_frozen, th_c)

        per_flight_rows.append({
            "flight_id": f_id,
            "domain": "R2_MOTION_TEST",
            "samples": len(grp),
            "model_a_auroc": mA["auroc"],
            "model_a_f1": mA["f1"],
            "model_a_rec": mA["recall"],
            "model_a_prec": mA["precision"],
            "model_b_auroc": mB["auroc"],
            "model_b_f1": mB["f1"],
            "model_b_rec": mB["recall"],
            "model_b_prec": mB["precision"],
            "model_c_f1": mC["f1"],
            "model_c_rec": mC["recall"],
            "model_c_prec": mC["precision"],
        })

    # Per-flight R3 test trials
    for f_id, grp in df_r3_test.groupby("trial_id", sort=False):
        y_f, p_f_frozen, p_f_ref = predict_all(grp)
        mA = compute_metrics(y_f, p_f_frozen, th_a)
        mB = compute_metrics(y_f, p_f_ref, th_b)
        mC = compute_metrics(y_f, p_f_frozen, th_c)

        per_flight_rows.append({
            "flight_id": f_id,
            "domain": "R3_VISUAL_TEST",
            "samples": len(grp),
            "model_a_auroc": mA["auroc"],
            "model_a_f1": mA["f1"],
            "model_a_rec": mA["recall"],
            "model_a_prec": mA["precision"],
            "model_b_auroc": mB["auroc"],
            "model_b_f1": mB["f1"],
            "model_b_rec": mB["recall"],
            "model_b_prec": mB["precision"],
            "model_c_f1": mC["f1"],
            "model_c_rec": mC["recall"],
            "model_c_prec": mC["precision"],
        })

    df_pf = pd.DataFrame(per_flight_rows)
    df_pf.to_csv(OUT_PER_FLIGHT_CSV, index=False)

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    with open(OUT_EPISODES_JSON, "w") as f:
        json.dump(episode_results, f, indent=2)

    # 8. Print Executive Summary
    print("\n" + "=" * 70)
    print("PROTECTED TEST EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    print("1. PROTECTED R3 VISUAL-DEGRADATION TEST (15 Trials, 11,460 Frames):")
    print("-" * 70)
    mA_v = evaluation_results["protected_r3_visual_test"]["model_a_frozen_r2"]
    mB_v = evaluation_results["protected_r3_visual_test"]["model_b_refined_mlp"]
    mC_v = evaluation_results["protected_r3_visual_test"]["model_c_calibrated_frozen_r2"]
    print(f"Model A (Frozen R2):       AUROC={mA_v['auroc']:.4f} | F1={mA_v['f1']:.4f} | Rec={mA_v['recall']:.4f} | Prec={mA_v['precision']:.4f}")
    print(f"Model B (Refined MLP):    AUROC={mB_v['auroc']:.4f} | F1={mB_v['f1']:.4f} | Rec={mB_v['recall']:.4f} | Prec={mB_v['precision']:.4f}")
    print(f"Model C (Calibrated R2):  AUROC={mC_v['auroc']:.4f} | F1={mC_v['f1']:.4f} | Rec={mC_v['recall']:.4f} | Prec={mC_v['precision']:.4f}")
    print(f"-> DELTA (B vs A):        dAUROC={mB_v['auroc']-mA_v['auroc']:+.4f} | dF1={mB_v['f1']-mA_v['f1']:+.4f} | dRec={mB_v['recall']-mA_v['recall']:+.4f} | dPrec={mB_v['precision']-mA_v['precision']:+.4f}")

    print("\n2. PROTECTED R2 MOTION-INDUCED TEST (14 Flights, 10,411 Frames):")
    print("-" * 70)
    mA_m = evaluation_results["protected_r2_motion_test"]["model_a_frozen_r2"]
    mB_m = evaluation_results["protected_r2_motion_test"]["model_b_refined_mlp"]
    mC_m = evaluation_results["protected_r2_motion_test"]["model_c_calibrated_frozen_r2"]
    print(f"Model A (Frozen R2):       AUROC={mA_m['auroc']:.4f} | F1={mA_m['f1']:.4f} | Rec={mA_m['recall']:.4f} | Prec={mA_m['precision']:.4f}")
    print(f"Model B (Refined MLP):    AUROC={mB_m['auroc']:.4f} | F1={mB_m['f1']:.4f} | Rec={mB_m['recall']:.4f} | Prec={mB_m['precision']:.4f}")
    print(f"Model C (Calibrated R2):  AUROC={mC_m['auroc']:.4f} | F1={mC_m['f1']:.4f} | Rec={mC_m['recall']:.4f} | Prec={mC_m['precision']:.4f}")
    print(f"-> DELTA (B vs A):        dAUROC={mB_m['auroc']-mA_m['auroc']:+.4f} | dF1={mB_m['f1']-mA_m['f1']:+.4f} | dRec={mB_m['recall']-mA_m['recall']:+.4f} | dPrec={mB_m['precision']-mA_m['precision']:+.4f}")

    print("\n3. CROSS-MECHANISM GENERALIZATION GAP (|Rec_motion - Rec_visual|):")
    print("-" * 70)
    gap_A = abs(mA_m['recall'] - mA_v['recall'])
    gap_B = abs(mB_m['recall'] - mB_v['recall'])
    gap_C = abs(mC_m['recall'] - mC_v['recall'])
    print(f"Model A (Frozen R2) Sensitivity Gap:            {gap_A*100:.2f}% (Motion: {mA_m['recall']*100:.1f}%, Visual: {mA_v['recall']*100:.1f}%)")
    print(f"Model B (Cross-Mechanism Refined) Gap:         {gap_B*100:.2f}% (Motion: {mB_m['recall']*100:.1f}%, Visual: {mB_v['recall']*100:.1f}%)")
    print(f"Model C (Calibrated Frozen R2) Gap:             {gap_C*100:.2f}% (Motion: {mC_m['recall']*100:.1f}%, Visual: {mC_v['recall']*100:.1f}%)")
    print(f"-> Cross-Mechanism Sensitivity Gap Reduction:   {(gap_A - gap_B)*100:+.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
