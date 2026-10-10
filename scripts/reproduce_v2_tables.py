#!/usr/bin/env python3
"""
Research 3 (MOSAIC) — Standalone Table Reproduction Script.

Recomputes pooled AUROC, per-regime AUROC, and motion AUROC from:
- `r3_v2/data/r3_v2_strict_dataset.csv`
- Saved model checkpoints and scalers in `r3_v2/models/` and `models/`
- Standard Logistic Regression baselines for C (20 image features) and D (regime ID + alpha)

Compares recomputed metrics directly against `r3_v2/results/comprehensive_model_comparison.csv`
and reports the maximum absolute difference.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

from r3_v2.src.evaluate_baselines_c_d import get_regime_features
from r3_v2.src.extract_v2_features import (
    FEAT_A_COLS,
    FEAT_B_COLS,
    FEAT_IMG_COLS,
)
from r3_v2.src.models import SmallMLP

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT_DATASET_PATH = REPO_ROOT / "r3_v2" / "data" / "r3_v2_strict_dataset.csv"
V2_MODELS_DIR = REPO_ROOT / "r3_v2" / "models"
FROZEN_MODEL_PATH = REPO_ROOT / "models" / "expanded_mlp.pt"
FROZEN_SCALER_PATH = REPO_ROOT / "models" / "expanded_scaler.joblib"
EXPECTED_TABLE_PATH = REPO_ROOT / "r3_v2" / "results" / "comprehensive_model_comparison.csv"

SEEDS = [42, 43, 44, 45, 46]
REGIMES = ["NOMINAL", "LOW", "TRANSITION", "MEDIUM", "HIGH"]
BASE_FLIGHTS_VIS = ["sweep_G_C_R1", "sweep_G_B_R2", "sweep_G_S_R2"]


def main():
    print("=" * 80)
    print("MOSAIC (Research 3) — V2 Table Reproduction and Numerical Verification")
    print("=" * 80)

    if not STRICT_DATASET_PATH.exists():
        print(f"ERROR: Dataset file not found: {STRICT_DATASET_PATH}")
        sys.exit(1)

    print(f"Loading strict dataset: {STRICT_DATASET_PATH}...")
    df = pd.read_csv(STRICT_DATASET_PATH)
    print(f"Total dataset frames: {len(df):,} across {df['sequence_id'].nunique()} sequences.")

    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)
    df_val = df[df["split"] == "val"].copy().reset_index(drop=True)
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)

    y_train = df_train["is_failure_within_next_k"].values.astype(int)
    y_test = df_test["is_failure_within_next_k"].values.astype(int)

    vis_mask = (df_test["domain"] == "R3_VISUAL").values
    mot_mask = (df_test["domain"] == "R2_MOTION").values

    # 1. Evaluate Frozen R2 Model
    print("\n[1/5] Evaluating Frozen R2 Baseline...")
    frozen_scaler = joblib.load(FROZEN_SCALER_PATH)
    frozen_model = SmallMLP(15, 32, 16)
    frozen_model.load_state_dict(torch.load(FROZEN_MODEL_PATH, map_location="cpu"))
    frozen_model.eval()

    X_te_frozen = frozen_scaler.transform(df_test[FEAT_A_COLS].values.astype(float))
    with torch.no_grad():
        p_frozen = torch.sigmoid(frozen_model(torch.tensor(X_te_frozen, dtype=torch.float32))).numpy()

    # 2. Evaluate Model A and Model B across seeds 42..46
    print("\n[2/5] Evaluating Model A and Model B across seeds 42..46 from saved weights...")
    p_a_seeds = []
    p_b_seeds = []

    for seed in SEEDS:
        # Model A
        sc_a = joblib.load(V2_MODELS_DIR / f"scaler_a_seed_{seed}.joblib")
        m_a = SmallMLP(15, 32, 16)
        m_a.load_state_dict(torch.load(V2_MODELS_DIR / f"model_a_seed_{seed}.pt", map_location="cpu"))
        m_a.eval()
        with torch.no_grad():
            p_a = torch.sigmoid(m_a(torch.tensor(sc_a.transform(df_test[FEAT_A_COLS].values.astype(float)), dtype=torch.float32))).numpy()
        p_a_seeds.append(p_a)

        # Model B
        sc_b = joblib.load(V2_MODELS_DIR / f"scaler_b_seed_{seed}.joblib")
        m_b = SmallMLP(35, 32, 16)
        m_b.load_state_dict(torch.load(V2_MODELS_DIR / f"model_b_seed_{seed}.pt", map_location="cpu"))
        m_b.eval()
        with torch.no_grad():
            p_b = torch.sigmoid(m_b(torch.tensor(sc_b.transform(df_test[FEAT_B_COLS].values.astype(float)), dtype=torch.float32))).numpy()
        p_b_seeds.append(p_b)

    # 3. Fit and evaluate Logistic Baselines C & D
    print("\n[3/5] Fitting Logistic Regression Baselines C (20 image features) and D (regime ID + alpha)...")
    sc_c = StandardScaler()
    X_tr_c = sc_c.fit_transform(df_train[FEAT_IMG_COLS].values.astype(float))
    X_te_c = sc_c.transform(df_test[FEAT_IMG_COLS].values.astype(float))
    clf_c = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_c.fit(X_tr_c, y_train)
    p_c_lr = clf_c.predict_proba(X_te_c)[:, 1]

    sc_d = StandardScaler()
    X_tr_d = sc_d.fit_transform(get_regime_features(df_train))
    X_te_d = sc_d.transform(get_regime_features(df_test))
    clf_d = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_d.fit(X_tr_d, y_train)
    p_d_lr = clf_d.predict_proba(X_te_d)[:, 1]

    # Model probability mapping for comparison
    models = {
        "Frozen R2 Baseline": [p_frozen],
        "Model A (15 Motion Features)": p_a_seeds,
        "Model B (35 Features: Motion + Image)": p_b_seeds,
        "Baseline C (LogReg: 20 Image Features Only)": [p_c_lr],
        "Baseline D (LogReg: Regime ID + Alpha)": [p_d_lr],
    }

    # 4. Recompute AUROC metrics
    print("\n[4/5] Computing recomputed AUROCs...")
    recomputed_results = {}

    for name, p_list in models.items():
        # Pooled visual
        vis_aucs = [roc_auc_score(y_test[vis_mask], p[vis_mask]) for p in p_list]
        mean_vis = float(np.mean(vis_aucs))
        std_vis = float(np.std(vis_aucs, ddof=1)) if len(vis_aucs) > 1 else 0.0

        # Motion
        mot_aucs = [roc_auc_score(y_test[mot_mask], p[mot_mask]) for p in p_list]
        mean_mot = float(np.mean(mot_aucs))
        std_mot = float(np.std(mot_aucs, ddof=1)) if len(mot_aucs) > 1 else 0.0

        # Per-regime range across 3 base flights (using soft-voting ensemble probabilities)
        p_ens = np.mean(p_list, axis=0)
        reg_ranges = {}
        for reg in REGIMES:
            sub = df_test[vis_mask & (df_test["regime"] == reg)]
            bf_aucs = []
            for bf in BASE_FLIGHTS_VIS:
                bf_sub = sub[sub["base_flight"] == bf]
                y_bf = bf_sub["is_failure_within_next_k"].values
                if len(np.unique(y_bf)) > 1:
                    idx_bf = bf_sub.index.values
                    bf_aucs.append(roc_auc_score(y_bf, p_ens[idx_bf]))
                else:
                    bf_aucs.append(float("nan"))

            if any(np.isnan(v) for v in bf_aucs):
                reg_ranges[reg] = "Undefined (single-class)"
            else:
                ens_r = roc_auc_score(sub["is_failure_within_next_k"].values, p_ens[sub.index.values])
                reg_ranges[reg] = (ens_r, min(bf_aucs), max(bf_aucs))

        recomputed_results[name] = {
            "pooled_vis_mean": mean_vis,
            "pooled_vis_std": std_vis,
            "motion_mean": mean_mot,
            "motion_std": std_mot,
            "regimes": reg_ranges,
        }

    # 5. Load and compare with expected table
    print("\n[5/5] Comparing against authoritative comprehensive_model_comparison.csv...")
    df_expected = pd.read_csv(EXPECTED_TABLE_PATH)
    max_abs_diff = 0.0

    print("\n" + "=" * 90)
    print(f"{'Model':<40} | {'Metric':<18} | {'Recomputed':<12} | {'Expected':<12} | {'Abs Diff':<10}")
    print("-" * 90)

    for _, row in df_expected.iterrows():
        model_name = row["model"]
        if model_name not in recomputed_results:
            continue

        res = recomputed_results[model_name]

        # Check Pooled Visual Mean
        exp_vis_str = str(row["pooled_visual_auroc_mean_std"]).split(" ± ")[0]
        exp_vis = float(exp_vis_str)
        rec_vis = res["pooled_vis_mean"]
        diff_vis = abs(rec_vis - exp_vis)
        max_abs_diff = max(max_abs_diff, diff_vis)
        print(f"{model_name:<40} | {'Pooled Vis AUROC':<18} | {rec_vis:<12.4f} | {exp_vis:<12.4f} | {diff_vis:<10.6f}")

        # Check Motion Mean
        exp_mot_str = str(row["motion_auroc_mean_std"]).split(" ± ")[0]
        exp_mot = float(exp_mot_str)
        rec_mot = res["motion_mean"]
        diff_mot = abs(rec_mot - exp_mot)
        max_abs_diff = max(max_abs_diff, diff_mot)
        print(f"{model_name:<40} | {'Motion AUROC':<18} | {rec_mot:<12.4f} | {exp_mot:<12.4f} | {diff_mot:<10.6f}")

        # Check Transition Range
        exp_trans = row["transition_per_flight_range_n3"]
        rec_trans_val = res["regimes"]["TRANSITION"][0]
        exp_trans_val = float(str(exp_trans).split(" ")[0])
        diff_trans = abs(rec_trans_val - exp_trans_val)
        max_abs_diff = max(max_abs_diff, diff_trans)
        print(f"{model_name:<40} | {'Transition Ens':<18} | {rec_trans_val:<12.4f} | {exp_trans_val:<12.4f} | {diff_trans:<10.6f}")

    print("=" * 90)
    print(f"VERIFICATION VERDICT: Maximum absolute difference = {max_abs_diff:.8f}")
    if max_abs_diff < 1e-4:
        print("PERFECT REPRODUCIBILITY CONFIRMED (Diff < 1e-4 across all recomputed metrics).")
        sys.exit(0)
    else:
        print("WARNING: Max absolute difference exceeds tolerance 1e-4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
