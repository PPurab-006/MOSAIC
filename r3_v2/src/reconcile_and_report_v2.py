#!/usr/bin/env python3
"""
Research 3 (MOSAIC) — V2 Reporting & Base-Flight Cluster Reconciliation.

1. Reconciles Macro-Average (mean over seeds 42-46, ddof=1) and Soft-Voting Ensemble definitions.
2. Evaluates without retraining Model A, Model B, or Frozen R2.
3. Generates per-trial AUROCs for each regime (3 trials per regime: G_C_R1, G_B_R2, G_S_R2).
4. Reports per-flight ranges (n=3) for visual regimes rather than pseudocontinuous CIs.
5. Computes meaningful paired cluster-bootstrap differences: B-A across all regimes, and
   B-D(MLP) strictly for Pooled (+0.033) and Transition (+0.028).
6. Regenerates all v2 summary tables and synchronizes standard deviations.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
torch.set_num_threads(2)

from r3_v2.src.extract_v2_features import (
    FEAT_A_COLS,
    FEAT_B_COLS,
    FEAT_IMG_COLS,
)
from r3_v2.src.models import (
    SmallMLP,
    set_seed,
    train_mlp_model,
)

R3_ROOT = REPO_ROOT
MODELS_DIR = R3_ROOT / "r3_v2" / "models"
RESULTS_DIR = R3_ROOT / "r3_v2" / "results"
DATA_PATH = R3_ROOT / "r3_v2" / "data" / "r3_v2_strict_dataset.csv"

FROZEN_MODEL_PATH = R3_ROOT / "models" / "expanded_mlp.pt"
FROZEN_SCALER_PATH = R3_ROOT / "models" / "expanded_scaler.joblib"

SEEDS = [42, 43, 44, 45, 46]
BASE_FLIGHTS_VIS = ["sweep_G_C_R1", "sweep_G_B_R2", "sweep_G_S_R2"]
REGIMES = ["NOMINAL", "LOW", "TRANSITION", "MEDIUM", "HIGH"]


def get_regime_features(df: pd.DataFrame) -> np.ndarray:
    regimes = ["NOMINAL", "LOW", "TRANSITION", "MEDIUM", "HIGH", "MOTION"]
    one_hots = [(df["regime"] == r).astype(float).values.reshape(-1, 1) for r in regimes]
    alpha_trans = np.where(df["regime"] == "TRANSITION", df["degradation_alpha"].values, 0.0).reshape(-1, 1)
    return np.hstack(one_hots + [alpha_trans])


def run_reconciliation():
    print("=" * 80)
    print("RESEARCH 3 (MOSAIC) — V2 RECONCILIATION & REPORTING CLEANUP")
    print("=" * 80)

    # 1. Load strict dataset
    df = pd.read_csv(DATA_PATH)
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)
    df_val = df[df["split"] == "val"].copy().reset_index(drop=True)
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)

    y_train = df_train["is_failure_within_next_k"].values.astype(float)
    y_val = df_val["is_failure_within_next_k"].values.astype(float)
    y_test = df_test["is_failure_within_next_k"].values.astype(int)

    vis_mask = (df_test["domain"] == "R3_VISUAL").values
    mot_mask = (df_test["domain"] == "R2_MOTION").values

    # 2. Frozen R2
    print("\n[1/5] Loading Frozen R2 baseline checkpoint...")
    frozen_scaler = joblib.load(FROZEN_SCALER_PATH)
    frozen_model = SmallMLP(15, 32, 16)
    frozen_model.load_state_dict(torch.load(FROZEN_MODEL_PATH, map_location="cpu"))
    frozen_model.eval()

    X_te_frozen = frozen_scaler.transform(df_test[FEAT_A_COLS].values.astype(float))
    with torch.no_grad():
        p_frozen = torch.sigmoid(frozen_model(torch.tensor(X_te_frozen, dtype=torch.float32))).numpy()

    # 3. Model A and Model B from disk across 5 seeds
    print("\n[2/5] Loading Model A & Model B checkpoints and fitting Baselines C & D across 5 seeds...")
    p_a_by_seed = []
    p_b_by_seed = []
    p_c_mlp_by_seed = []
    p_c_lr_by_seed = []
    p_d_mlp_by_seed = []
    p_d_lr_by_seed = []

    # Features for Baseline C
    X_tr_c_raw = df_train[FEAT_IMG_COLS].values.astype(float)
    X_va_c_raw = df_val[FEAT_IMG_COLS].values.astype(float)
    X_te_c_raw = df_test[FEAT_IMG_COLS].values.astype(float)

    # Features for Baseline D
    X_tr_d_raw = get_regime_features(df_train)
    X_va_d_raw = get_regime_features(df_val)
    X_te_d_raw = get_regime_features(df_test)

    for seed in SEEDS:
        set_seed(seed)
        # Load saved Model A
        sc_a = joblib.load(MODELS_DIR / f"scaler_a_seed_{seed}.joblib")
        m_a = SmallMLP(15, 32, 16)
        m_a.load_state_dict(torch.load(MODELS_DIR / f"model_a_seed_{seed}.pt", map_location="cpu"))
        m_a.eval()
        with torch.no_grad():
            p_a = torch.sigmoid(m_a(torch.tensor(sc_a.transform(df_test[FEAT_A_COLS].values.astype(float)), dtype=torch.float32))).numpy()
        p_a_by_seed.append(p_a)

        # Load saved Model B
        sc_b = joblib.load(MODELS_DIR / f"scaler_b_seed_{seed}.joblib")
        m_b = SmallMLP(35, 32, 16)
        m_b.load_state_dict(torch.load(MODELS_DIR / f"model_b_seed_{seed}.pt", map_location="cpu"))
        m_b.eval()
        with torch.no_grad():
            p_b = torch.sigmoid(m_b(torch.tensor(sc_b.transform(df_test[FEAT_B_COLS].values.astype(float)), dtype=torch.float32))).numpy()
        p_b_by_seed.append(p_b)

        # Baseline C: MLP
        sc_c = StandardScaler()
        Xtr_c = sc_c.fit_transform(X_tr_c_raw)
        Xva_c = sc_c.transform(X_va_c_raw)
        Xte_c = sc_c.transform(X_te_c_raw)
        m_c, _ = train_mlp_model(Xtr_c, y_train, Xva_c, y_val, input_dim=20, seed=seed)
        with torch.no_grad():
            p_c_mlp = torch.sigmoid(m_c(torch.tensor(Xte_c, dtype=torch.float32))).numpy()
        p_c_mlp_by_seed.append(p_c_mlp)

        # Baseline C: LogReg
        clf_c = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf_c.fit(Xtr_c, y_train)
        p_c_lr_by_seed.append(clf_c.predict_proba(Xte_c)[:, 1])

        # Baseline D: MLP
        sc_d = StandardScaler()
        Xtr_d = sc_d.fit_transform(X_tr_d_raw)
        Xva_d = sc_d.transform(X_va_d_raw)
        Xte_d = sc_d.transform(X_te_d_raw)
        m_d, _ = train_mlp_model(Xtr_d, y_train, Xva_d, y_val, input_dim=7, seed=seed)
        with torch.no_grad():
            p_d_mlp = torch.sigmoid(m_d(torch.tensor(Xte_d, dtype=torch.float32))).numpy()
        p_d_mlp_by_seed.append(p_d_mlp)

        # Baseline D: LogReg
        clf_d = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf_d.fit(Xtr_d, y_train)
        p_d_lr_by_seed.append(clf_d.predict_proba(Xte_d)[:, 1])

    # Ensembles
    p_a_ens = np.mean(p_a_by_seed, axis=0)
    p_b_ens = np.mean(p_b_by_seed, axis=0)
    p_c_mlp_ens = np.mean(p_c_mlp_by_seed, axis=0)
    p_c_lr_ens = np.mean(p_c_lr_by_seed, axis=0)
    p_d_mlp_ens = np.mean(p_d_mlp_by_seed, axis=0)
    p_d_lr_ens = np.mean(p_d_lr_by_seed, axis=0)

    model_seed_probs = {
        "frozen_r2": [p_frozen] * len(SEEDS),
        "model_a": p_a_by_seed,
        "model_b": p_b_by_seed,
        "baseline_c_mlp": p_c_mlp_by_seed,
        "baseline_c_lr": p_c_lr_by_seed,
        "baseline_d_mlp": p_d_mlp_by_seed,
        "baseline_d_lr": p_d_lr_by_seed,
    }

    model_ens_probs = {
        "frozen_r2": p_frozen,
        "model_a": p_a_ens,
        "model_b": p_b_ens,
        "baseline_c_mlp": p_c_mlp_ens,
        "baseline_c_lr": p_c_lr_ens,
        "baseline_d_mlp": p_d_mlp_ens,
        "baseline_d_lr": p_d_lr_ens,
    }

    # 4. Per-Trial AUROC Listing for Each Regime (Requirement 2)
    print("\n[3/5] Listing Per-Trial AUROC for each regime (3 base flight trials per regime)...")
    per_trial_rows = []
    te_vis = df_test[vis_mask].copy()

    trial_auroc_cache = {m_key: {reg: {} for reg in REGIMES} for m_key in model_ens_probs}

    for reg in REGIMES:
        sub_reg = te_vis[te_vis["regime"] == reg]
        for bf in BASE_FLIGHTS_VIS:
            trial_sub = sub_reg[sub_reg["base_flight"] == bf]
            seq_id = trial_sub["sequence_id"].iloc[0]
            y_tr = trial_sub["is_failure_within_next_k"].values
            n_frames = len(trial_sub)
            n_pos = int(y_tr.sum())
            is_single_class = (len(np.unique(y_tr)) < 2)

            idx_tr = trial_sub.index.values

            row = {
                "regime": reg,
                "base_flight": bf,
                "sequence_id": seq_id,
                "frames": n_frames,
                "positives": n_pos,
                "positive_rate_pct": round(n_pos / n_frames * 100.0, 1),
            }

            for m_key, m_ens in model_ens_probs.items():
                if is_single_class:
                    row[f"{m_key}_auroc"] = "Undefined (single-class)"
                    trial_auroc_cache[m_key][reg][bf] = float("nan")
                else:
                    auc_val = float(roc_auc_score(y_tr, m_ens[idx_tr]))
                    row[f"{m_key}_auroc"] = round(auc_val, 4)
                    trial_auroc_cache[m_key][reg][bf] = auc_val

            per_trial_rows.append(row)

    df_per_trial = pd.DataFrame(per_trial_rows)
    df_per_trial.to_csv(RESULTS_DIR / "per_trial_regime_auroc.csv", index=False)
    print(df_per_trial[["regime", "base_flight", "frozen_r2_auroc", "model_a_auroc", "model_b_auroc", "baseline_c_mlp_auroc", "baseline_d_mlp_auroc"]].to_string(index=False))

    # 5. Base-Flight Cluster Bootstrap (Requirement 2 & 3)
    print("\n[4/5] Executing Base-Flight Cluster Bootstrap (1,000 resamples, seed 42)...")
    rng = np.random.RandomState(42)
    B = 1000

    base_flights_arr = np.array(BASE_FLIGHTS_VIS)
    bf_to_indices = {bf: df_test[(df_test["domain"] == "R3_VISUAL") & (df_test["base_flight"] == bf)].index.values for bf in base_flights_arr}

    r2_motion_flights = np.array(sorted(df_test[df_test["domain"] == "R2_MOTION"]["sequence_id"].unique()))
    seq_to_indices_mot = {s: df_test[(df_test["domain"] == "R2_MOTION") & (df_test["sequence_id"] == s)].index.values for s in r2_motion_flights}

    # Store bootstrap metrics
    boot_model_metrics = {
        m: {
            "POOLED": [],
            "MOTION": [],
        }
        for m in model_ens_probs
    }

    # Keep ONLY meaningful paired differences: B-A across regimes, and B-D(MLP) strictly for POOLED and TRANSITION
    boot_paired_diffs = {
        "B_minus_A": {"POOLED": [], "NOMINAL": [], "LOW": [], "TRANSITION": [], "MEDIUM": []},
        "B_minus_D_mlp": {"POOLED": [], "TRANSITION": []},
    }

    for b in range(B):
        # 1. Resample visual base flights
        resampled_bf = rng.choice(base_flights_arr, size=3, replace=True)
        idx_b_vis = np.concatenate([bf_to_indices[bf] for bf in resampled_bf])
        sub_vis = df_test.loc[idx_b_vis]
        y_b_vis = sub_vis["is_failure_within_next_k"].values

        # 2. Resample motion test flights
        resampled_mf = rng.choice(r2_motion_flights, size=len(r2_motion_flights), replace=True)
        idx_b_mot = np.concatenate([seq_to_indices_mot[s] for s in resampled_mf])
        y_b_mot = df_test.loc[idx_b_mot, "is_failure_within_next_k"].values

        # Pooled visual
        for m_key, p_m in model_ens_probs.items():
            auc_vis_b = float(roc_auc_score(y_b_vis, p_m[idx_b_vis]))
            boot_model_metrics[m_key]["POOLED"].append(auc_vis_b)

            if len(np.unique(y_b_mot)) > 1:
                auc_mot_b = float(roc_auc_score(y_b_mot, p_m[idx_b_mot]))
                boot_model_metrics[m_key]["MOTION"].append(auc_mot_b)

        # Paired differences
        p_b_vis = model_ens_probs["model_b"][idx_b_vis]
        p_a_vis = model_ens_probs["model_a"][idx_b_vis]
        p_d_vis = model_ens_probs["baseline_d_mlp"][idx_b_vis]

        auc_b_pool = roc_auc_score(y_b_vis, p_b_vis)
        auc_a_pool = roc_auc_score(y_b_vis, p_a_vis)
        auc_d_pool = roc_auc_score(y_b_vis, p_d_vis)

        boot_paired_diffs["B_minus_A"]["POOLED"].append(auc_b_pool - auc_a_pool)
        boot_paired_diffs["B_minus_D_mlp"]["POOLED"].append(auc_b_pool - auc_d_pool)

        for reg in ["NOMINAL", "LOW", "TRANSITION", "MEDIUM"]:
            m_reg = (sub_vis["regime"] == reg).values
            y_r = y_b_vis[m_reg]
            idx_r = idx_b_vis[m_reg]

            auc_b_r = roc_auc_score(y_r, model_ens_probs["model_b"][idx_r])
            auc_a_r = roc_auc_score(y_r, model_ens_probs["model_a"][idx_r])
            boot_paired_diffs["B_minus_A"][reg].append(auc_b_r - auc_a_r)

            if reg == "TRANSITION":
                auc_d_r = roc_auc_score(y_r, model_ens_probs["baseline_d_mlp"][idx_r])
                boot_paired_diffs["B_minus_D_mlp"]["TRANSITION"].append(auc_b_r - auc_d_r)

    # 6. Build Paired Cluster-Bootstrap Differences Table (Requirement 3)
    paired_rows = []
    for comp_name, comp_dict in boot_paired_diffs.items():
        for target, diff_vals in comp_dict.items():
            arr = np.array(diff_vals)
            paired_rows.append({
                "comparison": comp_name,
                "target_subset": target,
                "mean_diff": round(float(np.mean(arr)), 4),
                "ci_lower_2.5": round(float(np.percentile(arr, 2.5)), 4),
                "ci_upper_97.5": round(float(np.percentile(arr, 97.5)), 4),
                "ci_display": f"{np.mean(arr):+.4f} [{np.percentile(arr, 2.5):+.4f}, {np.percentile(arr, 97.5):+.4f}]",
            })
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(RESULTS_DIR / "cluster_bootstrap_paired_differences.csv", index=False)
    print("\n--- Filtered Meaningful Paired Cluster Bootstrap Differences (95% CIs) ---")
    print(df_paired[["comparison", "target_subset", "ci_display"]].to_string(index=False))

    # 7. Compute Macro-Average across seeds (Mean ± Std, sample std ddof=1)
    print("\n[5/5] Regenerating Comprehensive Master Comparison Table with Per-Flight Ranges (n=3)...")
    model_labels = [
        ("Frozen R2 Baseline", "frozen_r2"),
        ("Model A (15 Motion Features)", "model_a"),
        ("Model B (35 Features: Motion + Image)", "model_b"),
        ("Baseline C (MLP: 20 Image Features Only)", "baseline_c_mlp"),
        ("Baseline C (LogReg: 20 Image Features Only)", "baseline_c_lr"),
        ("Baseline D (MLP: Regime ID + Alpha)", "baseline_d_mlp"),
        ("Baseline D (LogReg: Regime ID + Alpha)", "baseline_d_lr"),
    ]

    comp_table_rows = []

    for display_label, m_key in model_labels:
        # Per-seed metrics across seeds 42-46
        seed_vis_aucs = []
        seed_mot_aucs = []

        for p_s in model_seed_probs[m_key]:
            seed_vis_aucs.append(float(roc_auc_score(y_test[vis_mask], p_s[vis_mask])))
            seed_mot_aucs.append(float(roc_auc_score(y_test[mot_mask], p_s[mot_mask])))

        # Sample standard deviation across seeds (ddof=1)
        def mean_std_str(arr):
            m = np.mean(arr)
            s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            return f"{m:.4f} ± {s:.4f}"

        # Per-flight range for visual regimes (min and max across the 3 base flight trials)
        def per_flight_range_str(reg_name):
            vals = list(trial_auroc_cache[m_key][reg_name].values())
            if any(np.isnan(v) for v in vals):
                return "Undefined (single-class)"
            min_v = min(vals)
            max_v = max(vals)
            # overall ensemble or macro mean for that regime
            r_mask = (df_test["domain"] == "R3_VISUAL") & (df_test["regime"] == reg_name)
            ens_r = float(roc_auc_score(y_test[r_mask.values], model_ens_probs[m_key][r_mask.values]))
            return f"{ens_r:.4f} [{min_v:.4f}, {max_v:.4f}]"

        # Pooled visual cluster bootstrap CI (across 3 base flights)
        arr_pool = boot_model_metrics[m_key]["POOLED"]
        lo_pool = np.percentile(arr_pool, 2.5)
        hi_pool = np.percentile(arr_pool, 97.5)
        ens_vis = roc_auc_score(y_test[vis_mask], model_ens_probs[m_key][vis_mask])
        pooled_vis_range_str = f"{ens_vis:.4f} [{lo_pool:.4f}, {hi_pool:.4f}]"

        # Motion cluster bootstrap CI (across 14 flights)
        arr_mot = boot_model_metrics[m_key]["MOTION"]
        lo_mot = np.percentile(arr_mot, 2.5)
        hi_mot = np.percentile(arr_mot, 97.5)
        ens_mot = roc_auc_score(y_test[mot_mask], model_ens_probs[m_key][mot_mask])
        motion_ci_str = f"{ens_mot:.4f} [{lo_mot:.4f}, {hi_mot:.4f}]"

        comp_table_rows.append({
            "model": display_label,
            "pooled_visual_auroc_mean_std": mean_std_str(seed_vis_aucs),
            "pooled_visual_per_flight_range_n3": pooled_vis_range_str,
            "nominal_per_flight_range_n3": per_flight_range_str("NOMINAL"),
            "low_per_flight_range_n3": per_flight_range_str("LOW"),
            "transition_per_flight_range_n3": per_flight_range_str("TRANSITION"),
            "medium_per_flight_range_n3": per_flight_range_str("MEDIUM"),
            "high_per_flight_range_n3": "Undefined (single-class)",
            "motion_auroc_mean_std": mean_std_str(seed_mot_aucs),
            "motion_flight_cluster_ci_n14": motion_ci_str,
        })

    df_comp = pd.DataFrame(comp_table_rows)
    df_comp.to_csv(RESULTS_DIR / "comprehensive_model_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print("RECONCILED V2 MASTER COMPARISON TABLE (PER-FLIGHT RANGE n=3)")
    print("=" * 90)
    print(df_comp[["model", "pooled_visual_auroc_mean_std", "nominal_per_flight_range_n3", "low_per_flight_range_n3", "transition_per_flight_range_n3", "medium_per_flight_range_n3", "motion_auroc_mean_std"]].to_string(index=False))
    print("=" * 90)

    # Update summary JSON
    with open(RESULTS_DIR / "v2_summary.json", "r") as f:
        v2_summary = json.load(f)

    v2_summary["reconciled_master_comparison"] = comp_table_rows
    v2_summary["cluster_bootstrap_paired_differences"] = paired_rows
    v2_summary["per_trial_regime_auroc"] = per_trial_rows

    with open(RESULTS_DIR / "v2_summary.json", "w") as f:
        json.dump(v2_summary, f, indent=2)
    print(f"\n-> Updated summary JSON: {RESULTS_DIR / 'v2_summary.json'}")


if __name__ == "__main__":
    run_reconciliation()
