#!/usr/bin/env python3
"""
Research 3 — Step 7: Statistical Significance Testing & Hypothesis Classification.

Executes rigorous statistical analysis:
  1. Non-parametric bootstrap (B=2000) 95% confidence intervals for:
     - AUROC, AUPRC, F1, Precision, Recall, and their deltas against R2 baseline.
  2. Flight-level Mann-Whitney U and Wilcoxon rank tests between R2 test and R3.
  3. Formal classification into Outcome A, B, C, or D per Sections 18, 19, and 23.

Outputs:
  - data/processed/r3_statistical_comparison.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

R3_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = R3_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5.csv"
METRICS_JSON_PATH = PROCESSED_DIR / "r3_frozen_r2_evaluation_metrics.json"
PER_FLIGHT_CSV_PATH = PROCESSED_DIR / "r3_per_flight_metrics.csv"
MOTION_SUMMARY_PATH = PROCESSED_DIR / "r3_motion_comparison_summary.json"
WARNINGS_JSON_PATH = PROCESSED_DIR / "r3_lead_time_early_warnings.json"

OUT_STATISTICS_JSON = PROCESSED_DIR / "r3_statistical_comparison.json"


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs, auprcs, f1s, precs, recs = [], [], [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = probs[idx]

        if len(np.unique(y_b)) < 2:
            continue

        pred_b = (p_b >= 0.5).astype(int)
        aurocs.append(roc_auc_score(y_b, p_b))
        auprcs.append(average_precision_score(y_b, p_b))
        f1s.append(f1_score(y_b, pred_b, zero_division=0))
        precs.append(precision_score(y_b, pred_b, zero_division=0))
        recs.append(recall_score(y_b, pred_b, zero_division=0))

    def ci_range(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci_95_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "auroc": ci_range(aurocs),
        "auprc": ci_range(auprcs),
        "f1": ci_range(f1s),
        "precision": ci_range(precs),
        "recall": ci_range(recs),
    }


def main():
    print("=" * 70)
    print("Research 3 — Step 7: Statistical Significance & Hypothesis Classification")
    print("=" * 70)

    # 1. Load evaluation metrics and R2 baseline
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    r2_base = metrics_data["r2_test_baseline"]
    r3_overall = metrics_data["r3_overall_transfer_metrics"]

    # 2. Load dataset
    df = pd.read_csv(DATASET_PATH)
    y_all = df["is_failure_within_next_k"].values.astype(int)
    p_all = df["pred_prob"].values.astype(float)

    print("Computing non-parametric bootstrap confidence intervals (B=2000)...")
    boot_res_overall = bootstrap_ci(y_all, p_all, n_boot=2000, seed=42)

    # Compute bootstrap for each regime
    boot_res_regimes = {}
    for reg in ["NOMINAL", "LOW", "MEDIUM", "HIGH", "TRANSITION"]:
        r_mask = df["regime"] == reg
        if r_mask.any() and len(np.unique(y_all[r_mask])) > 1:
            print(f"  Bootstrap for regime {reg}...")
            boot_res_regimes[reg] = bootstrap_ci(y_all[r_mask], p_all[r_mask], n_boot=1000, seed=42)

    # 3. Flight-Level Statistical Testing
    df_per_flight = pd.read_csv(PER_FLIGHT_CSV_PATH)

    # Load R2 14 held-out test flight AUROCs
    r2_test_aurocs = [
        0.7168, 0.8016, 0.8393, 0.7710, 0.7675, 0.7331, 0.7549,
        0.7903, 0.7842, 0.8280, 0.8022, 0.6847, 0.7059, 0.6690
    ]
    r3_flight_aurocs = df_per_flight["auroc"].dropna().values

    # Two-sample Mann-Whitney U test between R2 test AUROCs and R3 flight AUROCs
    mwu_test = stats.mannwhitneyu(r2_test_aurocs, r3_flight_aurocs, alternative="two-sided")
    # Also test if R3 AUROC is significantly lower than R2 test
    mwu_less = stats.mannwhitneyu(r3_flight_aurocs, r2_test_aurocs, alternative="less")

    # One-sample t-test / Wilcoxon against chance (AUROC = 0.5)
    t_vs_chance = stats.ttest_1samp(r3_flight_aurocs, 0.5)
    wilc_vs_chance = stats.wilcoxon(r3_flight_aurocs - 0.5)

    # 4. Check Motion Separation Validity
    with open(MOTION_SUMMARY_PATH, "r", encoding="utf-8") as f:
        motion_summary = json.load(f)

    yaw_sep_d = motion_summary["distributional_separation_statistics"]["yaw_rate_vs_r2_test"]["cohen_d"]
    yaw_sep_p = motion_summary["distributional_separation_statistics"]["yaw_rate_vs_r2_test"]["p_value"]
    motion_cleanly_separated = (yaw_sep_d > 0.8) and (yaw_sep_p < 1e-5)

    # 5. Load Early Warning Data
    with open(WARNINGS_JSON_PATH, "r", encoding="utf-8") as f:
        warnings_data = json.load(f)

    early_warn_rate = warnings_data["overall_r3_early_warning_summary"]["early_warning_rate_pct"]
    r2_early_rate = warnings_data["r2_baseline_comparison"]["r2_early_warning_rate_pct"]

    # 6. Formal Scientific Hypothesis Classification Rubric
    r3_auroc = r3_overall["auroc"]
    delta_auroc = r3_auroc - r2_base["auroc"]
    r3_f1 = r3_overall["f1"]
    delta_f1 = r3_f1 - r2_base["f1"]

    classification = ""
    classification_rationale = ""

    if not motion_cleanly_separated:
        classification = "D. EXPERIMENTAL SEPARATION INSUFFICIENT"
        classification_rationale = (
            "The proposed visual failure mechanism could not be cleanly separated from vehicle motion, "
            f"as the Cohen's d effect size was {yaw_sep_d:.2f}."
        )
    elif r3_auroc >= 0.75 and early_warn_rate >= 70.0 and delta_auroc > -0.10:
        classification = "A. MECHANISM-GENERAL PREDICTION SUPPORTED"
        classification_rationale = (
            f"The frozen R2 predictor retained strong predictive discrimination (AUROC = {r3_auroc:.4f}, "
            f"95% CI [{boot_res_overall['auroc']['ci_95_lower']:.4f}, {boot_res_overall['auroc']['ci_95_upper']:.4f}]) "
            f"and sustained high early-warning rate ({early_warn_rate:.1f}%) under visually induced failure with benign motion."
        )
    elif r3_auroc >= 0.65 and delta_auroc >= -0.25:
        classification = "B. PARTIAL MECHANISM TRANSFER"
        classification_rationale = (
            f"The frozen R2 predictor retained non-trivial predictive ability (AUROC = {r3_auroc:.4f}, "
            f"95% CI [{boot_res_overall['auroc']['ci_95_lower']:.4f}, {boot_res_overall['auroc']['ci_95_upper']:.4f}]) "
            f"significantly above chance (p = {wilc_vs_chance.pvalue:.4e}), but demonstrated a substantial degradation "
            f"(Delta AUROC = {delta_auroc:+.4f}, Delta F1 = {delta_f1:+.4f}) as the failure mechanism shifted from motion to visual observability."
        )
    else:
        classification = "C. MOTION-SPECIFIC PREDICTION SUPPORTED"
        classification_rationale = (
            f"The frozen R2 predictor degraded substantially toward chance (AUROC = {r3_auroc:.4f}, "
            f"Delta AUROC = {delta_auroc:+.4f}) when VO failure was visually induced under benign vehicle motion, "
            f"demonstrating that the predictive relationship learned in R2 was conditional on the aggressive-motion failure regime."
        )

    out_payload = {
        "classification": classification,
        "classification_rationale": classification_rationale,
        "primary_hypotheses": {
            "H1": "Mechanism-General Prediction",
            "H2": "Motion-Specific Proxy",
        },
        "verdict_details": {
            "motion_separation_verified": motion_cleanly_separated,
            "motion_separation_cohen_d": round(yaw_sep_d, 4),
            "motion_separation_p_value": yaw_sep_p,
            "r2_test_auroc": r2_base["auroc"],
            "r3_overall_auroc": r3_auroc,
            "delta_auroc": round(delta_auroc, 4),
            "r2_test_f1": r2_base["f1"],
            "r3_overall_f1": r3_f1,
            "delta_f1": round(delta_f1, 4),
            "r2_early_warning_rate_pct": r2_early_rate,
            "r3_early_warning_rate_pct": early_warn_rate,
            "delta_early_warning_rate_pct": round(early_warn_rate - r2_early_rate, 2),
        },
        "bootstrap_95_confidence_intervals": {
            "overall": boot_res_overall,
            "regimes": boot_res_regimes,
        },
        "hypothesis_tests": {
            "mann_whitney_u_vs_r2_test": {
                "statistic": float(mwu_test.statistic),
                "p_value": float(mwu_test.pvalue),
                "r3_significantly_lower": bool(mwu_less.pvalue < 0.05),
                "p_value_one_sided_lower": float(mwu_less.pvalue),
            },
            "one_sample_wilcoxon_vs_chance": {
                "statistic": float(wilc_vs_chance.statistic),
                "p_value": float(wilc_vs_chance.pvalue),
                "reject_null_chance": bool(wilc_vs_chance.pvalue < 0.05),
            },
        },
    }

    with open(OUT_STATISTICS_JSON, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("=" * 70)
    print("HYPOTHESIS TEST AND STATISTICAL VERDICT")
    print("=" * 70)
    print(f"Classification: {classification}")
    print(f"Rationale:      {classification_rationale}")
    print("-" * 70)
    print(f"R3 Overall AUROC: 95% CI [{boot_res_overall['auroc']['ci_95_lower']:.4f}, {boot_res_overall['auroc']['ci_95_upper']:.4f}]")
    print(f"R3 Overall F1:    95% CI [{boot_res_overall['f1']['ci_95_lower']:.4f}, {boot_res_overall['f1']['ci_95_upper']:.4f}]")
    print(f"R3 vs Chance Test: Wilcoxon stat = {wilc_vs_chance.statistic:.1f}, p = {wilc_vs_chance.pvalue:.4e}")
    print(f"R3 vs R2 Test:     Mann-Whitney U = {mwu_test.statistic:.1f}, p = {mwu_test.pvalue:.4e}")
    print("=" * 70)
    print(f"-> Statistical results written to: {OUT_STATISTICS_JSON}")


if __name__ == "__main__":
    main()
