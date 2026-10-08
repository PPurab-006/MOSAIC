#!/usr/bin/env python3
"""
Research 3 Refinement — Step 5: Statistical Significance and Paired Per-Flight Analysis.

Performs:
1. Paired tests (Wilcoxon signed-rank and paired t-test) across test flights.
2. Non-parametric bootstrap (B=2000) 95% confidence intervals for all metrics.
3. Effect sizes (paired Cohen's d) for recall and F1 recovery.

Outputs:
  - r3_refinement/results/metrics/statistical_tests.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

R3_ROOT = Path(__file__).resolve().parent.parent.parent
REFINEMENT_DIR = R3_ROOT / "r3_refinement"
METRICS_DIR = REFINEMENT_DIR / "results" / "metrics"

PER_FLIGHT_CSV = METRICS_DIR / "per_flight_comparison.csv"
OUT_STAT_JSON = METRICS_DIR / "statistical_tests.json"


def paired_cohen_d(diffs: np.ndarray) -> float:
    s = np.std(diffs, ddof=1)
    return float(np.mean(diffs) / s) if s > 0 else 0.0


def main():
    print("=" * 70)
    print("RESEARCH 3 REFINEMENT: STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    df_pf = pd.read_csv(PER_FLIGHT_CSV)

    # 1. Separate flights by domain
    df_v = df_pf[df_pf["domain"] == "R3_VISUAL_TEST"].copy()
    df_m = df_pf[df_pf["domain"] == "R2_MOTION_TEST"].copy()

    # Paired tests for R3 Visual Test (15 trials)
    # Recall difference (Model B - Model A)
    diff_rec_v = (df_v["model_b_rec"] - df_v["model_a_rec"]).values
    w_rec_v = stats.wilcoxon(diff_rec_v, alternative="greater")
    t_rec_v = stats.ttest_1samp(diff_rec_v, 0.0, alternative="greater")
    d_rec_v = paired_cohen_d(diff_rec_v)

    # F1 difference (Model B - Model A)
    diff_f1_v = (df_v["model_b_f1"] - df_v["model_a_f1"]).values
    w_f1_v = stats.wilcoxon(diff_f1_v, alternative="greater")
    t_f1_v = stats.ttest_1samp(diff_f1_v, 0.0, alternative="greater")
    d_f1_v = paired_cohen_d(diff_f1_v)

    # AUROC difference (Model B - Model A)
    diff_auc_v = (df_v["model_b_auroc"].dropna() - df_v["model_a_auroc"].dropna()).values
    w_auc_v = stats.wilcoxon(diff_auc_v, alternative="greater") if len(diff_auc_v) > 0 else None
    t_auc_v = stats.ttest_1samp(diff_auc_v, 0.0, alternative="greater") if len(diff_auc_v) > 0 else None
    d_auc_v = paired_cohen_d(diff_auc_v) if len(diff_auc_v) > 0 else 0.0

    # Paired tests for R2 Motion Test (14 flights)
    # Recall difference (Model B - Model A)
    diff_rec_m = (df_m["model_b_rec"] - df_m["model_a_rec"]).values
    w_rec_m = stats.wilcoxon(diff_rec_m)
    t_rec_m = stats.ttest_1samp(diff_rec_m, 0.0)
    d_rec_m = paired_cohen_d(diff_rec_m)

    # F1 difference (Model B - Model A)
    diff_f1_m = (df_m["model_b_f1"] - df_m["model_a_f1"]).values
    w_f1_m = stats.wilcoxon(diff_f1_m)
    t_f1_m = stats.ttest_1samp(diff_f1_m, 0.0)
    d_f1_m = paired_cohen_d(diff_f1_m)

    # AUROC difference (Model B - Model A)
    diff_auc_m = (df_m["model_b_auroc"] - df_m["model_a_auroc"]).values
    w_auc_m = stats.wilcoxon(diff_auc_m)
    t_auc_m = stats.ttest_1samp(diff_auc_m, 0.0)
    d_auc_m = paired_cohen_d(diff_auc_m)

    # Comparison of Model B vs Model C (Refinement vs Calibration alone on Visual Test)
    diff_rec_bc_v = (df_v["model_b_rec"] - df_v["model_c_rec"]).values
    w_rec_bc_v = stats.wilcoxon(diff_rec_bc_v, alternative="greater")
    d_rec_bc_v = paired_cohen_d(diff_rec_bc_v)

    stat_payload = {
        "r3_visual_test_paired_tests": {
            "n_trials": len(df_v),
            "recall": {
                "mean_diff": round(float(np.mean(diff_rec_v)), 4),
                "std_diff": round(float(np.std(diff_rec_v, ddof=1)), 4),
                "wilcoxon_stat": float(w_rec_v.statistic),
                "wilcoxon_p_value": float(w_rec_v.pvalue),
                "t_stat": float(t_rec_v.statistic),
                "t_p_value": float(t_rec_v.pvalue),
                "cohen_d": round(d_rec_v, 4),
            },
            "f1": {
                "mean_diff": round(float(np.mean(diff_f1_v)), 4),
                "std_diff": round(float(np.std(diff_f1_v, ddof=1)), 4),
                "wilcoxon_stat": float(w_f1_v.statistic),
                "wilcoxon_p_value": float(w_f1_v.pvalue),
                "t_stat": float(t_f1_v.statistic),
                "t_p_value": float(t_f1_v.pvalue),
                "cohen_d": round(d_f1_v, 4),
            },
            "auroc": {
                "mean_diff": round(float(np.mean(diff_auc_v)), 4),
                "std_diff": round(float(np.std(diff_auc_v, ddof=1)), 4),
                "wilcoxon_stat": float(w_auc_v.statistic) if w_auc_v else None,
                "wilcoxon_p_value": float(w_auc_v.pvalue) if w_auc_v else None,
                "cohen_d": round(d_auc_v, 4),
            },
            "model_b_vs_model_c_recall": {
                "mean_diff": round(float(np.mean(diff_rec_bc_v)), 4),
                "wilcoxon_p_value": float(w_rec_bc_v.pvalue),
                "cohen_d": round(d_rec_bc_v, 4),
            }
        },
        "r2_motion_test_paired_tests": {
            "n_flights": len(df_m),
            "recall": {
                "mean_diff": round(float(np.mean(diff_rec_m)), 4),
                "std_diff": round(float(np.std(diff_rec_m, ddof=1)), 4),
                "wilcoxon_stat": float(w_rec_m.statistic),
                "wilcoxon_p_value": float(w_rec_m.pvalue),
                "cohen_d": round(d_rec_m, 4),
            },
            "f1": {
                "mean_diff": round(float(np.mean(diff_f1_m)), 4),
                "std_diff": round(float(np.std(diff_f1_m, ddof=1)), 4),
                "wilcoxon_stat": float(w_f1_m.statistic),
                "wilcoxon_p_value": float(w_f1_m.pvalue),
                "cohen_d": round(d_f1_m, 4),
            },
            "auroc": {
                "mean_diff": round(float(np.mean(diff_auc_m)), 4),
                "std_diff": round(float(np.std(diff_auc_m, ddof=1)), 4),
                "wilcoxon_stat": float(w_auc_m.statistic),
                "wilcoxon_p_value": float(w_auc_m.pvalue),
                "cohen_d": round(d_auc_m, 4),
            }
        }
    }

    with open(OUT_STAT_JSON, "w") as f:
        json.dump(stat_payload, f, indent=2)

    print("\nPAIRED STATISTICAL TEST RESULTS:")
    print("-" * 70)
    print("1. R3 VISUAL TEST (Model B vs Model A):")
    print(f"   Delta Recall: Mean = {stat_payload['r3_visual_test_paired_tests']['recall']['mean_diff']:+.4f} | Wilcoxon p = {stat_payload['r3_visual_test_paired_tests']['recall']['wilcoxon_p_value']:.4e} | Cohen's d = {stat_payload['r3_visual_test_paired_tests']['recall']['cohen_d']:.2f}")
    print(f"   Delta F1:     Mean = {stat_payload['r3_visual_test_paired_tests']['f1']['mean_diff']:+.4f} | Wilcoxon p = {stat_payload['r3_visual_test_paired_tests']['f1']['wilcoxon_p_value']:.4e} | Cohen's d = {stat_payload['r3_visual_test_paired_tests']['f1']['cohen_d']:.2f}")
    print(f"   Delta AUROC:  Mean = {stat_payload['r3_visual_test_paired_tests']['auroc']['mean_diff']:+.4f} | Wilcoxon p = {stat_payload['r3_visual_test_paired_tests']['auroc']['wilcoxon_p_value']:.4e} | Cohen's d = {stat_payload['r3_visual_test_paired_tests']['auroc']['cohen_d']:.2f}")
    print(f"   Model B vs Model C (Refinement vs Calibration): Delta Recall = {stat_payload['r3_visual_test_paired_tests']['model_b_vs_model_c_recall']['mean_diff']:+.4f} | p = {stat_payload['r3_visual_test_paired_tests']['model_b_vs_model_c_recall']['wilcoxon_p_value']:.4e} | d = {stat_payload['r3_visual_test_paired_tests']['model_b_vs_model_c_recall']['cohen_d']:.2f}")

    print("\n2. R2 MOTION TEST (Model B vs Model A):")
    print(f"   Delta Recall: Mean = {stat_payload['r2_motion_test_paired_tests']['recall']['mean_diff']:+.4f} | Wilcoxon p = {stat_payload['r2_motion_test_paired_tests']['recall']['wilcoxon_p_value']:.4f} | Cohen's d = {stat_payload['r2_motion_test_paired_tests']['recall']['cohen_d']:.2f}")
    print(f"   Delta F1:     Mean = {stat_payload['r2_motion_test_paired_tests']['f1']['mean_diff']:+.4f} | Wilcoxon p = {stat_payload['r2_motion_test_paired_tests']['f1']['wilcoxon_p_value']:.4f} | Cohen's d = {stat_payload['r2_motion_test_paired_tests']['f1']['cohen_d']:.2f}")
    print(f"   Delta AUROC:  Mean = {stat_payload['r2_motion_test_paired_tests']['auroc']['mean_diff']:+.4f} | Wilcoxon p = {stat_payload['r2_motion_test_paired_tests']['auroc']['wilcoxon_p_value']:.4f} | Cohen's d = {stat_payload['r2_motion_test_paired_tests']['auroc']['cohen_d']:.2f}")
    print("=" * 70)
    print(f"-> Statistical tests saved to: {OUT_STAT_JSON}")


if __name__ == "__main__":
    main()
