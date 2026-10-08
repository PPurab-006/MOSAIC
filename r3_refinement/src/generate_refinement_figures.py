#!/usr/bin/env python3
"""
Research 3 Refinement — Step 7: Publication-Quality Figures Generation.

Outputs:
  - r3_refinement/results/figures/refinement_fig1_three_way_comparison.png
  - r3_refinement/results/figures/refinement_fig2_pr_roc_curves.png
  - r3_refinement/results/figures/refinement_fig3_per_flight_deltas.png
  - r3_refinement/results/figures/refinement_fig4_early_warning_comparison.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

R3_ROOT = Path(__file__).resolve().parent.parent.parent
REFINEMENT_DIR = R3_ROOT / "r3_refinement"
METRICS_DIR = REFINEMENT_DIR / "results" / "metrics"
FIG_DIR = REFINEMENT_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON = METRICS_DIR / "protected_evaluation_metrics.json"
PER_FLIGHT_CSV = METRICS_DIR / "per_flight_comparison.csv"
EPISODES_JSON = METRICS_DIR / "episode_early_warning_comparison.json"
ABLATION_JSON = METRICS_DIR / "ablation_results.json"

MANIFEST_PATH = REFINEMENT_DIR / "manifests" / "refinement_split_manifest.json"
LOCKED_THRESHOLDS_PATH = REFINEMENT_DIR / "models" / "locked_thresholds.json"
DATASET_R2 = Path("/home/purab/Purab/Projects/Research2/data/processed/expanded_frames_v2_k5.csv")
DATASET_R3 = R3_ROOT / "data" / "processed" / "r3_dataset_k5.csv"


def plot_fig1(metrics_data):
    """Figure 1: Three-way comparison across Visual Test, Motion Test, and Combined Test."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    domains = [
        ("protected_r3_visual_test", "Visual Degradation Test (15 Trials)", axes[0]),
        ("protected_r2_motion_test", "Motion-Induced Test (14 Flights)", axes[1]),
        ("combined_protected_cross_mechanism_test", "Combined Cross-Mechanism Test", axes[2]),
    ]

    metrics = ["Recall", "Precision", "F1 Score", "AUROC"]
    x = np.arange(len(metrics))
    width = 0.25

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    labels = ["Model A (Frozen R2)", "Model B (Refined MLP)", "Model C (Calibrated Frozen R2)"]

    for dom_key, title, ax in domains:
        d = metrics_data[dom_key]
        mA = [d["model_a_frozen_r2"]["recall"], d["model_a_frozen_r2"]["precision"], d["model_a_frozen_r2"]["f1"], d["model_a_frozen_r2"]["auroc"]]
        mB = [d["model_b_refined_mlp"]["recall"], d["model_b_refined_mlp"]["precision"], d["model_b_refined_mlp"]["f1"], d["model_b_refined_mlp"]["auroc"]]
        mC = [d["model_c_calibrated_frozen_r2"]["recall"], d["model_c_calibrated_frozen_r2"]["precision"], d["model_c_calibrated_frozen_r2"]["f1"], d["model_c_calibrated_frozen_r2"]["auroc"]]

        ax.bar(x - width, mA, width, label=labels[0], color=colors[0], alpha=0.9, edgecolor="black")
        ax.bar(x, mB, width, label=labels[1], color=colors[1], alpha=0.9, edgecolor="black")
        ax.bar(x + width, mC, width, label=labels[2], color=colors[2], alpha=0.9, edgecolor="black")

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

    axes[0].set_ylabel("Metric Score")
    axes[0].legend(loc="lower right")

    plt.tight_layout()
    out_path = FIG_DIR / "refinement_fig1_three_way_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"-> Saved: {out_path}")


def plot_fig2():
    """Figure 2: PR and ROC curves comparing Model A, Model B, and Model C."""
    import joblib, torch
    with open(MANIFEST_PATH, "r") as f:
        split_manifest = json.load(f)
    with open(LOCKED_THRESHOLDS_PATH, "r") as f:
        locked_th = json.load(f)
    with open(R3_ROOT / "models" / "frozen_r2_manifest.json") as f:
        feat_cols = json.load(f)["feature_specification"]["feature_names_ordered"]

    import torch.nn as nn
    class SmallMLP(nn.Module):
        def __init__(self, input_dim=15, hidden1=32, hidden2=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    scaler_r2 = joblib.load(locked_th["model_a_frozen_r2"]["scaler_path"])
    model_r2 = SmallMLP(15, 32, 16)
    model_r2.load_state_dict(torch.load(locked_th["model_a_frozen_r2"]["model_path"], map_location="cpu"))
    model_r2.eval()

    scaler_ref = joblib.load(locked_th["model_b_refined_mlp"]["scaler_path"])
    model_ref = SmallMLP(15, 32, 16)
    model_ref.load_state_dict(torch.load(locked_th["model_b_refined_mlp"]["model_path"], map_location="cpu"))
    model_ref.eval()

    # Load data
    df_r2 = pd.read_csv(DATASET_R2)
    df_r2_test = df_r2[df_r2["run_dir"].isin(split_manifest["r2_motion_mechanism"]["protected_test_flights"])].copy()
    df_r3 = pd.read_csv(DATASET_R3)
    df_r3_test = df_r3[df_r3["trial_id"].isin(split_manifest["r3_visual_mechanism"]["protected_test_trials"])].copy()

    y_v = df_r3_test["is_failure_within_next_k"].values.astype(int)
    X_v_raw = df_r3_test[feat_cols].values.astype(float)
    with torch.no_grad():
        p_v_frozen = torch.sigmoid(model_r2(torch.tensor(scaler_r2.transform(X_v_raw), dtype=torch.float32))).numpy()
        p_v_ref = torch.sigmoid(model_ref(torch.tensor(scaler_ref.transform(X_v_raw), dtype=torch.float32))).numpy()

    y_m = df_r2_test["is_failure_within_next_k"].values.astype(int)
    X_m_raw = df_r2_test[feat_cols].values.astype(float)
    with torch.no_grad():
        p_m_frozen = torch.sigmoid(model_r2(torch.tensor(scaler_r2.transform(X_m_raw), dtype=torch.float32))).numpy()
        p_m_ref = torch.sigmoid(model_ref(torch.tensor(scaler_ref.transform(X_m_raw), dtype=torch.float32))).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ROC Visual Test
    fpr_v_f, tpr_v_f, _ = roc_curve(y_v, p_v_frozen)
    fpr_v_r, tpr_v_r, _ = roc_curve(y_v, p_v_ref)
    axes[0, 0].plot(fpr_v_f, tpr_v_f, label="Model A (Frozen R2): AUROC=0.792", color="#2b5c8f", linewidth=2)
    axes[0, 0].plot(fpr_v_r, tpr_v_r, label="Model B (Refined MLP): AUROC=0.937", color="#d95f02", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 0].set_title("ROC Curve — Visual Degradation Test", fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # PR Visual Test
    prec_v_f, rec_v_f, _ = precision_recall_curve(y_v, p_v_frozen)
    prec_v_r, rec_v_r, _ = precision_recall_curve(y_v, p_v_ref)
    axes[0, 1].plot(rec_v_f, prec_v_f, label="Model A (Frozen R2): AUPRC=0.860", color="#2b5c8f", linewidth=2)
    axes[0, 1].plot(rec_v_r, prec_v_r, label="Model B (Refined MLP): AUPRC=0.966", color="#d95f02", linewidth=2)
    axes[0, 1].set_title("Precision-Recall — Visual Degradation Test", fontweight="bold")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # ROC Motion Test
    fpr_m_f, tpr_m_f, _ = roc_curve(y_m, p_m_frozen)
    fpr_m_r, tpr_m_r, _ = roc_curve(y_m, p_m_ref)
    axes[1, 0].plot(fpr_m_f, tpr_m_f, label="Model A (Frozen R2): AUROC=0.805", color="#2b5c8f", linewidth=2)
    axes[1, 0].plot(fpr_m_r, tpr_m_r, label="Model B (Refined MLP): AUROC=0.801", color="#d95f02", linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1, 0].set_title("ROC Curve — Motion-Induced Test", fontweight="bold")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # PR Motion Test
    prec_m_f, rec_m_f, _ = precision_recall_curve(y_m, p_m_frozen)
    prec_m_r, rec_m_r, _ = precision_recall_curve(y_m, p_m_ref)
    axes[1, 1].plot(rec_m_f, prec_m_f, label="Model A (Frozen R2): AUPRC=0.726", color="#2b5c8f", linewidth=2)
    axes[1, 1].plot(rec_m_r, prec_m_r, label="Model B (Refined MLP): AUPRC=0.721", color="#d95f02", linewidth=2)
    axes[1, 1].set_title("Precision-Recall — Motion-Induced Test", fontweight="bold")
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = FIG_DIR / "refinement_fig2_pr_roc_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"-> Saved: {out_path}")


def plot_fig3():
    """Figure 3: Per-flight paired delta distributions across test flights."""
    df_pf = pd.read_csv(PER_FLIGHT_CSV)
    df_v = df_pf[df_pf["domain"] == "R3_VISUAL_TEST"].copy()
    df_m = df_pf[df_pf["domain"] == "R2_MOTION_TEST"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Visual Test flight deltas
    x_v = np.arange(len(df_v))
    d_rec_v = (df_v["model_b_rec"] - df_v["model_a_rec"]) * 100
    d_f1_v = (df_v["model_b_f1"] - df_v["model_a_f1"]) * 100

    width = 0.35
    axes[0].bar(x_v - width/2, d_rec_v, width, label="Delta Recall (%)", color="#1b9e77", alpha=0.9, edgecolor="black")
    axes[0].bar(x_v + width/2, d_f1_v, width, label="Delta F1 (%)", color="#d95f02", alpha=0.9, edgecolor="black")
    axes[0].axhline(0, color="black", linestyle="-", linewidth=1)
    axes[0].set_title("Visual Degradation Test Trials (N=15)\nModel B vs Model A", fontweight="bold")
    axes[0].set_xlabel("Test Trial Index")
    axes[0].set_ylabel("Performance Delta (%)")
    axes[0].set_xticks(x_v)
    axes[0].set_xticklabels([f"T{i+1}" for i in range(len(df_v))])
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Motion Test flight deltas
    x_m = np.arange(len(df_m))
    d_rec_m = (df_m["model_b_rec"] - df_m["model_a_rec"]) * 100
    d_f1_m = (df_m["model_b_f1"] - df_m["model_a_f1"]) * 100

    axes[1].bar(x_m - width/2, d_rec_m, width, label="Delta Recall (%)", color="#1b9e77", alpha=0.9, edgecolor="black")
    axes[1].bar(x_m + width/2, d_f1_m, width, label="Delta F1 (%)", color="#d95f02", alpha=0.9, edgecolor="black")
    axes[1].axhline(0, color="black", linestyle="-", linewidth=1)
    axes[1].set_title("Motion-Induced Test Flights (N=14)\nModel B vs Model A", fontweight="bold")
    axes[1].set_xlabel("Test Flight Index")
    axes[1].set_ylabel("Performance Delta (%)")
    axes[1].set_xticks(x_m)
    axes[1].set_xticklabels([f"F{i+1}" for i in range(len(df_m))])
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = FIG_DIR / "refinement_fig3_per_flight_deltas.png"
    plt.savefig(out_path)
    plt.close()
    print(f"-> Saved: {out_path}")


def plot_fig4(episodes_data):
    """Figure 4: Episode-level early warning rate and missed failure comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    domains = [
        ("protected_r3_visual_test", "Visual Degradation Test", axes[0]),
        ("protected_r2_motion_test", "Motion-Induced Test", axes[1]),
    ]

    models = ["Model A (Frozen R2)", "Model B (Refined MLP)", "Model C (Calibrated Frozen)"]
    keys = ["model_a_frozen_r2", "model_b_refined_mlp", "model_c_calibrated_frozen_r2"]

    for dom_key, title, ax in domains:
        d = episodes_data[dom_key]
        early_rates = [d[k]["early_rate_pct"] for k in keys]
        missed_rates = [d[k]["missed_rate_pct"] for k in keys]
        post_rates = [d[k]["post_rate_pct"] for k in keys]

        x = np.arange(len(models))
        width = 0.55

        ax.bar(x, early_rates, width, label="Early Warning (< t_onset)", color="#2ca02c", alpha=0.85, edgecolor="black")
        ax.bar(x, post_rates, width, bottom=early_rates, label="Post-Onset Detection", color="#ff7f0e", alpha=0.85, edgecolor="black")
        bottom_miss = [e + p for e, p in zip(early_rates, post_rates)]
        ax.bar(x, missed_rates, width, bottom=bottom_miss, label="Missed Episode", color="#d62728", alpha=0.85, edgecolor="black")

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Model A\n(Frozen)", "Model B\n(Refined)", "Model C\n(Calibrated)"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Episode Outcome Distribution (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].legend(loc="lower left")

    plt.tight_layout()
    out_path = FIG_DIR / "refinement_fig4_early_warning_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"-> Saved: {out_path}")


def main():
    print("=" * 70)
    print("RESEARCH 3 REFINEMENT: GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    with open(METRICS_JSON) as f:
        m_data = json.load(f)
    with open(EPISODES_JSON) as f:
        e_data = json.load(f)

    plot_fig1(m_data)
    plot_fig2()
    plot_fig3()
    plot_fig4(e_data)

    print("=" * 70)
    print("All 4 refinement figures successfully generated in:", FIG_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
