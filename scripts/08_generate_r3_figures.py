#!/usr/bin/env python3
"""
Research 3 — Step 8: Publication-Quality Figures Generation.

Produces the 4 required figures:
  1. figures/fig1_r2_vs_r3_performance.png
     - Direct comparison of R2 Test Baseline vs R3 Transfer with 95% bootstrap CIs.
  2. figures/fig2_performance_vs_degradation_level.png
     - Monocular VO tracking degradation and predictor performance vs visual degradation severity.
  3. figures/fig3_motion_distribution_comparison.png
     - Distributional separation of vehicle motion (yaw rate, flow, velocity, tilt) between R2 and R3.
  4. figures/fig4_early_warning_behavior.png
     - Temporal episode dynamics, lead time distribution, and pre-failure warning classification.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling for scientific publication
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

import shutil

R3_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = R3_ROOT / "data" / "processed"
FIGURES_DIR = R3_ROOT / "figures"
ASSETS_DIR = R3_ROOT / "assets" / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5.csv"
METRICS_JSON_PATH = PROCESSED_DIR / "r3_frozen_r2_evaluation_metrics.json"
PER_FLIGHT_CSV_PATH = PROCESSED_DIR / "r3_per_flight_metrics.csv"
MOTION_SUMMARY_PATH = PROCESSED_DIR / "r3_motion_comparison_summary.json"
WARNINGS_JSON_PATH = PROCESSED_DIR / "r3_lead_time_early_warnings.json"
STATISTICS_JSON_PATH = PROCESSED_DIR / "r3_statistical_comparison.json"
EPISODES_CSV_PATH = PROCESSED_DIR / "r3_episode_details.csv"

R2_EXPANDED_K5 = Path("/home/purab/Purab/Projects/Research2/data/processed/expanded_frames_v2_k5.csv")
R2_FLIGHT_SPLIT = Path("/home/purab/Purab/Projects/Research2/data/processed/expanded_flight_split.csv")


def plot_fig1_r2_vs_r3(metrics_data, stats_data):
    """Figure 1: R2 vs R3 Performance Comparison."""
    r2_base = metrics_data["r2_test_baseline"]
    r3_over = metrics_data["r3_overall_transfer_metrics"]
    ci = stats_data["bootstrap_95_confidence_intervals"]["overall"]

    metrics = ["AUROC", "AUPRC", "F1 Score", "Precision", "Recall", "Early Warning Rate"]
    r2_vals = [
        r2_base["auroc"],
        r2_base["auprc"],
        r2_base["f1"],
        r2_base["precision"],
        r2_base["recall"],
        r2_base["early_warning_rate_pct"] / 100.0,
    ]
    r3_vals = [
        r3_over["auroc"],
        r3_over["auprc"],
        r3_over["f1"],
        r3_over["precision"],
        r3_over["recall"],
        stats_data["verdict_details"]["r3_early_warning_rate_pct"] / 100.0,
    ]

    # Analytical 95% binomial confidence interval for episode-level early warning rate (N=5,964)
    ew_rate = r3_vals[5]
    n_episodes = 5964
    binom_err = 1.96 * np.sqrt(ew_rate * (1.0 - ew_rate) / n_episodes)

    r3_err_low = [
        r3_vals[0] - ci["auroc"]["ci_95_lower"],
        r3_vals[1] - ci["auprc"]["ci_95_lower"],
        r3_vals[2] - ci["f1"]["ci_95_lower"],
        r3_vals[3] - ci["precision"]["ci_95_lower"],
        r3_vals[4] - ci["recall"]["ci_95_lower"],
        binom_err,
    ]
    r3_err_high = [
        ci["auroc"]["ci_95_upper"] - r3_vals[0],
        ci["auprc"]["ci_95_upper"] - r3_vals[1],
        ci["f1"]["ci_95_upper"] - r3_vals[2],
        ci["precision"]["ci_95_upper"] - r3_vals[3],
        ci["recall"]["ci_95_upper"] - r3_vals[4],
        binom_err,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    rects1 = ax.bar(x - width / 2, r2_vals, width, label="R2 Motion-Induced Baseline (Held-Out Test)", color="#2b5c8f", alpha=0.9, edgecolor="black")
    rects2 = ax.bar(x + width / 2, r3_vals, width, yerr=[r3_err_low, r3_err_high], capsize=4,
                    label="R3 Visual-Degradation Transfer (Frozen Predictor)", color="#e07a5f", alpha=0.9, edgecolor="black")

    ax.set_ylabel("Score / Rate [0, 1]")
    ax.set_title("Figure 1: Cross-Mechanism Generalization — R2 Motion Baseline vs R3 Visual Degradation", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=12, ha="right")
    ax.set_ylim(0, 1.22)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance Level (AUROC=0.5)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.text(0.02, 0.94, "*Error bars: 95% bootstrap CIs (frame metrics); 95% binomial CI (episodes, N=5,964)",
            transform=ax.transAxes, fontsize=8.5, color="#555555", style="italic")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_r2_vs_r3_performance.png"
    plt.savefig(out_path)
    shutil.copy(out_path, ASSETS_DIR / out_path.name)
    plt.close()
    print(f"-> Saved: {out_path} and synced to {ASSETS_DIR / out_path.name}")


def plot_fig2_degradation_severity(metrics_data):
    """Figure 2: Performance vs Degradation Severity."""
    regimes = ["NOMINAL", "LOW", "MEDIUM", "HIGH"]
    reg_data = metrics_data["regime_breakdown"]

    alpha_labels = ["Nominal\n(α=0.000)", "Low\n(α=0.920)", "Medium\n(α=0.960)", "High\n(α=0.985)"]
    fail_rates = [reg_data[r]["positive_rate_pct"] for r in regimes]
    f1s = [reg_data[r]["f1"] for r in regimes]
    recalls = [reg_data[r]["recall"] for r in regimes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Panel A: Failure Rate vs Severity
    ax1.plot(alpha_labels, fail_rates, marker="o", color="#d62828", linewidth=2.5, markersize=8, label="VO Failure Rate (K=5)")
    ax1.axhline(34.62, color="blue", linestyle="--", alpha=0.7, label="R2 Test Base Rate (34.6%)")
    ax1.set_ylabel("Positive Failure Label Rate (%)")
    ax1.set_title("(A) VO Failure Rate vs Degradation Severity")
    ax1.set_ylim(-2, 114)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    for x_idx, y_val in enumerate(fail_rates):
        ax1.annotate(f"{y_val:.1f}%", xy=(x_idx, y_val), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Predictor Metrics vs Severity
    # AUROC is well-defined for Nominal, Low, Medium; undefined for High (100% positive, 0 negative)
    auroc_x = [0, 1, 2]
    auroc_y = [reg_data["NOMINAL"]["auroc"], reg_data["LOW"]["auroc"], reg_data["MEDIUM"]["auroc"]]
    ax2.plot(auroc_x, auroc_y, marker="s", color="#1d3557", linewidth=2.2, markersize=7, label="AUROC")
    ax2.plot(range(len(regimes)), f1s, marker="^", color="#2a9d8f", linewidth=2.2, markersize=7, label="F1 Score")
    ax2.plot(range(len(regimes)), recalls, marker="d", color="#e76f51", linewidth=2.2, markersize=7, label="Recall")

    # Highlight High AUROC undefined status
    ax2.plot([3], [0.5], marker="x", color="#1d3557", markersize=8, markeredgewidth=2)
    ax2.annotate("N/A*\n(100% Pos)", xy=(3, 0.5), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=8.5, color="#1d3557", fontweight="bold")

    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance Level (0.5)")
    ax2.set_xticks(range(len(regimes)))
    ax2.set_xticklabels(alpha_labels)
    ax2.set_ylabel("Score [0, 1]")
    ax2.set_title("(B) Frozen R2 Performance vs Degradation Severity")
    ax2.set_ylim(0.0, 1.12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", framealpha=0.9)

    for x_idx, y_val in zip(auroc_x, auroc_y):
        ax2.annotate(f"{y_val:.2f}", xy=(x_idx, y_val), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax2.text(0.38, 0.04, "*AUROC undefined at High (0 negative samples)",
             transform=ax2.transAxes, fontsize=8, color="#555555")

    plt.suptitle("Figure 2: Physical Degradation Response & Predictor Transfer Breakdown", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig2_performance_vs_degradation_level.png"
    plt.savefig(out_path)
    shutil.copy(out_path, ASSETS_DIR / out_path.name)
    plt.close()
    print(f"-> Saved: {out_path} and synced to {ASSETS_DIR / out_path.name}")


def plot_fig3_motion_distributions():
    """Figure 3: Motion Distribution Comparison between R2 and R3."""
    df_r3 = pd.read_csv(DATASET_PATH)
    df_r2 = pd.read_csv(R2_EXPANDED_K5)
    df_split = pd.read_csv(R2_FLIGHT_SPLIT)
    test_runs = set(df_split.loc[df_split["split"] == "test", "run_dir"])
    df_r2_test = df_r2[df_r2["run_dir"].isin(test_runs)].copy()
    df_r2_agg = df_r2[df_r2["family"].str.startswith("A_") | df_r2["family"].str.startswith("E_")].copy()

    r3_yaw = df_r3["eis_yaw_rate_deg_lag0"].values
    r2_test_yaw = df_r2_test["eis_yaw_rate_deg_lag0"].values
    r2_agg_yaw = df_r2_agg["eis_yaw_rate_deg_lag0"].values

    r3_flow = df_r3["feature_vel_mean_lag0"].values
    r2_test_flow = df_r2_test["feature_vel_mean_lag0"].values
    r2_agg_flow = df_r2_agg["feature_vel_mean_lag0"].values

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: Yaw Rate Distributions
    bins_yaw = np.linspace(0, 120, 60)
    ax1.hist(r3_yaw, bins=bins_yaw, density=True, alpha=0.6, color="#2a9d8f", label=f"R3 Benign Visual (Mean: {np.mean(r3_yaw):.1f}°/s)")
    ax1.hist(r2_test_yaw, bins=bins_yaw, density=True, alpha=0.5, color="#2b5c8f", label=f"R2 Test Baseline (Mean: {np.mean(r2_test_yaw):.1f}°/s)")
    ax1.hist(r2_agg_yaw, bins=bins_yaw, density=True, alpha=0.4, color="#e63946", label=f"R2 Aggressive Motion (Mean: {np.mean(r2_agg_yaw):.1f}°/s)")
    ax1.set_xlabel("Yaw Rate |ω_z| [deg/s]")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("(A) Vehicle Yaw Rate Separation")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Optical Flow Distributions
    bins_flow = np.linspace(0, 60, 60)
    ax2.hist(r3_flow, bins=bins_flow, density=True, alpha=0.6, color="#2a9d8f", label=f"R3 Benign Visual (Mean: {np.mean(r3_flow):.1f} px)")
    ax2.hist(r2_test_flow, bins=bins_flow, density=True, alpha=0.5, color="#2b5c8f", label=f"R2 Test Baseline (Mean: {np.mean(r2_test_flow):.1f} px)")
    ax2.hist(r2_agg_flow, bins=bins_flow, density=True, alpha=0.4, color="#e63946", label=f"R2 Aggressive Motion (Mean: {np.mean(r2_agg_flow):.1f} px)")
    ax2.set_xlabel("KLT Optical Flow Magnitude [px/frame]")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("(B) Optical Flow Velocity Separation")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Panel C: R3 Speed Distribution from motion stats CSV
    df_motion = pd.read_csv(PROCESSED_DIR / "r3_motion_statistics.csv")
    speeds = df_motion["gt_speed_mean"].values
    ax3.bar(range(len(speeds)), speeds, color="#457b9d", alpha=0.85, edgecolor="black")
    ax3.axhline(0.8, color="red", linestyle="--", label="Target Cruise Velocity (0.8 m/s)")
    ax3.set_xlabel("R3 Experimental Trials")
    ax3.set_ylabel("Mean Translational Speed [m/s]")
    ax3.set_title("(C) R3 Controlled Cruise Speed per Flight")
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel D: R3 Attitude Tilt Distribution
    tilts = df_motion["gt_tilt_mean"].values
    ax4.bar(range(len(tilts)), tilts, color="#e07a5f", alpha=0.85, edgecolor="black")
    ax4.axhline(45.0, color="darkred", linestyle=":", linewidth=1.8, label="Hard Safety Gate Tilt Limit (45°)")
    ax4.axhspan(0, 10, color="green", alpha=0.08, label="Benign Flight Envelope (<10°)")
    ax4.set_xlabel("R3 Experimental Trials")
    ax4.set_ylabel("Mean Attitude Tilt θ_tilt [deg]")
    ax4.set_title("(D) R3 Attitude Tilt per Flight (Benign < 5°)")
    ax4.set_ylim(0, 50)
    ax4.legend(loc="upper right", fontsize=8.5)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Figure 3: Kinematic Decoupling Verification — Vehicle Motion Comparison (R3 vs R2)", y=1.01)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_motion_distribution_comparison.png"
    plt.savefig(out_path)
    shutil.copy(out_path, ASSETS_DIR / out_path.name)
    plt.close()
    print(f"-> Saved: {out_path} and synced to {ASSETS_DIR / out_path.name}")


def plot_fig4_early_warnings(warnings_data):
    """Figure 4: Early-Warning Dynamics & Temporal Behavior."""
    df_ep = pd.read_csv(EPISODES_CSV_PATH)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Panel A: Outcome Proportions across all 5 regimes
    regimes = ["NOMINAL", "LOW", "TRANSITION", "MEDIUM", "HIGH"]
    reg_labels = ["Nominal\n(α=0.00)", "Low\n(α=0.92)", "Transition\n(0.0→0.98)", "Medium\n(α=0.96)", "High\n(α=0.985)"]
    early_pcts = [warnings_data["regime_breakdown"][r]["early_warning_rate_pct"] for r in regimes]
    post_pcts = [warnings_data["regime_breakdown"][r]["post_detection_rate_pct"] for r in regimes]
    miss_pcts = [warnings_data["regime_breakdown"][r]["missed_failure_rate_pct"] for r in regimes]

    x = np.arange(len(regimes))
    width = 0.55

    p1 = ax1.bar(x, early_pcts, width, label="True Early Warning", color="#2a9d8f", edgecolor="black")
    p2 = ax1.bar(x, post_pcts, width, bottom=early_pcts, label="Post-Failure Detection", color="#e9c46a", edgecolor="black")
    bottom_miss = [e + p for e, p in zip(early_pcts, post_pcts)]
    p3 = ax1.bar(x, miss_pcts, width, bottom=bottom_miss, label="Missed Failure", color="#e76f51", edgecolor="black")

    # Annotate early warning percentage on each bar
    for i, ew_val in enumerate(early_pcts):
        ax1.annotate(f"{ew_val:.1f}%", xy=(x[i], ew_val / 2),
                     ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Proportion of Episodes (%)")
    ax1.set_title("(A) Warning Timing Distribution by Degradation Regime")
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_labels)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Lead Time Distribution for Early Warnings
    early_leads = df_ep.loc[df_ep["classification"] == "EARLY_WARNING", "lead_time_s"].dropna().values
    if len(early_leads) > 0:
        ax2.hist(early_leads, bins=15, color="#2b5c8f", alpha=0.8, edgecolor="black", density=True)
        ax2.axvline(np.median(early_leads), color="red", linestyle="--", linewidth=2, label=f"R3 Median: {np.median(early_leads):.3f}s")
        ax2.axvline(0.165, color="orange", linestyle=":", linewidth=2, label="R2 Baseline Median: 0.165s")
        ax2.set_xlabel("Pre-Failure Lead Time Δt_lead [seconds]")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("(B) Lead-Time Distribution for True Early Warnings")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Figure 4: Pre-Failure Warning Dynamics Under Visual Observability Degradation", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig4_early_warning_behavior.png"
    plt.savefig(out_path)
    shutil.copy(out_path, ASSETS_DIR / out_path.name)
    plt.close()
    print(f"-> Saved: {out_path} and synced to {ASSETS_DIR / out_path.name}")


def main():
    print("=" * 70)
    print("Research 3 — Step 8: Generating Publication Figures")
    print("=" * 70)

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    with open(STATISTICS_JSON_PATH, "r", encoding="utf-8") as f:
        stats_data = json.load(f)
    with open(WARNINGS_JSON_PATH, "r", encoding="utf-8") as f:
        warnings_data = json.load(f)

    plot_fig1_r2_vs_r3(metrics_data, stats_data)
    plot_fig2_degradation_severity(metrics_data)
    plot_fig3_motion_distributions()
    plot_fig4_early_warnings(warnings_data)

    print("=" * 70)
    print(f"All figures successfully generated in: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
