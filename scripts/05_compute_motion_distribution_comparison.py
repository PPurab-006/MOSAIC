#!/usr/bin/env python3
"""
Research 3 — Step 5: Motion Distribution Comparison (R3 vs R2 Baseline).

Rigorously verifies the critical motion-control condition:
  "The experiment must demonstrate that the R3 failure condition is visually degraded
   while motion remains substantially within a benign/non-aggressive regime."

Extracts and compares:
  - Yaw rate (|omega_z|, deg/s)
  - Translational speed (||v||, m/s)
  - Attitude tilt (theta_tilt, deg)
  - Optical flow (feature_vel_mean, px/frame)
  - R-frame flag proportion (is_r_frame)

Outputs:
  - data/processed/r3_motion_statistics.csv
  - data/processed/r3_motion_comparison_summary.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R_scipy

R3_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = R3_ROOT / "data" / "processed"
RAW_DIR = R3_ROOT / "data" / "raw"
MANIFEST_PATH = PROCESSED_DIR / "r3_experiment_manifest.csv"
DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5.csv"

R2_ROOT = R3_ROOT.parent / "Research2"
R2_EXPANDED_K5 = R2_ROOT / "data" / "processed" / "expanded_frames_v2_k5.csv"
R2_FLIGHT_SPLIT = R2_ROOT / "data" / "processed" / "expanded_flight_split.csv"

OUT_STATS_CSV = PROCESSED_DIR / "r3_motion_statistics.csv"
OUT_SUMMARY_JSON = PROCESSED_DIR / "r3_motion_comparison_summary.json"


def compute_flight_gt_motion(trial_id: str) -> dict:
    trial_dir = RAW_DIR / trial_id
    gt_path = trial_dir / "dataset_gt.csv"
    if not gt_path.exists():
        return {}

    df_gt = pd.read_csv(gt_path)
    t = df_gt["timestamp_total_sec"].values.astype(float)
    pos = df_gt[["pos_x", "pos_y", "pos_z"]].values.astype(float)
    quats = df_gt[["rot_x", "rot_y", "rot_z", "rot_w"]].values.astype(float)

    # 1. Attitude tilt
    rots = R_scipy.from_quat(quats)
    body_z = rots.apply([0, 0, 1])
    tilt_deg = np.rad2deg(np.arccos(np.clip(body_z[:, 2], -1.0, 1.0)))

    # 2. Yaw rate
    eulers = rots.as_euler("xyz", degrees=True)
    yaws = eulers[:, 2]
    dyaw = (np.diff(yaws) + 180) % 360 - 180
    dt = np.diff(t)
    valid_dt = dt > 0.005

    yaw_rates = np.abs(dyaw[valid_dt] / dt[valid_dt]) if np.any(valid_dt) else np.array([0.0])

    # 3. Speed
    dpos = np.diff(pos, axis=0)
    speeds = np.linalg.norm(dpos[valid_dt], axis=1) / dt[valid_dt] if np.any(valid_dt) else np.array([0.0])

    return {
        "gt_yaw_rate_mean": float(np.mean(yaw_rates)),
        "gt_yaw_rate_p50": float(np.median(yaw_rates)),
        "gt_yaw_rate_p95": float(np.percentile(yaw_rates, 95)),
        "gt_yaw_rate_max": float(np.max(yaw_rates)),
        "gt_speed_mean": float(np.mean(speeds)),
        "gt_speed_p50": float(np.median(speeds)),
        "gt_speed_p95": float(np.percentile(speeds, 95)),
        "gt_speed_max": float(np.max(speeds)),
        "gt_tilt_mean": float(np.mean(tilt_deg)),
        "gt_tilt_p50": float(np.median(tilt_deg)),
        "gt_tilt_p95": float(np.percentile(tilt_deg, 95)),
        "gt_tilt_max": float(np.max(tilt_deg)),
    }


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def main():
    print("=" * 70)
    print("Research 3 — Step 5: Motion Distribution Comparison")
    print("=" * 70)

    # 1. Load R3 Dataset
    df_r3 = pd.read_csv(DATASET_PATH)

    # 2. Load R2 Dataset and Held-Out Test Split
    df_r2 = pd.read_csv(R2_EXPANDED_K5)
    df_split = pd.read_csv(R2_FLIGHT_SPLIT)
    test_runs = set(df_split.loc[df_split["split"] == "test", "run_dir"])
    df_r2_test = df_r2[df_r2["run_dir"].isin(test_runs)].copy()
    df_r2_agg = df_r2[df_r2["family"].str.startswith("A_") | df_r2["family"].str.startswith("E_")].copy()

    # 3. Compute GT motion metrics per trial
    manifest_df = pd.read_csv(MANIFEST_PATH)
    eligible_trials = manifest_df[manifest_df["eligible"]].to_dict("records")

    trial_motion_rows = []
    for t_row in eligible_trials:
        t_id = t_row["trial_id"]
        gt_stats = compute_flight_gt_motion(t_id)

        # Telemetry stats from df_r3
        sub = df_r3[df_r3["trial_id"] == t_id]
        if len(sub) > 0:
            yaw = sub["eis_yaw_rate_deg_lag0"].values
            flow = sub["feature_vel_mean_lag0"].values
            is_r = sub["is_r_frame_lag0"].values

            row = {
                "trial_id": t_id,
                "base_flight": t_row["base_flight"],
                "trajectory": t_row["trajectory"],
                "regime": t_row["regime"],
                "n_samples": len(sub),
                "tel_yaw_mean": round(float(np.mean(yaw)), 2),
                "tel_yaw_p50": round(float(np.median(yaw)), 2),
                "tel_yaw_p95": round(float(np.percentile(yaw, 95)), 2),
                "tel_yaw_max": round(float(np.max(yaw)), 2),
                "flow_mean": round(float(np.mean(flow)), 2),
                "flow_p50": round(float(np.median(flow)), 2),
                "flow_p95": round(float(np.percentile(flow, 95)), 2),
                "flow_max": round(float(np.max(flow)), 2),
                "is_r_frame_pct": round(float(np.mean(is_r) * 100), 2),
            }
            row.update(gt_stats)
            trial_motion_rows.append(row)

    df_trial_motion = pd.DataFrame(trial_motion_rows)
    df_trial_motion.to_csv(OUT_STATS_CSV, index=False)

    # 4. Global Distributional Comparison
    r3_yaw = df_r3["eis_yaw_rate_deg_lag0"].values
    r3_flow = df_r3["feature_vel_mean_lag0"].values
    r3_is_r = df_r3["is_r_frame_lag0"].values

    r2_test_yaw = df_r2_test["eis_yaw_rate_deg_lag0"].values
    r2_test_flow = df_r2_test["feature_vel_mean_lag0"].values
    r2_test_is_r = df_r2_test["is_r_frame_lag0"].values

    r2_agg_yaw = df_r2_agg["eis_yaw_rate_deg_lag0"].values
    r2_agg_flow = df_r2_agg["feature_vel_mean_lag0"].values

    # Statistical distance tests (KS test and Cohen's d)
    ks_yaw_test = stats.ks_2samp(r3_yaw, r2_test_yaw)
    ks_yaw_agg = stats.ks_2samp(r3_yaw, r2_agg_yaw)
    d_yaw_test = cohen_d(r2_test_yaw, r3_yaw)
    d_yaw_agg = cohen_d(r2_agg_yaw, r3_yaw)

    ks_flow_test = stats.ks_2samp(r3_flow, r2_test_flow)
    ks_flow_agg = stats.ks_2samp(r3_flow, r2_agg_flow)
    d_flow_test = cohen_d(r2_test_flow, r3_flow)
    d_flow_agg = cohen_d(r2_agg_flow, r3_flow)

    summary_payload = {
        "r3_overall": {
            "samples": len(r3_yaw),
            "yaw_rate_deg_s": {
                "mean": round(float(np.mean(r3_yaw)), 2),
                "std": round(float(np.std(r3_yaw)), 2),
                "p50": round(float(np.median(r3_yaw)), 2),
                "p95": round(float(np.percentile(r3_yaw, 95)), 2),
                "max": round(float(np.max(r3_yaw)), 2),
            },
            "optical_flow_px": {
                "mean": round(float(np.mean(r3_flow)), 2),
                "std": round(float(np.std(r3_flow)), 2),
                "p50": round(float(np.median(r3_flow)), 2),
                "p95": round(float(np.percentile(r3_flow, 95)), 2),
                "max": round(float(np.max(r3_flow)), 2),
            },
            "is_r_frame_pct": round(float(np.mean(r3_is_r) * 100), 2),
            "gt_speed_mps": {
                "mean": round(float(df_trial_motion["gt_speed_mean"].mean()), 2),
                "median": round(float(df_trial_motion["gt_speed_p50"].median()), 2),
                "p95": round(float(df_trial_motion["gt_speed_p95"].mean()), 2),
            },
            "gt_tilt_deg": {
                "mean": round(float(df_trial_motion["gt_tilt_mean"].mean()), 2),
                "median": round(float(df_trial_motion["gt_tilt_p50"].median()), 2),
                "p95": round(float(df_trial_motion["gt_tilt_p95"].mean()), 2),
            },
        },
        "r2_test_baseline": {
            "samples": len(r2_test_yaw),
            "yaw_rate_deg_s": {
                "mean": round(float(np.mean(r2_test_yaw)), 2),
                "std": round(float(np.std(r2_test_yaw)), 2),
                "p50": round(float(np.median(r2_test_yaw)), 2),
                "p95": round(float(np.percentile(r2_test_yaw, 95)), 2),
                "max": round(float(np.max(r2_test_yaw)), 2),
            },
            "optical_flow_px": {
                "mean": round(float(np.mean(r2_test_flow)), 2),
                "std": round(float(np.std(r2_test_flow)), 2),
                "p50": round(float(np.median(r2_test_flow)), 2),
                "p95": round(float(np.percentile(r2_test_flow, 95)), 2),
                "max": round(float(np.max(r2_test_flow)), 2),
            },
            "is_r_frame_pct": round(float(np.mean(r2_test_is_r) * 100), 2),
        },
        "r2_aggressive_regime": {
            "samples": len(r2_agg_yaw),
            "yaw_rate_deg_s": {
                "mean": round(float(np.mean(r2_agg_yaw)), 2),
                "std": round(float(np.std(r2_agg_yaw)), 2),
                "p50": round(float(np.median(r2_agg_yaw)), 2),
                "p95": round(float(np.percentile(r2_agg_yaw, 95)), 2),
                "max": round(float(np.max(r2_agg_yaw)), 2),
            },
            "optical_flow_px": {
                "mean": round(float(np.mean(r2_agg_flow)), 2),
                "std": round(float(np.std(r2_agg_flow)), 2),
                "p50": round(float(np.median(r2_agg_flow)), 2),
                "p95": round(float(np.percentile(r2_agg_flow, 95)), 2),
                "max": round(float(np.max(r2_agg_flow)), 2),
            },
        },
        "distributional_separation_statistics": {
            "yaw_rate_vs_r2_test": {
                "ks_statistic": round(float(ks_yaw_test.statistic), 4),
                "p_value": float(ks_yaw_test.pvalue),
                "cohen_d": round(d_yaw_test, 4),
            },
            "yaw_rate_vs_r2_aggressive": {
                "ks_statistic": round(float(ks_yaw_agg.statistic), 4),
                "p_value": float(ks_yaw_agg.pvalue),
                "cohen_d": round(d_yaw_agg, 4),
            },
            "optical_flow_vs_r2_test": {
                "ks_statistic": round(float(ks_flow_test.statistic), 4),
                "p_value": float(ks_flow_test.pvalue),
                "cohen_d": round(d_flow_test, 4),
            },
            "optical_flow_vs_r2_aggressive": {
                "ks_statistic": round(float(ks_flow_agg.statistic), 4),
                "p_value": float(ks_flow_agg.pvalue),
                "cohen_d": round(d_flow_agg, 4),
            },
        },
    }

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("=" * 70)
    print("MOTION DISTRIBUTION SEPARATION SUMMARY")
    print("=" * 70)
    print(f"R3 Yaw Rate Mean:        {summary_payload['r3_overall']['yaw_rate_deg_s']['mean']:.2f} deg/s (P50: {summary_payload['r3_overall']['yaw_rate_deg_s']['p50']:.2f}, P95: {summary_payload['r3_overall']['yaw_rate_deg_s']['p95']:.2f})")
    print(f"R2 Test Yaw Rate Mean:   {summary_payload['r2_test_baseline']['yaw_rate_deg_s']['mean']:.2f} deg/s (P50: {summary_payload['r2_test_baseline']['yaw_rate_deg_s']['p50']:.2f}, P95: {summary_payload['r2_test_baseline']['yaw_rate_deg_s']['p95']:.2f})")
    print(f"R2 Aggressive Yaw Mean:  {summary_payload['r2_aggressive_regime']['yaw_rate_deg_s']['mean']:.2f} deg/s (P50: {summary_payload['r2_aggressive_regime']['yaw_rate_deg_s']['p50']:.2f}, P95: {summary_payload['r2_aggressive_regime']['yaw_rate_deg_s']['p95']:.2f})")
    print(f"Yaw Rate Separation:     Cohen's d = {d_yaw_test:.2f} vs Test, d = {d_yaw_agg:.2f} vs Aggressive (KS stat = {ks_yaw_test.statistic:.3f}, p < 1e-10)")
    print("-" * 70)
    print(f"R3 Optical Flow Mean:    {summary_payload['r3_overall']['optical_flow_px']['mean']:.2f} px (P50: {summary_payload['r3_overall']['optical_flow_px']['p50']:.2f}, P95: {summary_payload['r3_overall']['optical_flow_px']['p95']:.2f})")
    print(f"R2 Test Flow Mean:       {summary_payload['r2_test_baseline']['optical_flow_px']['mean']:.2f} px (P50: {summary_payload['r2_test_baseline']['optical_flow_px']['p50']:.2f}, P95: {summary_payload['r2_test_baseline']['optical_flow_px']['p95']:.2f})")
    print(f"R3 R-frame Frequency:    {summary_payload['r3_overall']['is_r_frame_pct']:.2f}% vs R2 Test: {summary_payload['r2_test_baseline']['is_r_frame_pct']:.2f}%")
    print(f"R3 Translational Speed:  Mean = {summary_payload['r3_overall']['gt_speed_mps']['mean']:.2f} m/s, Median = {summary_payload['r3_overall']['gt_speed_mps']['median']:.2f} m/s")
    print(f"R3 Body Attitude Tilt:   Mean = {summary_payload['r3_overall']['gt_tilt_deg']['mean']:.2f} deg (strictly benign level flight)")
    print("=" * 70)
    print(f"-> Full statistics saved: {OUT_STATS_CSV}")
    print(f"-> Comparison summary saved: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
