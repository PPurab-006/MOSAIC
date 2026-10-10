#!/usr/bin/env python3
"""
Research 3 — Step 3: Feature Construction and Causal K=5 Label Generation.

Processes raw_vo.csv from each trial:
  1. Computes raw failure label: num_inliers_pose < 8
  2. Constructs forward-looking failure window target: is_failure_within_next_k for K=5
  3. Constructs backward-lagged features across lags {0, 1, 2, 3, 5} for:
     - eis_yaw_rate_deg
     - feature_vel_mean
     - is_r_frame
  4. Preserves causal boundaries (drops first 5 frames and last K=5 frames)
  5. Aggregates into data/processed/r3_dataset_k5.csv
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

R3_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = R3_ROOT / "data" / "raw"
PROCESSED_DIR = R3_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "r3_experiment_manifest.csv"
OUTPUT_DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5.csv"
OUTPUT_STRICT_DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5_strict.csv"

BASE_FEATURES = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame"]
FEATURE_LAGS = [0, 1, 2, 3, 5]
FEATURE_COLS = [f"{col}_lag{lag}" for col in BASE_FEATURES for lag in FEATURE_LAGS]

META_COLS = [
    "trial_id",
    "base_flight",
    "trajectory",
    "regime",
    "regime_type",
    "degradation_alpha",
    "frame_idx",
    "timestamp_total_sec",
    "num_inliers_pose",
    "is_failure_current_frame",
    "is_failure_within_next_k",
]

FINAL_COLS = META_COLS + FEATURE_COLS


def process_trial_features(trial_row: dict, k: int = 5, legacy_label: bool = False) -> pd.DataFrame:
    trial_id = trial_row["trial_id"]
    trial_dir = RAW_DIR / trial_id
    raw_vo_path = trial_dir / "raw_vo.csv"

    if not raw_vo_path.exists():
        raise FileNotFoundError(f"Missing raw_vo.csv for {trial_id}")

    df = pd.read_csv(raw_vo_path)
    n_raw = len(df)

    # 1. Failure definition
    df["is_failure"] = (df["num_inliers_pose"].astype(int) < 8).astype(int)
    df["is_failure_current_frame"] = df["is_failure"]

    # 2. Forward-looking failure window for next K frames
    # Legacy: includes current frame step=0 (range(0, k+1))
    # Strict: strictly future frames step in range(1, k+1)
    shift_steps = range(k + 1) if legacy_label else range(1, k + 1)
    shifts = pd.concat([df["is_failure"].shift(-step) for step in shift_steps], axis=1)
    df["is_failure_within_next_k"] = shifts.max(axis=1)
    df.loc[shifts.isna().any(axis=1), "is_failure_within_next_k"] = np.nan

    # 3. Backward lagged features
    for col in BASE_FEATURES:
        for lag in FEATURE_LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 4. Filter valid frames (boundary drops: first 5, last K)
    valid_cols = FEATURE_COLS + ["is_failure_within_next_k"]
    df_valid = df.dropna(subset=valid_cols).copy().reset_index(drop=True)

    expected_len = n_raw - (max(FEATURE_LAGS) + k)
    if len(df_valid) != expected_len:
        raise ValueError(f"{trial_id}: expected {expected_len} frames, got {len(df_valid)}")

    # 5. Strict label requirement: purge frames where failure has already occurred (F_t == 1)
    if not legacy_label:
        df_valid = df_valid[df_valid["is_failure_current_frame"] == 0].copy().reset_index(drop=True)

    # Add metadata
    df_valid["trial_id"] = trial_id
    df_valid["base_flight"] = trial_row["base_flight"]
    df_valid["trajectory"] = trial_row["trajectory"]
    df_valid["regime"] = trial_row["regime"]
    df_valid["regime_type"] = trial_row["regime_type"]
    if "degradation_alpha" not in df_valid.columns:
        df_valid["degradation_alpha"] = float(trial_row.get("parameter_alpha", 0.0))

    df_valid["is_failure_current_frame"] = df_valid["is_failure_current_frame"].astype(int)
    df_valid["is_failure_within_next_k"] = df_valid["is_failure_within_next_k"].astype(int)
    for lag in FEATURE_LAGS:
        df_valid[f"is_r_frame_lag{lag}"] = df_valid[f"is_r_frame_lag{lag}"].astype(int)

    return df_valid[FINAL_COLS]


def main():
    parser = argparse.ArgumentParser(description="Research 3 — Step 3: Feature Extraction and Dataset Assembly")
    parser.add_argument(
        "--legacy-label",
        action="store_true",
        help="Use legacy label window range(0, k+1) without purging F_t=1 frames (v1 behavior)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output dataset CSV path (default: data/processed/r3_dataset_k5_strict.csv, or r3_dataset_k5.csv if --legacy-label)",
    )
    args = parser.parse_args()

    target_output_path = args.output or (OUTPUT_DATASET_PATH if args.legacy_label else OUTPUT_STRICT_DATASET_PATH)

    print("=" * 70)
    mode_str = "LEGACY (v1, range(0, k+1), includes F_t=1)" if args.legacy_label else "STRICT (v2, range(1, k+1), purges F_t=1)"
    print(f"Research 3 — Step 3: Feature Extraction [{mode_str}]")
    print("=" * 70)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    eligible_trials = manifest_df[manifest_df["eligible"]].to_dict("records")

    trial_dfs = []
    print(f"Extracting features across {len(eligible_trials)} eligible trials (K=5)...")

    for i, t_row in enumerate(eligible_trials, 1):
        t_id = t_row["trial_id"]
        df_t = process_trial_features(t_row, k=5, legacy_label=args.legacy_label)
        trial_dfs.append(df_t)

    full_df = pd.concat(trial_dfs, axis=0, ignore_index=True)
    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(target_output_path, index=False)

    print("=" * 70)
    print(f"-> Full R3 Dataset saved: {target_output_path}")
    print(f"   Total trials: {len(trial_dfs)}")
    print(f"   Total valid samples (K=5): {len(full_df)}")
    print(f"   Overall failure rate (positive label): {full_df['is_failure_within_next_k'].mean()*100:.2f}%")
    print("-" * 70)
    print("Breakdown by Degradation Regime:")
    reg_summary = full_df.groupby("regime").agg(
        n_samples=("is_failure_within_next_k", "count"),
        n_positives=("is_failure_within_next_k", "sum"),
        pos_rate_pct=("is_failure_within_next_k", lambda x: round(x.mean() * 100, 2)),
        mean_inliers=("num_inliers_pose", "mean"),
    ).reset_index()
    print(reg_summary.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
