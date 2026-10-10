#!/usr/bin/env python3
"""
Research 3 — Step 2: Run Offline Monocular Visual Odometry across all R3 Trials.

Executes the authoritative monocular VO pipeline (KLT tracking, 5-point Essential
matrix RANSAC, recoverPose with distanceThresh=1000.0) from ROS/src/pipelines/run_offline_vo.py
on every eligible experimental trial under controlled visual observability/texture degradation.

Saves:
  - data/raw/<trial_id>/raw_vo.csv
  - data/raw/<trial_id>/dataset_gt.csv (copied for self-contained flight evaluation)
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

R3_ROOT = Path(__file__).resolve().parent.parent
ROS_REPO = R3_ROOT.parent / "ROS"
sys.path.insert(0, str(ROS_REPO / "src" / "pipelines"))
from run_offline_vo import OfflineVOProcessor

R3_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = R3_ROOT / "data" / "raw"
PROCESSED_DIR = R3_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "r3_experiment_manifest.csv"


def process_single_trial(trial_row: dict) -> dict:
    trial_id = trial_row["trial_id"]
    trial_dir = RAW_DIR / trial_id
    meta_path = trial_dir / "metadata.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    source_dir = Path(meta["source_dataset_dir"])
    gt_csv_src = source_dir / "dataset_gt.csv"
    cam_csv_src = source_dir / "camera_frames.csv"
    img_dir_src = source_dir / "images"

    gt_df = pd.read_csv(gt_csv_src)
    cam_df = pd.read_csv(cam_csv_src)

    # Active flight window: pos_z >= 2.0
    act_mask = gt_df["pos_z"].astype(float) >= 2.0
    t0 = float(gt_df.loc[act_mask, "timestamp_total_sec"].min())
    t1 = float(gt_df.loc[act_mask, "timestamp_total_sec"].max())

    cam_act = cam_df.loc[(cam_df["timestamp_total_sec"] >= t0) & (cam_df["timestamp_total_sec"] <= t1)].copy().reset_index(drop=True)
    gt_act = gt_df.loc[(gt_df["timestamp_total_sec"] >= t0) & (gt_df["timestamp_total_sec"] <= t1)].copy().reset_index(drop=True)

    # Copy GT slice into trial dir
    gt_act.to_csv(trial_dir / "dataset_gt.csv", index=False)

    vo_csv_out = trial_dir / "raw_vo.csv"

    # Instantiate canonical OfflineVOProcessor
    proc = OfflineVOProcessor(output_csv_path=str(vo_csv_out), mode="klt")
    proc.load_telemetry(str(gt_csv_src))

    reg_type = meta["regime_type"]
    alpha_val = meta.get("alpha", 0.0)
    alpha_start = meta.get("alpha_start", 0.0)
    alpha_end = meta.get("alpha_end", 0.975)
    trans_frames = meta.get("transition_frames", [200, 350])

    t_start = time.perf_counter()
    mean_ambient = 128.0

    applied_alphas = []
    for f_idx, row in cam_act.iterrows():
        # Determine alpha for current frame
        if reg_type == "constant":
            alpha = float(alpha_val)
        else:
            if f_idx < trans_frames[0]:
                alpha = float(alpha_start)
            elif f_idx < trans_frames[1]:
                prog = (f_idx - trans_frames[0]) / (trans_frames[1] - trans_frames[0])
                alpha = float(alpha_start + prog * (alpha_end - alpha_start))
            else:
                alpha = float(alpha_end)

        applied_alphas.append(alpha)

        # Load raw frame
        img_path = img_dir_src / row["filename"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing frame image: {img_path}")

        # Apply degradation
        if alpha > 0.0:
            img_deg = np.clip((1.0 - alpha) * img.astype(float) + alpha * mean_ambient, 0, 255).astype(np.uint8)
        else:
            img_deg = img

        proc.process_frame(
            img_deg,
            int(row["timestamp_sec"]),
            int(row["timestamp_nanosec"]),
            float(row["timestamp_total_sec"])
        )

    proc.save_csv()
    t_elapsed = time.perf_counter() - t_start

    # Load produced raw_vo.csv and append applied degradation parameter column
    vo_df = pd.read_csv(vo_csv_out)
    vo_df["degradation_alpha"] = applied_alphas
    vo_df.to_csv(vo_csv_out, index=False)

    inliers = vo_df["num_inliers_pose"]
    fail_rate = (inliers < 8).mean() * 100.0

    return {
        "trial_id": trial_id,
        "n_frames": len(vo_df),
        "mean_inliers": float(inliers.mean()),
        "p50_inliers": float(inliers.median()),
        "fail_rate_pct": float(fail_rate),
        "elapsed_s": float(t_elapsed),
    }


def main():
    print("=" * 70)
    print("Research 3 — Step 2: Executing Monocular VO Pipeline on All Trials")
    print("=" * 70)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    eligible_trials = manifest_df[manifest_df["eligible"]].to_dict("records")

    print(f"Total trials to process: {len(eligible_trials)}")

    results = []
    total_start = time.perf_counter()

    for i, t_row in enumerate(eligible_trials, 1):
        t_id = t_row["trial_id"]
        reg = t_row["regime"]
        base = t_row["base_flight"]
        print(f"[{i:02d}/{len(eligible_trials):02d}] Processing {t_id} (Base: {base}, Regime: {reg})...", end="", flush=True)

        res = process_single_trial(t_row)
        results.append(res)

        print(f" Done in {res['elapsed_s']:.1f}s | Inliers: {res['mean_inliers']:.1f} | Failures: {res['fail_rate_pct']:.1f}%")

    total_time = time.perf_counter() - total_start
    print("=" * 70)
    print(f"-> All {len(eligible_trials)} trials completed in {total_time:.1f}s ({total_time/60:.2f} min).")
    print("=" * 70)


if __name__ == "__main__":
    main()
