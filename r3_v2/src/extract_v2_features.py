#!/usr/bin/env python3
"""
Research 3 (MOSAIC) — V2 Leakage-Free Feature and Strict Label Extraction Pipeline.

Extracts:
  1. Strict Label: y_t = max(F_{t+1}..F_{t+5}), strictly future, drops F_t=1 frames.
  2. Feature Set A: Baseline 15 motion telemetry features across lags {0, 1, 2, 3, 5}.
  3. Feature Set B: Set A + 20 image-quality features across lags {0, 1, 2, 3, 5}
     computed from the reconstructed degraded frame I=(1-alpha)*I_clean + alpha*128.
     Features:
       - mean gradient magnitude (Sobel magnitude mean)
       - Laplacian variance
       - pixel std
       - number of Shi-Tomasi corners found
  4. Preserves all flight splits exactly (R2 motion and R3 visual).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(1)
import numpy as np
import pandas as pd

R3_ROOT = Path(__file__).resolve().parent.parent.parent
R2_ROOT = R3_ROOT.parent / "Research2"
ROS_ROOT = R3_ROOT.parent / "ROS"

CACHE_DIR = R3_ROOT / "r3_v2" / "data" / "cache"
OUTPUT_DATA_DIR = R3_ROOT / "r3_v2" / "data"
MANIFEST_PATH = R3_ROOT / "r3_refinement" / "manifests" / "refinement_split_manifest.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_MOTION_FEATURES = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame"]
IMAGE_METRIC_NAMES = ["mean_grad_mag", "laplacian_var", "pixel_std", "shi_tomasi_corners"]
FEATURE_LAGS = [0, 1, 2, 3, 5]

FEAT_A_COLS = [f"{col}_lag{lag}" for col in BASE_MOTION_FEATURES for lag in FEATURE_LAGS]
FEAT_IMG_COLS = [f"{col}_lag{lag}" for col in IMAGE_METRIC_NAMES for lag in FEATURE_LAGS]
FEAT_B_COLS = FEAT_A_COLS + FEAT_IMG_COLS

META_COLS = [
    "sequence_id",
    "domain",
    "split",
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


def compute_image_metrics_single_frame(gray_img: np.ndarray) -> Tuple[float, float, float, float]:
    """Computes the 4 image-quality metrics on a grayscale frame."""
    # 1. Mean Gradient Magnitude
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mean_grad = float(cv2.magnitude(gx, gy).mean())

    # 2. Laplacian Variance (blur metric)
    lap_var = float(cv2.Laplacian(gray_img, cv2.CV_32F).var())

    # 3. Pixel Standard Deviation (contrast metric)
    pix_std = float(gray_img.std())

    # 4. Shi-Tomasi corners found
    pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    corners = float(len(pts) if pts is not None else 0)

    return mean_grad, lap_var, pix_std, corners


def process_r3_trial(trial_id: str, split: str, k: int = 5) -> pd.DataFrame:
    """Processes one R3 visual degradation trial."""
    cache_file = CACHE_DIR / f"{trial_id}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)

    meta_path = R3_ROOT / "data" / "raw" / trial_id / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base_flight = meta["base_flight"]
    source_dir = ROS_ROOT / "results" / "datasets" / base_flight
    if not source_dir.exists():
        # Fallback to metadata path if relative differs
        source_dir = Path(meta["source_dataset_dir"])

    cam_df = pd.read_csv(source_dir / "camera_frames.csv")
    gt_df = pd.read_csv(source_dir / "dataset_gt.csv")
    raw_vo_df = pd.read_csv(R3_ROOT / "data" / "raw" / trial_id / "raw_vo.csv")
    img_dir = source_dir / "images"

    # Active flight window
    act_mask = gt_df["pos_z"].astype(float) >= 2.0
    t0 = float(gt_df.loc[act_mask, "timestamp_total_sec"].min())
    t1 = float(gt_df.loc[act_mask, "timestamp_total_sec"].max())

    cam_act = cam_df.loc[(cam_df["timestamp_total_sec"] >= t0) & (cam_df["timestamp_total_sec"] <= t1)].copy().reset_index(drop=True)
    vo_act = raw_vo_df.copy().reset_index(drop=True)

    if len(cam_act) != len(vo_act):
        raise ValueError(f"{trial_id}: cam_act ({len(cam_act)}) != vo_act ({len(vo_act)})")

    # 1. Failure indicator
    vo_act["is_failure"] = (vo_act["num_inliers_pose"].astype(int) < 8).astype(int)
    vo_act["is_failure_current_frame"] = vo_act["is_failure"]

    # 2. Strict forward label: range(1, k+1)
    shifts = pd.concat([vo_act["is_failure"].shift(-step) for step in range(1, k + 1)], axis=1)
    vo_act["is_failure_within_next_k"] = shifts.max(axis=1)
    vo_act.loc[shifts.isna().any(axis=1), "is_failure_within_next_k"] = np.nan

    # 3. Base motion lagged features
    for col in BASE_MOTION_FEATURES:
        for lag in FEATURE_LAGS:
            vo_act[f"{col}_lag{lag}"] = vo_act[col].shift(lag)

    # 4. Compute image features frame by frame
    grad_list, lap_list, std_list, corn_list = [], [], [], []
    for idx, row in cam_act.iterrows():
        alpha = float(vo_act.iloc[idx]["degradation_alpha"])
        img_path = img_dir / row["filename"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing image {img_path}")

        if alpha > 0.0:
            img_deg = np.clip((1.0 - alpha) * img.astype(float) + alpha * 128.0, 0, 255).astype(np.uint8)
        else:
            img_deg = img

        g, l, s, c = compute_image_metrics_single_frame(img_deg)
        grad_list.append(g)
        lap_list.append(l)
        std_list.append(s)
        corn_list.append(c)

    vo_act["mean_grad_mag"] = grad_list
    vo_act["laplacian_var"] = lap_list
    vo_act["pixel_std"] = std_list
    vo_act["shi_tomasi_corners"] = corn_list

    # Lag image features
    for col in IMAGE_METRIC_NAMES:
        for lag in FEATURE_LAGS:
            vo_act[f"{col}_lag{lag}"] = vo_act[col].shift(lag)

    # Boundary drop (first 5, last 5)
    valid_cols = FEAT_B_COLS + ["is_failure_within_next_k"]
    df_valid = vo_act.dropna(subset=valid_cols).copy().reset_index(drop=True)

    expected_len = len(vo_act) - (max(FEATURE_LAGS) + k)
    if len(df_valid) != expected_len:
        raise ValueError(f"{trial_id}: expected {expected_len} frames, got {len(df_valid)}")

    # Strict label purge: drop frames where failure has already occurred (F_t == 1)
    df_strict = df_valid[df_valid["is_failure_current_frame"] == 0].copy().reset_index(drop=True)

    # Add metadata
    df_strict["sequence_id"] = trial_id
    df_strict["domain"] = "R3_VISUAL"
    df_strict["split"] = split
    df_strict["base_flight"] = base_flight
    df_strict["trajectory"] = meta.get("trajectory", "UNKNOWN")
    df_strict["regime"] = meta.get("regime", "UNKNOWN").upper()
    df_strict["regime_type"] = meta.get("regime_type", "UNKNOWN")
    df_strict["is_failure_current_frame"] = df_strict["is_failure_current_frame"].astype(int)
    df_strict["is_failure_within_next_k"] = df_strict["is_failure_within_next_k"].astype(int)
    for lag in FEATURE_LAGS:
        df_strict[f"is_r_frame_lag{lag}"] = df_strict[f"is_r_frame_lag{lag}"].astype(int)

    final_df = df_strict[META_COLS + FEAT_B_COLS].copy()
    final_df.to_csv(cache_file, index=False)
    return final_df


def process_r2_flight(flight_name: str, split: str, k: int = 5) -> pd.DataFrame:
    """Processes one R2 motion flight."""
    cache_file = CACHE_DIR / f"{flight_name}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)

    source_dir = ROS_ROOT / "results" / "datasets" / flight_name
    cam_df = pd.read_csv(source_dir / "camera_frames.csv")
    gt_df = pd.read_csv(source_dir / "dataset_gt.csv")
    raw_vo_df = pd.read_csv(R2_ROOT / "data" / "raw" / flight_name / "raw_vo.csv")
    img_dir = source_dir / "images"

    # Active flight window: pos_z >= 2.0
    act_mask = gt_df["pos_z"].astype(float) >= 2.0
    t0 = float(gt_df.loc[act_mask, "timestamp_total_sec"].min())
    t1 = float(gt_df.loc[act_mask, "timestamp_total_sec"].max())

    cam_act = cam_df.loc[(cam_df["timestamp_total_sec"] >= t0) & (cam_df["timestamp_total_sec"] <= t1)].copy().reset_index(drop=True)
    raw_t = raw_vo_df["timestamp_total_sec"].astype(float)
    vo_act = raw_vo_df.loc[(raw_t >= t0) & (raw_t <= t1)].copy().reset_index(drop=True)

    if len(cam_act) != len(vo_act):
        raise ValueError(f"{flight_name}: cam_act ({len(cam_act)}) != vo_act ({len(vo_act)})")

    # 1. Failure definition
    vo_act["is_failure"] = (vo_act["num_inliers_pose"].astype(int) < 8).astype(int)
    vo_act["is_failure_current_frame"] = vo_act["is_failure"]

    # 2. Strict forward label: range(1, k+1)
    shifts = pd.concat([vo_act["is_failure"].shift(-step) for step in range(1, k + 1)], axis=1)
    vo_act["is_failure_within_next_k"] = shifts.max(axis=1)
    vo_act.loc[shifts.isna().any(axis=1), "is_failure_within_next_k"] = np.nan

    # 3. Base motion lagged features
    for col in BASE_MOTION_FEATURES:
        for lag in FEATURE_LAGS:
            vo_act[f"{col}_lag{lag}"] = vo_act[col].shift(lag)

    # 4. Image features on clean frames (alpha = 0.0)
    grad_list, lap_list, std_list, corn_list = [], [], [], []
    for idx, row in cam_act.iterrows():
        img_path = img_dir / row["filename"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing image {img_path}")

        g, l, s, c = compute_image_metrics_single_frame(img)
        grad_list.append(g)
        lap_list.append(l)
        std_list.append(s)
        corn_list.append(c)

    vo_act["mean_grad_mag"] = grad_list
    vo_act["laplacian_var"] = lap_list
    vo_act["pixel_std"] = std_list
    vo_act["shi_tomasi_corners"] = corn_list

    # Lag image features
    for col in IMAGE_METRIC_NAMES:
        for lag in FEATURE_LAGS:
            vo_act[f"{col}_lag{lag}"] = vo_act[col].shift(lag)

    # Boundary drop (first 5, last 5)
    valid_cols = FEAT_B_COLS + ["is_failure_within_next_k"]
    df_valid = vo_act.dropna(subset=valid_cols).copy().reset_index(drop=True)

    expected_len = len(vo_act) - (max(FEATURE_LAGS) + k)
    if len(df_valid) != expected_len:
        raise ValueError(f"{flight_name}: expected {expected_len} frames, got {len(df_valid)}")

    # Strict label purge: drop frames where failure has already occurred (F_t == 1)
    df_strict = df_valid[df_valid["is_failure_current_frame"] == 0].copy().reset_index(drop=True)

    # Metadata
    parts = flight_name.split("_")
    traj = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else "MOTION"

    df_strict["sequence_id"] = flight_name
    df_strict["domain"] = "R2_MOTION"
    df_strict["split"] = split
    df_strict["base_flight"] = flight_name
    df_strict["trajectory"] = traj
    df_strict["regime"] = "MOTION"
    df_strict["regime_type"] = "MOTION"
    df_strict["degradation_alpha"] = 0.0
    df_strict["is_failure_current_frame"] = df_strict["is_failure_current_frame"].astype(int)
    df_strict["is_failure_within_next_k"] = df_strict["is_failure_within_next_k"].astype(int)
    for lag in FEATURE_LAGS:
        df_strict[f"is_r_frame_lag{lag}"] = df_strict[f"is_r_frame_lag{lag}"].astype(int)

    final_df = df_strict[META_COLS + FEAT_B_COLS].copy()
    final_df.to_csv(cache_file, index=False)
    return final_df


def _worker_process_sequence(args: Tuple[str, str, str]) -> Tuple[str, int, float]:
    """Helper worker for multiprocessing pool."""
    cv2.setNumThreads(1)
    seq_type, seq_id, split = args
    t0 = time.perf_counter()
    if seq_type == "R3":
        df = process_r3_trial(seq_id, split)
    else:
        df = process_r2_flight(seq_id, split)
    dt = time.perf_counter() - t0
    return seq_id, len(df), dt


def build_v2_datasets() -> pd.DataFrame:
    """Builds the complete V2 dataset across all 82 sequences in parallel."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = []
    # R3 visual trials
    for trial in manifest["r3_visual_mechanism"]["train_trials"]:
        tasks.append(("R3", trial, "train"))
    for trial in manifest["r3_visual_mechanism"]["val_trials"]:
        tasks.append(("R3", trial, "val"))
    for trial in manifest["r3_visual_mechanism"]["protected_test_trials"]:
        tasks.append(("R3", trial, "test"))

    # R2 motion flights
    for flight in manifest["r2_motion_mechanism"]["train_flights"]:
        tasks.append(("R2", flight, "train"))
    for flight in manifest["r2_motion_mechanism"]["val_flights"]:
        tasks.append(("R2", flight, "val"))
    for flight in manifest["r2_motion_mechanism"]["protected_test_flights"]:
        tasks.append(("R2", flight, "test"))

    print(f"Extracting V2 strict features across {len(tasks)} sequences ({len(tasks)} flights/trials)...")
    num_cpus = min(12, os.cpu_count() or 4)
    print(f"Using {num_cpus} parallel workers.")

    t_start = time.perf_counter()
    with mp.Pool(num_cpus) as pool:
        for idx, (seq_id, n_frames, dt) in enumerate(pool.imap_unordered(_worker_process_sequence, tasks), 1):
            print(f"[{idx:02d}/{len(tasks):02d}] Processed {seq_id}: {n_frames} strict frames ({dt:.1f}s)")

    total_time = time.perf_counter() - t_start
    print(f"All {len(tasks)} sequences extracted in {total_time:.1f}s ({total_time/60:.2f} min).")

    # Combine into master datasets
    all_dfs = []
    for seq_type, seq_id, _ in tasks:
        cache_file = CACHE_DIR / f"{seq_id}.csv"
        all_dfs.append(pd.read_csv(cache_file))

    master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    master_path = OUTPUT_DATA_DIR / "r3_v2_strict_dataset.csv"
    master_df.to_csv(master_path, index=False)
    print(f"-> Master V2 dataset saved: {master_path} ({len(master_df)} frames)")

    return master_df


if __name__ == "__main__":
    build_v2_datasets()
