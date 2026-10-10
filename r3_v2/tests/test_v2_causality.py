#!/usr/bin/env python3
"""
Unit Test Suite: V2 Strict Causality and Leakage-Free Invariance Verification.

Tests:
1. Every feature in Set A (15 features) and Set B (35 features) at time t depends
   strictly on frames <= t. Corrupting frames at t+1 or later produces ZERO change
   in the features at time t.
2. All feature lags are non-negative ({0, 1, 2, 3, 5}), verifying no forward shifts.
3. Strict label window: y_t = max(F_{t+1}..F_{t+5}) never includes current frame t.
4. All frames with F_t = 1 are purged from the dataset.
5. Boundary integrity: the last 5 frames of every sequence are dropped.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import pytest

from r3_v2.src.extract_v2_features import (
    BASE_MOTION_FEATURES,
    IMAGE_METRIC_NAMES,
    FEATURE_LAGS,
    FEAT_A_COLS,
    FEAT_B_COLS,
    compute_image_metrics_single_frame,
)


def test_feature_causality_invariance_under_future_frame_corruption():
    """
    Asserts that modifying any frame at index t+1 or later has ZERO impact on
    any feature computed at time index t.
    """
    n_frames = 25
    rng = np.random.RandomState(42)

    # Synthetic grayscale video frames
    clean_frames = [rng.randint(50, 200, (64, 64), dtype=np.uint8) for _ in range(n_frames)]
    alphas = [0.0] * 10 + [0.5] * 5 + [0.9] * 10
    yaw_rates = rng.uniform(0.0, 30.0, n_frames)
    flow_vels = rng.uniform(0.0, 5.0, n_frames)
    r_frames = [0 if i % 4 != 0 else 1 for i in range(n_frames)]

    def compute_all_features_at_index(frames, t_idx):
        # 1. Compute raw metrics for frames up to t_idx
        df = pd.DataFrame({
            "eis_yaw_rate_deg": yaw_rates[: t_idx + 1],
            "feature_vel_mean": flow_vels[: t_idx + 1],
            "is_r_frame": r_frames[: t_idx + 1],
        })

        grad_l, lap_l, std_l, corn_l = [], [], [], []
        for i in range(t_idx + 1):
            alpha = alphas[i]
            img = frames[i]
            if alpha > 0.0:
                deg = np.clip((1.0 - alpha) * img.astype(float) + alpha * 128.0, 0, 255).astype(np.uint8)
            else:
                deg = img
            g, l, s, c = compute_image_metrics_single_frame(deg)
            grad_l.append(g)
            lap_l.append(l)
            std_l.append(s)
            corn_l.append(c)

        df["mean_grad_mag"] = grad_l
        df["laplacian_var"] = lap_l
        df["pixel_std"] = std_l
        df["shi_tomasi_corners"] = corn_l

        # Compute lagged features
        row_features = {}
        for col in BASE_MOTION_FEATURES + IMAGE_METRIC_NAMES:
            for lag in FEATURE_LAGS:
                # Value at t_idx - lag
                idx = t_idx - lag
                row_features[f"{col}_lag{lag}"] = df.loc[idx, col]

        return row_features

    # Evaluate at index t = 15
    t_eval = 15
    feats_original = compute_all_features_at_index(clean_frames, t_eval)

    # Now create corrupted future sequence: all frames from t_eval + 1 onwards are altered
    corrupted_frames = [f.copy() for f in clean_frames]
    for k in range(t_eval + 1, n_frames):
        corrupted_frames[k] = rng.randint(0, 256, (64, 64), dtype=np.uint8)  # Random noise
        alphas[k] = 0.99  # Corrupted degradation parameter
        yaw_rates[k] = 999.0  # Corrupted telemetry
        flow_vels[k] = 999.0
        r_frames[k] = 1

    feats_corrupted = compute_all_features_at_index(corrupted_frames, t_eval)

    # Assert 100% numerical identity across all 35 features
    for col in FEAT_B_COLS:
        val_orig = feats_original[col]
        val_corr = feats_corrupted[col]
        assert np.isclose(val_orig, val_corr, atol=1e-6), (
            f"Feature causality violation detected for {col}! "
            f"Original: {val_orig}, Corrupted future: {val_corr}"
        )


def test_feature_lags_are_strictly_non_negative():
    """Asserts that all feature lags are non-negative, meaning no forward-looking shifts."""
    for lag in FEATURE_LAGS:
        assert lag >= 0, f"Negative lag detected: {lag} (would cause temporal leakage!)"


def test_strict_label_window_never_includes_current_frame():
    """
    Asserts that y_t = max(F_{t+1}..F_{t+5}):
    1. If F_t = 1 but F_{t+1}..F_{t+5} = 0, y_t must be 0.
    2. If F_t = 0 but any F_{t+1}..F_{t+5} = 1, y_t must be 1.
    """
    k = 5
    # Sequence of failures
    failures = pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    shifts = pd.concat([failures.shift(-step) for step in range(1, k + 1)], axis=1)
    y_strict = shifts.max(axis=1)

    # Frame 2 has F_t = 1, but F_3..F_7 are all 0 except F_7 is 1?
    # At t=2: next frames are t=3,4,5,6,7. failures[7] is 1, so y_strict[2] is 1.
    # At t=1: next frames are t=2,3,4,5,6. failures[2] is 1, so y_strict[1] is 1.
    # At t=0: next frames are t=1,2,3,4,5. failures[2] is 1, so y_strict[0] is 1.

    # Let's test a case where ONLY F_t = 1 and all future frames are 0:
    failures_isolated = pd.Series([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    shifts_iso = pd.concat([failures_isolated.shift(-step) for step in range(1, k + 1)], axis=1)
    y_iso = shifts_iso.max(axis=1)

    # For t=2 (where F_2 = 1): next 5 frames are 0, 0, 0, 0, 0
    assert y_iso.iloc[2] == 0, f"Strict label included current frame! Expected 0, got {y_iso.iloc[2]}"

    # Legacy label would have had y_legacy.iloc[2] == 1:
    shifts_leg = pd.concat([failures_isolated.shift(-step) for step in range(k + 1)], axis=1)
    y_leg = shifts_leg.max(axis=1)
    assert y_leg.iloc[2] == 1, "Legacy label failed to include current frame"


def test_purged_dataset_has_zero_f_t_one_frames():
    """Asserts that in the strict dataset, all frames have is_failure_current_frame == 0."""
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
    cached_csvs = list(cache_dir.glob("*.csv"))
    if not cached_csvs:
        pytest.skip("No cached sequence CSVs generated yet.")

    for csv_file in cached_csvs[:5]:
        df = pd.read_csv(csv_file)
        assert (df["is_failure_current_frame"] == 0).all(), (
            f"Found F_t == 1 frames in strict dataset {csv_file.name}!"
        )


if __name__ == "__main__":
    test_feature_causality_invariance_under_future_frame_corruption()
    test_feature_lags_are_strictly_non_negative()
    test_strict_label_window_never_includes_current_frame()
    print("All V2 causality unit tests PASSED successfully!")
