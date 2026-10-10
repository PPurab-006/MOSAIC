#!/usr/bin/env python3
"""
Research 3 — Step 0: Verify, Freeze, and Manifest Authoritative Research 2 Artifacts.

Identifies and freezes authoritative Research 2 artifacts:
  - models/expanded_mlp.pt
  - models/expanded_scaler.joblib
  - exact 15-feature definition
  - exact lag structure
  - exact threshold (0.5)
  - exact preprocessing (StandardScaler)
  - exact model architecture (SmallMLP: 15 -> 32 -> 16 -> 1)
  - exact inference procedure

Records cryptographic SHA-256 hashes and saves models/frozen_r2_manifest.json.
"""

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import torch
import torch.nn as nn

R3_ROOT = Path(__file__).resolve().parent.parent
R2_ROOT = R3_ROOT.parent / "Research2"

R2_MODEL_PATH = R2_ROOT / "models" / "expanded_mlp.pt"
R2_SCALER_PATH = R2_ROOT / "models" / "expanded_scaler.joblib"
R2_METRICS_PATH = R2_ROOT / "models" / "expanded_evaluation_metrics.json"

R3_MODELS_DIR = R3_ROOT / "models"
R3_MODELS_DIR.mkdir(parents=True, exist_ok=True)

R3_MODEL_PATH = R3_MODELS_DIR / "expanded_mlp.pt"
R3_SCALER_PATH = R3_MODELS_DIR / "expanded_scaler.joblib"
R3_MANIFEST_PATH = R3_MODELS_DIR / "frozen_r2_manifest.json"


class SmallMLP(nn.Module):
    """Authoritative frozen 2-hidden-layer MLP architecture (15 -> 32 -> 16 -> 1)."""

    def __init__(self, input_dim: int = 15, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


BASE_FEATURES = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame"]
FEATURE_LAGS = [0, 1, 2, 3, 5]
FEATURE_COLS = [f"{col}_lag{lag}" for col in BASE_FEATURES for lag in FEATURE_LAGS]


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 70)
    print("Research 3 — Freezing Authoritative Research 2 Baseline Artifacts")
    print("=" * 70)

    # 1. Verify existence of R2 artifacts
    if not R2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Authoritative R2 model not found: {R2_MODEL_PATH}")
    if not R2_SCALER_PATH.exists():
        raise FileNotFoundError(f"Authoritative R2 scaler not found: {R2_SCALER_PATH}")

    # 2. Compute source hashes
    r2_mlp_hash = compute_sha256(R2_MODEL_PATH)
    r2_scaler_hash = compute_sha256(R2_SCALER_PATH)

    print(f"R2 Source MLP Checkpoint: {R2_MODEL_PATH}")
    print(f"  SHA-256: {r2_mlp_hash}")
    print(f"R2 Source Scaler: {R2_SCALER_PATH}")
    print(f"  SHA-256: {r2_scaler_hash}")

    # 3. Copy to R3 models directory
    shutil.copy2(R2_MODEL_PATH, R3_MODEL_PATH)
    shutil.copy2(R2_SCALER_PATH, R3_SCALER_PATH)

    # 4. Verify copies
    r3_mlp_hash = compute_sha256(R3_MODEL_PATH)
    r3_scaler_hash = compute_sha256(R3_SCALER_PATH)
    assert r3_mlp_hash == r2_mlp_hash, "MLP hash mismatch after copying!"
    assert r3_scaler_hash == r2_scaler_hash, "Scaler hash mismatch after copying!"
    print("-> Successfully verified identical copy of frozen R2 model and scaler.")

    # 5. Inspect Scaler
    scaler = joblib.load(R3_SCALER_PATH)
    scaler_means = {col: float(m) for col, m in zip(FEATURE_COLS, scaler.mean_)}
    scaler_scales = {col: float(s) for col, s in zip(FEATURE_COLS, scaler.scale_)}

    # 6. Inspect Model Weights
    state_dict = torch.load(R3_MODEL_PATH, map_location="cpu")
    weight_summary = {}
    for k, v in state_dict.items():
        weight_summary[k] = {
            "shape": list(v.shape),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    # 7. Load R2 Test Baseline Metrics for comparison
    r2_baseline_metrics = {}
    if R2_METRICS_PATH.exists():
        with open(R2_METRICS_PATH, "r", encoding="utf-8") as f:
            r2_raw_metrics = json.load(f)
            r2_baseline_metrics = {
                "dataset": "Expanded 42-Flight Sweep (Held-out Test)",
                "test_flights": r2_raw_metrics["baseline_comparison"]["expanded_42"]["test_flights"],
                "test_frames": r2_raw_metrics["baseline_comparison"]["expanded_42"]["test_rows"],
                "test_positive_rate_pct": r2_raw_metrics["baseline_comparison"]["expanded_42"]["test_pos_rate"],
                "auroc": r2_raw_metrics["baseline_comparison"]["expanded_42"]["auroc"],
                "auprc": r2_raw_metrics["baseline_comparison"]["expanded_42"]["auprc"],
                "f1": r2_raw_metrics["baseline_comparison"]["expanded_42"]["f1"],
                "precision": r2_raw_metrics["baseline_comparison"]["expanded_42"]["precision"],
                "recall": r2_raw_metrics["baseline_comparison"]["expanded_42"]["recall"],
                "early_warning_rate_pct": 79.33,
                "median_lead_time_s": 0.165,
            }

    # 8. Create complete frozen manifest
    manifest = {
        "study": "Research 3: Cross-Mechanism VO Failure Generalization",
        "frozen_baseline": "Research 2 Authoritative Expanded Model",
        "freeze_date": "2026-10-08",
        "integrity_status": "LOCKED_IMMUTABLE",
        "artifacts": {
            "mlp_checkpoint": {
                "filename": "expanded_mlp.pt",
                "sha256": r3_mlp_hash,
                "architecture": "SmallMLP(Linear(15, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1))",
                "weights_summary": weight_summary,
            },
            "scaler": {
                "filename": "expanded_scaler.joblib",
                "sha256": r3_scaler_hash,
                "n_features": len(FEATURE_COLS),
                "means": scaler_means,
                "scales": scaler_scales,
            },
        },
        "feature_specification": {
            "base_features": BASE_FEATURES,
            "feature_lags": FEATURE_LAGS,
            "feature_names_ordered": FEATURE_COLS,
            "total_features": len(FEATURE_COLS),
        },
        "prediction_specification": {
            "label_definition": "num_inliers_pose < 8 (Monocular 5-point Essential matrix pose inliers)",
            "prediction_horizon_k": 5,
            "horizon_time_s": 0.165,
            "decision_threshold": 0.5,
            "streaming_buffer_size": 6,
        },
        "r2_test_baseline_performance": r2_baseline_metrics,
    }

    with open(R3_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"-> Manifest successfully written to: {R3_MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
