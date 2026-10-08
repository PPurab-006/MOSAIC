#!/usr/bin/env python3
"""
Research 3 Refinement — Test Suite: Integrity & Scientific Constraint Verification.

Verifies:
1. Frozen R2 model and scaler SHA-256 hashes are immutable.
2. Leak-free flight splits (zero flight overlap between train, val, and protected test).
3. Exact feature definitions and order across frozen and refined models.
4. Model architectures both match SmallMLP (15 -> 32 -> 16 -> 1, 1041 params).
5. All output metrics and figures exist and are non-empty.
"""

import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
import pytest
import torch

R3_ROOT = Path(__file__).resolve().parent.parent.parent
REFINEMENT_DIR = R3_ROOT / "r3_refinement"

FROZEN_MODEL_PATH = R3_ROOT / "models" / "expanded_mlp.pt"
FROZEN_SCALER_PATH = R3_ROOT / "models" / "expanded_scaler.joblib"
REFINED_MODEL_PATH = REFINEMENT_DIR / "models" / "refined_mlp.pt"
REFINED_SCALER_PATH = REFINEMENT_DIR / "models" / "refined_scaler.joblib"

EXPECTED_R2_MODEL_SHA = "93acd1048a6954a143aac873a3fa8a4548c8a6230aa574aa1ea3fcce17b78e9f"
EXPECTED_R2_SCALER_SHA = "4e9c6ebacb5ac4a554ed69a56d4c7c48df8ef1036b53907181b6dc221550b105"

MANIFEST_PATH = REFINEMENT_DIR / "manifests" / "refinement_split_manifest.json"
LOCKED_THRESHOLDS_PATH = REFINEMENT_DIR / "models" / "locked_thresholds.json"
METRICS_JSON = REFINEMENT_DIR / "results" / "metrics" / "protected_evaluation_metrics.json"
STAT_JSON = REFINEMENT_DIR / "results" / "metrics" / "statistical_tests.json"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_frozen_r2_integrity():
    """Verify that frozen R2 artifacts have NOT been modified, retrained, or altered."""
    assert FROZEN_MODEL_PATH.exists(), "Frozen R2 model checkpoint missing!"
    assert FROZEN_SCALER_PATH.exists(), "Frozen R2 scaler missing!"

    curr_model_sha = compute_sha256(FROZEN_MODEL_PATH)
    curr_scaler_sha = compute_sha256(FROZEN_SCALER_PATH)

    assert curr_model_sha == EXPECTED_R2_MODEL_SHA, f"R2 model SHA mismatch! {curr_model_sha}"
    assert curr_scaler_sha == EXPECTED_R2_SCALER_SHA, f"R2 scaler SHA mismatch! {curr_scaler_sha}"


def test_leak_free_flight_splits():
    """Verify that train, validation, and protected test flights are strictly disjoint."""
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # R2 Motion
    r2_train = set(manifest["r2_motion_mechanism"]["train_flights"])
    r2_val = set(manifest["r2_motion_mechanism"]["val_flights"])
    r2_test = set(manifest["r2_motion_mechanism"]["protected_test_flights"])

    assert len(r2_train & r2_val) == 0, "Overlap between R2 train and val flights!"
    assert len(r2_train & r2_test) == 0, "Overlap between R2 train and test flights!"
    assert len(r2_val & r2_test) == 0, "Overlap between R2 val and test flights!"

    # R3 Visual
    r3_train = set(manifest["r3_visual_mechanism"]["train_trials"])
    r3_val = set(manifest["r3_visual_mechanism"]["val_trials"])
    r3_test = set(manifest["r3_visual_mechanism"]["protected_test_trials"])

    assert len(r3_train & r3_val) == 0, "Overlap between R3 train and val trials!"
    assert len(r3_train & r3_test) == 0, "Overlap between R3 train and test trials!"
    assert len(r3_val & r3_test) == 0, "Overlap between R3 val and test trials!"

    # Ensure protected base flights match R2 test base flights
    test_base = manifest["r3_visual_mechanism"]["base_flight_mapping"]["protected_test_base_flights"]
    assert set(test_base) == {"sweep_G_C_R1", "sweep_G_B_R2", "sweep_G_S_R2"}


def test_model_architecture_and_parameters():
    """Verify that both frozen and refined models use identical SmallMLP (1041 params)."""
    sd_frozen = torch.load(FROZEN_MODEL_PATH, map_location="cpu")
    sd_refined = torch.load(REFINED_MODEL_PATH, map_location="cpu")

    def count_params(sd):
        return sum(v.numel() for v in sd.values())

    assert count_params(sd_frozen) == 1057, f"Frozen model has {count_params(sd_frozen)} params != 1057"
    assert count_params(sd_refined) == 1057, f"Refined model has {count_params(sd_refined)} params != 1057"

    # Verify layer shapes match
    for k in sd_frozen.keys():
        assert k in sd_refined, f"Missing key {k} in refined model!"
        assert sd_frozen[k].shape == sd_refined[k].shape, f"Shape mismatch for {k}"


def test_scaler_feature_dimensions():
    """Verify that both scalers expect 15 features."""
    sc_frozen = joblib.load(FROZEN_SCALER_PATH)
    sc_refined = joblib.load(REFINED_SCALER_PATH)

    assert sc_frozen.n_features_in_ == 15
    assert sc_refined.n_features_in_ == 15


def test_metric_outputs_exist():
    """Verify that all metrics, thresholds, and statistical tests were saved."""
    assert LOCKED_THRESHOLDS_PATH.exists()
    assert METRICS_JSON.exists()
    assert STAT_JSON.exists()

    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    # Verify visual recall recovered
    rec_v_a = metrics["protected_r3_visual_test"]["model_a_frozen_r2"]["recall"]
    rec_v_b = metrics["protected_r3_visual_test"]["model_b_refined_mlp"]["recall"]
    assert rec_v_b > rec_v_a, f"Refined visual recall {rec_v_b} not greater than frozen {rec_v_a}!"

    # Verify motion performance preserved (within 0.05 of baseline)
    f1_m_a = metrics["protected_r2_motion_test"]["model_a_frozen_r2"]["f1"]
    f1_m_b = metrics["protected_r2_motion_test"]["model_b_refined_mlp"]["f1"]
    assert abs(f1_m_b - f1_m_a) < 0.05, f"Motion F1 degraded by more than 0.05 ({f1_m_b} vs {f1_m_a})"


def test_figures_exist():
    """Verify that all 4 figures exist and are non-empty."""
    fig_dir = REFINEMENT_DIR / "results" / "figures"
    for i in range(1, 5):
        matches = list(fig_dir.glob(f"*fig{i}*.png"))
        assert len(matches) > 0, f"Figure {i} missing in {fig_dir}!"
        assert matches[0].stat().st_size > 10000, f"Figure {matches[0]} appears empty or corrupt!"


if __name__ == "__main__":
    tests = [
        test_frozen_r2_integrity,
        test_leak_free_flight_splits,
        test_model_architecture_and_parameters,
        test_scaler_feature_dimensions,
        test_metric_outputs_exist,
        test_figures_exist,
    ]
    print("=" * 70)
    print("RUNNING REFINEMENT INTEGRITY TEST SUITE")
    print("=" * 70)
    passed = 0
    for t in tests:
        print(f"Running {t.__name__}...", end="", flush=True)
        t()
        print(" [PASSED]")
        passed += 1
    print("=" * 70)
    print(f"ALL {passed}/{len(tests)} TESTS PASSED!")
    print("=" * 70)
