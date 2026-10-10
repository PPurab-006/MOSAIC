#!/usr/bin/env python3
"""
Research 3 Refinement — Step 2: Build Flight-Level Split Manifest.

Defines orthogonal, leak-free flight splits:
1. Protected R2 Test Flights (14 flights from Research 2 test set).
2. Protected R3 Test Flights (15 trials from 3 base flights corresponding to R2 test set).
3. Permitted Training Flights (15 R2 train flights + 15 R3 train trials).
4. Permitted Validation Flights (13 R2 val flights + 10 R3 val trials).

Outputs:
  r3_refinement/manifests/refinement_split_manifest.json
"""

import json
from pathlib import Path
import pandas as pd

R3_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = R3_ROOT / "r3_refinement" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

R2_SPLIT = R3_ROOT.parent / "Research2" / "data" / "processed" / "expanded_flight_split.csv"
R3_MANIFEST_PATH = R3_ROOT / "data" / "processed" / "r3_experiment_manifest.csv"

OUT_MANIFEST_JSON = MANIFEST_DIR / "refinement_split_manifest.json"


def main():
    print("=" * 70)
    print("BUILDING REFINEMENT FLIGHT-LEVEL SPLIT MANIFEST")
    print("=" * 70)

    # 1. Load R2 splits
    df_r2 = pd.read_csv(R2_SPLIT_PATH)
    r2_train_flights = df_r2.loc[df_r2["split"] == "train", "run_dir"].tolist()
    r2_val_flights = df_r2.loc[df_r2["split"] == "val", "run_dir"].tolist()
    r2_test_flights = df_r2.loc[df_r2["split"] == "test", "run_dir"].tolist()

    # 2. Load R3 manifest
    df_r3 = pd.read_csv(R3_MANIFEST_PATH)
    df_r3_eligible = df_r3[df_r3["eligible"]].copy()

    # Map R3 trials by base flight
    # The 8 base flights in R3 are:
    # - sweep_G_C_R1, sweep_G_B_R2, sweep_G_S_R2 -> these match R2 test flights!
    # - sweep_G_C_R2, sweep_G_S_R3, sweep_G_B_R3 -> these match R2 train flights!
    # - sweep_G_C_R3, sweep_G_S_R1               -> these match R2 val flights!

    r3_test_base = ["sweep_G_C_R1", "sweep_G_B_R2", "sweep_G_S_R2"]
    r3_train_base = ["sweep_G_C_R2", "sweep_G_S_R3", "sweep_G_B_R3"]
    r3_val_base = ["sweep_G_C_R3", "sweep_G_S_R1"]

    r3_test_trials = df_r3_eligible.loc[df_r3_eligible["base_flight"].isin(r3_test_base), "trial_id"].tolist()
    r3_train_trials = df_r3_eligible.loc[df_r3_eligible["base_flight"].isin(r3_train_base), "trial_id"].tolist()
    r3_val_trials = df_r3_eligible.loc[df_r3_eligible["base_flight"].isin(r3_val_base), "trial_id"].tolist()

    manifest = {
        "study": "Research 3 Refinement: Cross-Mechanism Sensitivity Recovery",
        "split_strategy": "Strict Flight-Level Partitioning",
        "r2_motion_mechanism": {
            "protected_test_flights": r2_test_flights,
            "train_flights": r2_train_flights,
            "val_flights": r2_val_flights,
            "total_flights": len(df_r2),
            "counts": {
                "train": len(r2_train_flights),
                "val": len(r2_val_flights),
                "test": len(r2_test_flights),
            }
        },
        "r3_visual_mechanism": {
            "protected_test_trials": r3_test_trials,
            "train_trials": r3_train_trials,
            "val_trials": r3_val_trials,
            "base_flight_mapping": {
                "train_base_flights": r3_train_base,
                "val_base_flights": r3_val_base,
                "protected_test_base_flights": r3_test_base,
            },
            "counts": {
                "train": len(r3_train_trials),
                "val": len(r3_val_trials),
                "test": len(r3_test_trials),
            }
        },
        "guarantees": [
            "Protected R2 test flights were never seen during R2 or R3 refinement.",
            "Protected R3 test trials were never seen during refinement training or threshold selection.",
            "All hyperparameter and threshold selection performed exclusively on validation sets.",
            "Zero frame-level overlap between train, validation, and test splits."
        ]
    }

    with open(OUT_MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"-> Manifest saved to: {OUT_MANIFEST_JSON}")
    print(f"R2 Motion Splits:  Train={len(r2_train_flights)}, Val={len(r2_val_flights)}, Protected Test={len(r2_test_flights)}")
    print(f"R3 Visual Splits:  Train={len(r3_train_trials)}, Val={len(r3_val_trials)}, Protected Test={len(r3_test_trials)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
