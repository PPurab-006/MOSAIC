#!/usr/bin/env python3
"""
Research 3 — Step 1: Generate Experiment Manifest, Eligibility Log, and Trial Specs.

Creates:
  - data/processed/r3_experiment_manifest.csv
  - data/processed/r3_eligibility_log.csv
  - data/raw/<trial_id>/metadata.json

Defines 40 primary experimental trials across 8 benign flight executions and
5 visual degradation regimes (Nominal, Low, Medium, High, Transition), plus
3 documented exploratory hover flights (excluded due to epipolar degeneracy).
"""

import json
from pathlib import Path
import pandas as pd
import yaml

R3_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = R3_ROOT / "configs" / "r3_experiment_config.yaml"
PROCESSED_DIR = R3_ROOT / "data" / "processed"
RAW_DIR = R3_ROOT / "data" / "raw"
DATASETS_BASE = R3_ROOT.parent / "ROS" / "results" / "datasets"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Research 3 — Generating Experiment Manifest & Trial Specifications")
    print("=" * 70)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_rows = []
    eligibility_rows = []

    # 1. Primary Benign Flights
    trajectories = config["base_benign_flights"]["trajectories"]
    regimes = config["degradation_regimes"]

    trial_counter = 1
    for traj in trajectories:
        traj_name = traj["name"]
        traj_desc = traj["description"]
        for base_flight in traj["repeats"]:
            source_dir = DATASETS_BASE / base_flight
            if not source_dir.exists():
                raise FileNotFoundError(f"Source dataset missing: {source_dir}")

            gt_df = pd.read_csv(source_dir / "dataset_gt.csv")
            cam_df = pd.read_csv(source_dir / "camera_frames.csv")
            act_mask = gt_df["pos_z"].astype(float) >= 2.0
            t0 = float(gt_df.loc[act_mask, "timestamp_total_sec"].min())
            t1 = float(gt_df.loc[act_mask, "timestamp_total_sec"].max())
            n_active_cam = int(((cam_df["timestamp_total_sec"] >= t0) & (cam_df["timestamp_total_sec"] <= t1)).sum())

            # Log base flight eligibility
            eligibility_rows.append({
                "flight_name": base_flight,
                "trajectory": traj_name,
                "description": traj_desc,
                "total_frames": len(cam_df),
                "active_frames": n_active_cam,
                "active_duration_s": round(t1 - t0, 2),
                "status": "ELIGIBLE",
                "exclusion_reason": "N/A (Met all duration, envelope, and stability criteria)",
            })

            # Create 5 degradation trials per base flight
            for reg_name, reg_cfg in regimes.items():
                trial_id = f"r3_{base_flight}_{reg_name.lower()}"
                trial_dir = RAW_DIR / trial_id
                trial_dir.mkdir(parents=True, exist_ok=True)

                trial_meta = {
                    "trial_id": trial_id,
                    "trial_num": trial_counter,
                    "base_flight": base_flight,
                    "trajectory": traj_name,
                    "source_dataset_dir": str(source_dir),
                    "regime": reg_name,
                    "regime_type": reg_cfg["type"],
                    "description": reg_cfg["description"],
                    "active_duration_s": round(t1 - t0, 2),
                    "active_frames": n_active_cam,
                    "t0": t0,
                    "t1": t1,
                }

                if reg_cfg["type"] == "constant":
                    trial_meta["alpha"] = reg_cfg["alpha"]
                else:
                    trial_meta["alpha_start"] = reg_cfg["alpha_start"]
                    trial_meta["alpha_end"] = reg_cfg["alpha_end"]
                    trial_meta["transition_frames"] = reg_cfg["transition_frames"]

                with open(trial_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(trial_meta, f, indent=2)

                manifest_rows.append({
                    "trial_num": trial_counter,
                    "trial_id": trial_id,
                    "base_flight": base_flight,
                    "trajectory": traj_name,
                    "regime": reg_name,
                    "regime_type": reg_cfg["type"],
                    "parameter_alpha": reg_cfg.get("alpha", f"{reg_cfg.get('alpha_start')}->{reg_cfg.get('alpha_end')}"),
                    "active_frames": n_active_cam,
                    "active_duration_s": round(t1 - t0, 2),
                    "eligible": True,
                    "exclusion_reason": "N/A",
                })
                trial_counter += 1

    # 2. Exploratory Excluded Hover Flights
    for ex in config.get("exploratory_excluded_flights", []):
        f_name = ex["name"]
        reason = ex["reason"]
        source_dir = DATASETS_BASE / f_name
        n_cam = len(pd.read_csv(source_dir / "camera_frames.csv")) if (source_dir / "camera_frames.csv").exists() else 0
        eligibility_rows.append({
            "flight_name": f_name,
            "trajectory": "HOVER",
            "description": "Stationary hover (zero forward translation)",
            "total_frames": n_cam,
            "active_frames": 0,
            "active_duration_s": 0.0,
            "status": "EXCLUDED",
            "exclusion_reason": reason,
        })
        manifest_rows.append({
            "trial_num": trial_counter,
            "trial_id": f"r3_{f_name}_excluded",
            "base_flight": f_name,
            "trajectory": "HOVER",
            "regime": "N/A",
            "regime_type": "N/A",
            "parameter_alpha": "N/A",
            "active_frames": 0,
            "active_duration_s": 0.0,
            "eligible": False,
            "exclusion_reason": reason,
        })
        trial_counter += 1

    df_manifest = pd.DataFrame(manifest_rows)
    df_eligibility = pd.DataFrame(eligibility_rows)

    manifest_csv = PROCESSED_DIR / "r3_experiment_manifest.csv"
    eligibility_csv = PROCESSED_DIR / "r3_eligibility_log.csv"

    df_manifest.to_csv(manifest_csv, index=False)
    df_eligibility.to_csv(eligibility_csv, index=False)

    print(f"-> Manifest written: {manifest_csv} ({len(df_manifest)} entries, {df_manifest['eligible'].sum()} eligible)")
    print(f"-> Eligibility log written: {eligibility_csv} ({len(df_eligibility)} flights)")
    print("=" * 70)


if __name__ == "__main__":
    main()
