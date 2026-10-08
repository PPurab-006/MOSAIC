#!/usr/bin/env python3
"""
Research 3 — Step 6: Temporal Lead-Time and Early-Warning Episode Dynamics.

Executes episode-level temporal analysis identical to Research 2:
  1. Identifies contiguous failure episodes (runs where num_inliers_pose < 8).
  2. Evaluates warning timing relative to episode onset t_onset:
     - True Early Warning: warning (prob >= 0.5) occurred prior to t_onset (t_warn < t_onset)
       within the causal lookahead horizon [t_onset - 0.165s, t_onset].
     - Post-Failure Detection: first warning occurred only after failure onset (t_warn >= t_onset).
     - Missed Failure: no warning emitted before or during the episode.
  3. Measures lead time: Delta t_lead = t_onset - t_first_warning (seconds).
  4. Directly compares R3 early-warning behavior against R2 baseline.

Outputs:
  - data/processed/r3_lead_time_early_warnings.json
  - data/processed/r3_episode_details.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

R3_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = R3_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "r3_dataset_k5.csv"
FROZEN_MANIFEST_PATH = R3_ROOT / "models" / "frozen_r2_manifest.json"
METRICS_JSON_PATH = PROCESSED_DIR / "r3_frozen_r2_evaluation_metrics.json"

OUT_WARNINGS_JSON = PROCESSED_DIR / "r3_lead_time_early_warnings.json"
OUT_EPISODES_CSV = PROCESSED_DIR / "r3_episode_details.csv"


def extract_episodes_from_trial(df_trial: pd.DataFrame, max_horizon_s: float = 0.165) -> list[dict]:
    """Extract contiguous failure episodes and analyze warning arrival timing."""
    t = df_trial["timestamp_total_sec"].values.astype(float)
    y_raw = df_trial["is_failure_current_frame"].values.astype(int)
    probs = df_trial["pred_prob"].values.astype(float)
    warn = (probs >= 0.5).astype(int)

    n = len(df_trial)
    episodes = []
    in_episode = False
    start_idx = 0

    for i in range(n):
        if y_raw[i] == 1 and not in_episode:
            in_episode = True
            start_idx = i
        elif y_raw[i] == 0 and in_episode:
            in_episode = False
            end_idx = i - 1
            episodes.append((start_idx, end_idx))
    if in_episode:
        episodes.append((start_idx, n - 1))

    episode_records = []
    for ep_id, (s_idx, e_idx) in enumerate(episodes, 1):
        t_onset = t[s_idx]
        t_end = t[e_idx]
        ep_duration = t_end - t_onset
        ep_length_frames = e_idx - s_idx + 1

        # Search pre-failure lookahead window: [t_onset - max_horizon_s, t_onset)
        pre_mask = (t >= (t_onset - max_horizon_s - 1e-4)) & (t < t_onset)
        pre_warn_indices = np.where(pre_mask & (warn == 1))[0]

        # Search intra-episode window: [t_onset, t_end]
        intra_mask = (t >= t_onset) & (t <= t_end)
        intra_warn_indices = np.where(intra_mask & (warn == 1))[0]

        if len(pre_warn_indices) > 0:
            first_warn_idx = pre_warn_indices[0]
            t_warn = t[first_warn_idx]
            lead_time = t_onset - t_warn
            status = "EARLY_WARNING"
        elif len(intra_warn_indices) > 0:
            first_warn_idx = intra_warn_indices[0]
            t_warn = t[first_warn_idx]
            lead_time = -(t_warn - t_onset)
            status = "POST_FAILURE_DETECTION"
        else:
            first_warn_idx = -1
            t_warn = float("nan")
            lead_time = float("nan")
            status = "MISSED_FAILURE"

        episode_records.append({
            "trial_id": df_trial["trial_id"].iloc[0],
            "base_flight": df_trial["base_flight"].iloc[0],
            "trajectory": df_trial["trajectory"].iloc[0],
            "regime": df_trial["regime"].iloc[0],
            "episode_num": ep_id,
            "start_frame": int(df_trial["frame_idx"].iloc[s_idx]),
            "end_frame": int(df_trial["frame_idx"].iloc[e_idx]),
            "length_frames": int(ep_length_frames),
            "t_onset_s": round(float(t_onset), 4),
            "t_end_s": round(float(t_end), 4),
            "duration_s": round(float(ep_duration), 4),
            "classification": status,
            "t_warn_s": round(float(t_warn), 4) if not np.isnan(t_warn) else None,
            "lead_time_s": round(float(lead_time), 4) if not np.isnan(lead_time) else None,
        })

    return episode_records


def main():
    print("=" * 70)
    print("Research 3 — Step 6: Temporal Lead-Time & Early-Warning Analysis")
    print("=" * 70)

    # 1. Load predictions
    df = pd.read_csv(DATASET_PATH)

    # Load frozen model & generate pred_prob if not saved in CSV
    if "pred_prob" not in df.columns:
        import joblib
        import torch
        import torch.nn as nn

        class SmallMLP(nn.Module):
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

        scaler = joblib.load(R3_ROOT / "models" / "expanded_scaler.joblib")
        model = SmallMLP(15, 32, 16)
        model.load_state_dict(torch.load(R3_ROOT / "models" / "expanded_mlp.pt", map_location="cpu"))
        model.eval()

        with open(FROZEN_MANIFEST_PATH, "r", encoding="utf-8") as f:
            feat_cols = json.load(f)["feature_specification"]["feature_names_ordered"]

        X_sc = scaler.transform(df[feat_cols].values.astype(float))
        with torch.no_grad():
            df["pred_prob"] = torch.sigmoid(model(torch.tensor(X_sc, dtype=torch.float32))).numpy()

    # 2. Extract episodes per trial
    all_episodes = []
    for trial_id, group in df.groupby("trial_id", sort=False):
        ep_list = extract_episodes_from_trial(group, max_horizon_s=0.165)
        all_episodes.extend(ep_list)

    df_ep = pd.DataFrame(all_episodes)
    df_ep.to_csv(OUT_EPISODES_CSV, index=False)

    # 3. Aggregate Early Warning Metrics
    total_episodes = len(df_ep)
    early_episodes = df_ep[df_ep["classification"] == "EARLY_WARNING"]
    post_episodes = df_ep[df_ep["classification"] == "POST_FAILURE_DETECTION"]
    missed_episodes = df_ep[df_ep["classification"] == "MISSED_FAILURE"]

    early_count = len(early_episodes)
    post_count = len(post_episodes)
    missed_count = len(missed_episodes)

    early_pct = round(early_count / total_episodes * 100.0, 2) if total_episodes > 0 else 0.0
    post_pct = round(post_count / total_episodes * 100.0, 2) if total_episodes > 0 else 0.0
    missed_pct = round(missed_count / total_episodes * 100.0, 2) if total_episodes > 0 else 0.0

    lead_times = early_episodes["lead_time_s"].dropna().values
    mean_lead = round(float(np.mean(lead_times)), 4) if len(lead_times) > 0 else 0.0
    median_lead = round(float(np.median(lead_times)), 4) if len(lead_times) > 0 else 0.0
    min_lead = round(float(np.min(lead_times)), 4) if len(lead_times) > 0 else 0.0
    max_lead = round(float(np.max(lead_times)), 4) if len(lead_times) > 0 else 0.0

    # 4. Per-Regime Breakdown
    regime_breakdown = {}
    for regime in ["NOMINAL", "LOW", "MEDIUM", "HIGH", "TRANSITION"]:
        r_ep = df_ep[df_ep["regime"] == regime]
        r_tot = len(r_ep)
        r_early = len(r_ep[r_ep["classification"] == "EARLY_WARNING"])
        r_post = len(r_ep[r_ep["classification"] == "POST_FAILURE_DETECTION"])
        r_miss = len(r_ep[r_ep["classification"] == "MISSED_FAILURE"])
        r_leads = r_ep.loc[r_ep["classification"] == "EARLY_WARNING", "lead_time_s"].dropna().values

        regime_breakdown[regime] = {
            "total_episodes": r_tot,
            "early_warnings": r_early,
            "early_warning_rate_pct": round(r_early / r_tot * 100.0, 2) if r_tot > 0 else 0.0,
            "post_failure_detections": r_post,
            "post_detection_rate_pct": round(r_post / r_tot * 100.0, 2) if r_tot > 0 else 0.0,
            "missed_failures": r_miss,
            "missed_failure_rate_pct": round(r_miss / r_tot * 100.0, 2) if r_tot > 0 else 0.0,
            "median_lead_time_s": round(float(np.median(r_leads)), 4) if len(r_leads) > 0 else 0.0,
            "mean_lead_time_s": round(float(np.mean(r_leads)), 4) if len(r_leads) > 0 else 0.0,
        }

    # 5. Load R2 Causal Replay Baseline
    r2_early_pct = 79.33  # 522/658 in R2 causal streaming replay
    r2_median_lead = 0.165

    delta_early_pct = round(early_pct - r2_early_pct, 2)
    delta_median_lead = round(median_lead - r2_median_lead, 4)

    summary_payload = {
        "overall_r3_early_warning_summary": {
            "total_failure_episodes": total_episodes,
            "early_warnings": early_count,
            "early_warning_rate_pct": early_pct,
            "post_failure_detections": post_count,
            "post_failure_rate_pct": post_pct,
            "missed_failures": missed_count,
            "missed_failure_rate_pct": missed_pct,
            "lead_time_s": {
                "mean": mean_lead,
                "median": median_lead,
                "min": min_lead,
                "max": max_lead,
            },
        },
        "r2_baseline_comparison": {
            "r2_early_warning_rate_pct": r2_early_pct,
            "r3_early_warning_rate_pct": early_pct,
            "early_warning_rate_delta": delta_early_pct,
            "r2_median_lead_time_s": r2_median_lead,
            "r3_median_lead_time_s": median_lead,
            "median_lead_time_delta_s": delta_median_lead,
        },
        "regime_breakdown": regime_breakdown,
    }

    with open(OUT_WARNINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("=" * 70)
    print("EARLY-WARNING & LEAD-TIME RESULTS")
    print("=" * 70)
    print(f"Total Failure Episodes Analyzed: {total_episodes}")
    print(f"R3 True Early Warning Rate:      {early_pct:.2f}% ({early_count}/{total_episodes})")
    print(f"R2 Baseline Early Warning Rate:  {r2_early_pct:.2f}% (522/658)")
    print(f"Early Warning Rate Delta:        {delta_early_pct:+.2f}%")
    print(f"R3 Post-Failure Detection Rate:  {post_pct:.2f}% ({post_count}/{total_episodes})")
    print(f"R3 Missed Failure Rate:          {missed_pct:.2f}% ({missed_count}/{total_episodes})")
    print(f"R3 Median Lead Time:             {median_lead:.4f} s (Mean: {mean_lead:.4f} s)")
    print("-" * 70)
    print("Regime Breakdown:")
    for reg, rm in regime_breakdown.items():
        print(f"  {reg:12s} | Episodes: {rm['total_episodes']:3d} | Early Warn: {rm['early_warning_rate_pct']:5.1f}% | Missed: {rm['missed_failure_rate_pct']:5.1f}% | Med Lead: {rm['median_lead_time_s']:.3f}s")
    print("=" * 70)
    print(f"-> Early warning metrics saved: {OUT_WARNINGS_JSON}")
    print(f"-> Episode details saved: {OUT_EPISODES_CSV}")


if __name__ == "__main__":
    main()
