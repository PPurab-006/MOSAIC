#!/usr/bin/env python3
"""
Research 3 Refinement — Step 3: Train Refined Model and Select Validation Thresholds.

STRICT CONSTRAINTS:
- Uses ONLY permitted training flights and validation flights.
- Protected R2 test flights and Protected R3 test trials are NOT loaded or accessed here.
- Fits a unified StandardScaler on joint training data.
- Trains SmallMLP (15 -> 32 -> 16 -> 1) with early stopping on validation loss.
- Determines optimal validation threshold for Model B and Model C on the joint validation split.
- Locks all models, scalers, and thresholds BEFORE test evaluation.

Outputs:
  - r3_refinement/models/refined_mlp.pt
  - r3_refinement/models/refined_scaler.joblib
  - r3_refinement/models/locked_thresholds.json
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

R3_ROOT = Path(__file__).resolve().parent.parent.parent
REFINEMENT_DIR = R3_ROOT / "r3_refinement"
MODELS_DIR = REFINEMENT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = REFINEMENT_DIR / "configs" / "refinement_config.yaml"
MANIFEST_PATH = REFINEMENT_DIR / "manifests" / "refinement_split_manifest.json"

DATASET_R2 = R3_ROOT.parent / "Research2" / "data" / "processed" / "expanded_frames_v2_k5.csv"
DATASET_R3 = R3_ROOT / "data" / "processed" / "r3_dataset_k5.csv"

FROZEN_MODEL_PATH = R3_ROOT / "models" / "expanded_mlp.pt"
FROZEN_SCALER_PATH = R3_ROOT / "models" / "expanded_scaler.joblib"


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


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 70)
    print("RESEARCH 3 REFINEMENT: TRAINING & VALIDATION CALIBRATION")
    print("=" * 70)

    set_seed(42)

    # 1. Load split manifest
    with open(MANIFEST_PATH, "r") as f:
        split_manifest = json.load(f)

    r2_train_flights = set(split_manifest["r2_motion_mechanism"]["train_flights"])
    r2_val_flights = set(split_manifest["r2_motion_mechanism"]["val_flights"])
    r3_train_trials = set(split_manifest["r3_visual_mechanism"]["train_trials"])
    r3_val_trials = set(split_manifest["r3_visual_mechanism"]["val_trials"])

    with open(R3_ROOT / "models" / "frozen_r2_manifest.json") as f:
        feat_cols = json.load(f)["feature_specification"]["feature_names_ordered"]

    # 2. Load permitted R2 training & validation data
    df_r2 = pd.read_csv(DATASET_R2)
    df_r2_train = df_r2[df_r2["run_dir"].isin(r2_train_flights)][feat_cols + ["is_failure_within_next_k"]].copy()
    df_r2_val = df_r2[df_r2["run_dir"].isin(r2_val_flights)][feat_cols + ["is_failure_within_next_k"]].copy()

    # 3. Load permitted R3 training & validation data
    df_r3 = pd.read_csv(DATASET_R3)
    df_r3_train = df_r3[df_r3["trial_id"].isin(r3_train_trials)][feat_cols + ["is_failure_within_next_k"]].copy()
    df_r3_val = df_r3[df_r3["trial_id"].isin(r3_val_trials)][feat_cols + ["is_failure_within_next_k"]].copy()

    # 4. Assemble joint training and validation sets
    df_train = pd.concat([df_r2_train, df_r3_train], axis=0, ignore_index=True)
    df_val = pd.concat([df_r2_val, df_r3_val], axis=0, ignore_index=True)

    print(f"Permitted Training Pool:   {len(df_train)} samples (R2 motion: {len(df_r2_train)}, R3 visual: {len(df_r3_train)})")
    print(f"  Positive rate (train):   {df_train['is_failure_within_next_k'].mean()*100:.2f}%")
    print(f"Permitted Validation Pool: {len(df_val)} samples (R2 motion: {len(df_r2_val)}, R3 visual: {len(df_r3_val)})")
    print(f"  Positive rate (val):     {df_val['is_failure_within_next_k'].mean()*100:.2f}%")

    # 5. Fit unified StandardScaler exclusively on joint training data
    scaler_refined = StandardScaler()
    X_train_raw = df_train[feat_cols].values.astype(float)
    y_train = df_train["is_failure_within_next_k"].values.astype(float)

    X_train_scaled = scaler_refined.fit_transform(X_train_raw)

    X_val_raw = df_val[feat_cols].values.astype(float)
    y_val = df_val["is_failure_within_next_k"].values.astype(float)
    X_val_scaled = scaler_refined.transform(X_val_raw)

    # Save refined scaler
    refined_scaler_path = MODELS_DIR / "refined_scaler.joblib"
    joblib.dump(scaler_refined, refined_scaler_path)
    print(f"-> Saved unified scaler: {refined_scaler_path}")

    # 6. Setup PyTorch DataLoaders
    batch_size = 256
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 7. Model Training: SmallMLP (15 -> 32 -> 16 -> 1)
    model_refined = SmallMLP(15, 32, 16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_refined.parameters(), lr=0.001, weight_decay=1e-4)

    max_epochs = 50
    patience = 10
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    print("\nTraining Refined SmallMLP on Joint Multi-Mechanism Pool...")
    for epoch in range(1, max_epochs + 1):
        model_refined.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model_refined(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_dataset)

        model_refined.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model_refined(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model_refined.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or patience_counter >= patience:
            print(f"  Epoch {epoch:02d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    # Save refined model weights
    refined_model_path = MODELS_DIR / "refined_mlp.pt"
    model_refined.load_state_dict(best_weights)
    torch.save(best_weights, refined_model_path)
    print(f"-> Saved refined model: {refined_model_path}")

    # 8. Evaluate on Joint Validation Set and Lock Optimal Thresholds
    model_refined.eval()
    with torch.no_grad():
        val_logits_refined = model_refined(torch.tensor(X_val_scaled, dtype=torch.float32))
        val_probs_refined = torch.sigmoid(val_logits_refined).numpy()

    # Also evaluate Frozen R2 model on the Joint Validation Set using its own frozen scaler
    scaler_frozen = joblib.load(FROZEN_SCALER_PATH)
    model_frozen = SmallMLP(15, 32, 16)
    model_frozen.load_state_dict(torch.load(FROZEN_MODEL_PATH, map_location="cpu"))
    model_frozen.eval()

    X_val_frozen_scaled = scaler_frozen.transform(X_val_raw)
    with torch.no_grad():
        val_logits_frozen = model_frozen(torch.tensor(X_val_frozen_scaled, dtype=torch.float32))
        val_probs_frozen = torch.sigmoid(val_logits_frozen).numpy()

    # Search for optimal threshold maximizing F1 on joint validation set
    threshold_grid = np.arange(0.05, 0.95, 0.01)

    def find_best_threshold(y_true, probs):
        best_th = 0.50
        best_f1 = 0.0
        best_prec = 0.0
        best_rec = 0.0
        for th in threshold_grid:
            preds = (probs >= th).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
                best_prec = float(precision_score(y_true, preds, zero_division=0))
                best_rec = float(recall_score(y_true, preds, zero_division=0))
        return {
            "threshold": round(best_th, 2),
            "val_f1": round(best_f1, 4),
            "val_precision": round(best_prec, 4),
            "val_recall": round(best_rec, 4),
            "val_auroc": round(float(roc_auc_score(y_true, probs)), 4),
            "val_auprc": round(float(average_precision_score(y_true, probs)), 4),
        }

    th_refined = find_best_threshold(y_val, val_probs_refined)
    th_frozen_calibrated = find_best_threshold(y_val, val_probs_frozen)

    locked_config = {
        "model_a_frozen_r2": {
            "model_path": str(FROZEN_MODEL_PATH),
            "scaler_path": str(FROZEN_SCALER_PATH),
            "threshold": 0.50,
            "description": "Original Frozen Research 2 Predictor at standard 0.50 threshold"
        },
        "model_b_refined_mlp": {
            "model_path": str(refined_model_path),
            "scaler_path": str(refined_scaler_path),
            "threshold": th_refined["threshold"],
            "val_metrics": th_refined,
            "description": "Cross-Mechanism Refined MLP trained on joint pool with validation-locked threshold"
        },
        "model_c_calibrated_frozen_r2": {
            "model_path": str(FROZEN_MODEL_PATH),
            "scaler_path": str(FROZEN_SCALER_PATH),
            "threshold": th_frozen_calibrated["threshold"],
            "val_metrics": th_frozen_calibrated,
            "description": "Original Frozen R2 Model with threshold calibrated on joint validation pool"
        }
    }

    locked_path = MODELS_DIR / "locked_thresholds.json"
    with open(locked_path, "w") as f:
        json.dump(locked_config, f, indent=2)

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD SELECTION RESULTS (LOCKED PRIOR TO TEST):")
    print("=" * 70)
    print(f"Model A (Frozen R2 Baseline):            Threshold = 0.50 (Fixed Baseline)")
    print(f"Model B (Cross-Mechanism Refined MLP):  Threshold = {th_refined['threshold']:.2f} (Val F1: {th_refined['val_f1']:.4f}, Rec: {th_refined['val_recall']:.4f}, Prec: {th_refined['val_precision']:.4f})")
    print(f"Model C (Validation-Calibrated Frozen):  Threshold = {th_frozen_calibrated['threshold']:.2f} (Val F1: {th_frozen_calibrated['val_f1']:.4f}, Rec: {th_frozen_calibrated['val_recall']:.4f}, Prec: {th_frozen_calibrated['val_precision']:.4f})")
    print("=" * 70)
    print(f"-> Locked thresholds saved to: {locked_path}")


if __name__ == "__main__":
    main()
