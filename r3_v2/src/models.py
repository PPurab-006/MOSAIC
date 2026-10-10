#!/usr/bin/env python3
"""
Research 3 (MOSAIC) — V2 Model Architecture & Training Utilities.
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SmallMLP(nn.Module):
    """Authoritative SmallMLP failure predictor architecture."""
    def __init__(self, input_dim: int = 15, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.input_dim = input_dim
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
    """Sets random seeds deterministically across all frameworks."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    seed: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 10,
) -> Tuple[SmallMLP, Dict]:
    """Trains SmallMLP with early stopping monitored on validation loss."""
    set_seed(seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SmallMLP(input_dim=input_dim, hidden1=32, hidden2=16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    best_epoch = 1

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_weights)
    model.eval()

    train_history = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "total_epochs": epoch,
    }
    return model, train_history


def find_best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> Dict:
    """Finds optimal decision threshold on validation split by maximizing F1 score."""
    threshold_grid = np.arange(0.05, 0.95, 0.01)
    best_th = 0.50
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for th in threshold_grid:
        preds = (probs >= th).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            best_prec = float(precision_score(y_true, preds, zero_division=0))
            best_rec = float(recall_score(y_true, preds, zero_division=0))

    auroc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "threshold": round(best_th, 2),
        "val_f1": round(best_f1, 4),
        "val_precision": round(best_prec, 4),
        "val_recall": round(best_rec, 4),
        "val_auroc": round(auroc, 4),
        "val_auprc": round(auprc, 4),
    }
