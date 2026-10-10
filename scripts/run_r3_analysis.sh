#!/usr/bin/env bash
# Research 3 — Automated Execution Pipeline for Analysis, Evaluation, and Figures
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"

echo "======================================================================"
echo "Starting Research 3 Analysis Pipeline"
echo "======================================================================"

echo "[1/6] Extracting features and assembling K=5 dataset..."
$PYTHON_BIN "${REPO_DIR}/scripts/03_extract_features_and_labels.py" --legacy-label

echo "[2/6] Evaluating frozen Research 2 predictor transfer..."
$PYTHON_BIN "${REPO_DIR}/scripts/04_evaluate_frozen_r2_transfer.py"

echo "[3/6] Computing motion distribution comparison..."
$PYTHON_BIN "${REPO_DIR}/scripts/05_compute_motion_distribution_comparison.py"

echo "[4/6] Analyzing lead-time and early-warning dynamics..."
$PYTHON_BIN "${REPO_DIR}/scripts/06_analyze_lead_time_and_early_warnings.py"

echo "[5/6] Running statistical significance tests & hypothesis classification..."
$PYTHON_BIN "${REPO_DIR}/scripts/07_statistical_significance_tests.py"

echo "[6/6] Generating publication figures..."
$PYTHON_BIN "${REPO_DIR}/scripts/08_generate_r3_figures.py"

echo "======================================================================"
echo "Research 3 Analysis Pipeline Complete!"
echo "======================================================================"
