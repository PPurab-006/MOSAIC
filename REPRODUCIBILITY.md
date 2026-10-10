# MOSAIC (Research 3) — Reproducibility & Verification Guide

This document specifies the exact environment, commands, cryptographic artifacts, and expected numerical outputs required to independently reproduce all findings and tables for release **v2.0.0**.

---

## 1. Cryptographic Checksums (SHA-256)

All core models and datasets are verified against their canonical SHA-256 digests:

| File Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `models/expanded_mlp.pt` | Frozen R2 Predictor Checkpoint (1,057 parameters) | `93acd1048a6954a143aac873a3fa8a4548c8a6230aa574aa1ea3fcce17b78e9f` |
| `models/expanded_scaler.joblib` | Frozen R2 StandardScaler (15 motion features) | `4e9c6ebacb5ac4a554ed69a56d4c7c48df8ef1036b53907181b6dc221550b105` |
| `r3_v2/data/r3_v2_strict_dataset.csv` | Full Strict prospective dataset (51,325 frames) | `99e17fc988e69ed1619378d5de3674836aecc8a66dd11380f48d224de65721b1` |

To verify checksums locally:
```bash
sha256sum -c <<EOF
93acd1048a6954a143aac873a3fa8a4548c8a6230aa574aa1ea3fcce17b78e9f  models/expanded_mlp.pt
4e9c6ebacb5ac4a554ed69a56d4c7c48df8ef1036b53907181b6dc221550b105  models/expanded_scaler.joblib
99e17fc988e69ed1619378d5de3674836aecc8a66dd11380f48d224de65721b1  r3_v2/data/r3_v2_strict_dataset.csv
EOF
```

---

## 2. Environment Setup

The repository uses Python 3.11 / 3.14 on Linux x86_64 with pinned dependencies in `requirements.txt`:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Reproduction Commands

### Command 1: Fast Table Reproduction & Numerical Verification
To reproduce all master tables from `r3_v2/data/r3_v2_strict_dataset.csv` and the saved model weights:
```bash
python scripts/reproduce_v2_tables.py
```
**Expected Outcome**: Evaluates Frozen R2, Model A, Model B, and Baselines C & D, comparing recomputed AUROCs with `r3_v2/results/comprehensive_model_comparison.csv`.
- Maximum absolute difference must be $< 10^{-4}$ (observed: $< 0.00005$).

### Command 2: Run Unit & Integrity Test Suites
```bash
pytest r3_v2/tests
```
**Expected Outcome**: All 11 tests pass in $< 2\text{ s}$, confirming:
- Strict forward window ($y_t = \max(F_{t+1}..F_{t+5})$)
- Purging of active failure frames ($F_t = 1$)
- Boundary frame drops (first 5 and last 5 frames dropped)
- Zero data leakage across train, val, and protected test splits
- Model architectures ($1,057$ parameters for 15-dim; $1,697$ parameters for 35-dim)

### Command 3: Full Analysis & Figure Generation Pipeline
```bash
bash scripts/run_r3_analysis.sh
```
**Expected Outcome**: Executes verification, feature extraction, distribution comparisons, statistical bootstrap ($B=2,000$), and figure generation with return code 0.

---

## 4. Expected Numerical Values

From `r3_v2/results/comprehensive_model_comparison.csv`:

### Macro-Average Across Seeds 42–46 (Mean ± Sample Std, $ddof=1$)
- **Model A (15 Motion Features)**:
  - Pooled Visual AUROC: $0.8798 \pm 0.0057$
  - Motion AUROC: $0.7556 \pm 0.0053$
- **Model B (35 Features: Motion + Image)**:
  - Pooled Visual AUROC: $0.9345 \pm 0.0014$
  - Motion AUROC: $0.7749 \pm 0.0052$
- **Frozen R2 Baseline**:
  - Pooled Visual AUROC: $0.6545 \pm 0.0000$
  - Motion AUROC: $0.7665 \pm 0.0000$
- **Baseline C (MLP: 20 Image Features Only)**:
  - Pooled Visual AUROC: $0.9085 \pm 0.0010$
  - Motion AUROC: $0.6513 \pm 0.0053$
- **Baseline D (MLP: Regime ID + $\alpha$)**:
  - Pooled Visual AUROC: $0.9029 \pm 0.0014$
  - Motion AUROC: $0.5000 \pm 0.0001$

### Visual Per-Flight Ranges ($n=3$)
Across the 3 visual base flights (`sweep_G_C_R1`, `sweep_G_B_R2`, `sweep_G_S_R2`):
- **Frozen R2**:
  - Nominal: $0.7661$ [$0.7420, 0.7886$]
  - Low: $0.6935$ [$0.5814, 0.7676$]
  - Transition: $0.4711$ [$0.4561, 0.4787$]
  - Medium: $0.2724$ [$0.1025, 0.4022$]
- **Model B**:
  - Nominal: $0.7473$ [$0.7205, 0.7786$]
  - Low: $0.6816$ [$0.5956, 0.7270$]
  - Transition: $0.9690$ [$0.9656, 0.9729$]
  - Medium: $0.7159$ [$0.6996, 0.7452$]

### Meaningful Paired Cluster Bootstrap Differences ($B=1,000$, Seed 42)
- **Model B vs. Model A ($B - A$)**:
  - Pooled Visual: $+0.0526$ [$+0.0433, +0.0592$]
  - Transition: $+0.0407$ [$+0.0267, +0.0470$]
  - Medium: $+0.0974$ [$-0.0076$, $+0.1818$] (suggestive)
- **Model B vs. Baseline D ($B - D(\text{MLP})$)**:
  - Pooled Visual: $+0.0330$ [$+0.0249, +0.0430$]
  - Transition: $+0.0281$ [$+0.0174$, $+0.0356$]

---

## 5. Scope of Dependencies

- **Table & Metric Reproduction**: Fully self-contained within this repository using `r3_v2/data/r3_v2_strict_dataset.csv`.
- **Raw Data Generation**: Synthesizing optical degradation frames from raw ROS bags and rerunning offline VO requires access to sibling folders `../Research2` (base flight bags) and `../ROS` (calibration profiles and VO nodes).
