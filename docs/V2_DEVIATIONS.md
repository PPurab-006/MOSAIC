# Research 3 (MOSAIC) — V2 Protocol Deviations & Engineering Audit Log

This document records the exact sequence of implementation details, reporting differences, and aggregation methods between evaluation runs of the V2 leakage-free pipeline.

---

## 1. Commit Timeline & Verification Disclaimer

> [!IMPORTANT]
> **Git Commit Ordering Notice**:
> The primary evaluation script `r3_v2/src/train_and_evaluate_v2.py` was committed to git as commit `9394a40` *after* the initial test set evaluation had already completed. Because git tracking began post-hoc, git history alone cannot cryptographically verify execution order or state that "no changes occurred" prior to or between initial evaluation runs. This document explicitly clarifies the modifications, aggregation choices, and methodological differences across runs.

---

## 2. Chronological Sequence of Pipeline Execution

### Run 1: Primary Model Benchmark (Model A, Model B, Frozen R2)
1. **Strict Dataset Assembly**:
   - `r3_v2/data/r3_v2_strict_dataset.csv` (51,325 frames across 82 sequences) generated under strict prospective window $y_t = \max(F_{t+1}..F_{t+5})$ (`range(1, k+1)`).
   - Purged all frames where VO failure was already active ($F_t = 1$).
   - Dropped the first 5 and last 5 frames of each trial.
2. **Training & Validation Calibration**:
   - Model A (15 lagged motion features) and Model B (35 features: 15 motion + 20 image-quality metrics) trained across 5 seeds (42, 43, 44, 45, 46).
   - Validation split (13 R2 flights + 10 R3 trials) used exclusively to tune decision threshold $\theta^*$ via $\max F_1$.
3. **Protected Test Evaluation & Reporting**:
   - Evaluated on protected test set (14 R2 flights + 15 R3 trials).
   - Metrics reported in `test_metrics_summary.csv`, `v2_summary.json`, and `comprehensive_model_comparison.csv` as the **macro-average across seeds** (mean ± sample std, $ddof=1$):
     - **Model A**: Visual AUROC = $0.8798 \pm 0.0057$, Motion AUROC = $0.7556 \pm 0.0053$
     - **Model B**: Visual AUROC = $0.9345 \pm 0.0014$, Motion AUROC = $0.7749 \pm 0.0052$
     - **Frozen R2**: Visual AUROC = $0.6545$, Motion AUROC = $0.7665$
     *(Note on stds: Population std $ddof=0$ yields $\pm 0.0051$ and $\pm 0.0012$; sample std $ddof=1$ yields $\pm 0.0057$ and $\pm 0.0014$. All master tables are harmonized to sample std $ddof=1$.)*
4. **Pre-Evaluation Script Edit (Crash Fix) & Git Commit**:
   - Prior to the completed run, an edit was made to `train_and_evaluate_v2.py` to fix a multiprocessing CPU oversubscription deadlock by adding `cv2.setNumThreads(1)` and worker thread caps.
   - Committed `train_and_evaluate_v2.py` as `9394a40` following completion of this first run.

---

### Run 2: Extension with Baselines C & D, Transition Ablation, and Aggregation Discrepancy
1. **Baselines C and D**:
   - Baseline C (20 image-quality features only; no yaw, flow, or `is_r_frame`) trained with MLP and Logistic Regression.
   - Baseline D (Regime ID one-hot + $\alpha$ for Transition frames) trained with MLP and Logistic Regression.
   - Script `r3_v2/src/evaluate_baselines_c_d.py` was introduced to house these additional baselines without modifying the committed `train_and_evaluate_v2.py`.
2. **Aggregation Discrepancy**:
   - In `evaluate_baselines_c_d.py`, the reporting table `comprehensive_model_comparison.csv` computed AUROC on the **soft-voting ensemble probabilities**:
     $$\bar{p} = \frac{1}{5} \sum_{s=1}^5 p_{\text{seed}_s}$$
   - This ensemble soft-voting slightly reduces per-frame prediction variance, shifting AUROC upward:
     - **Model A**: $0.8798 \to 0.8838$ (+0.0040)
     - **Model B**: $0.9345 \to 0.9365$ (+0.0020)
     - **Model A Motion**: $0.7557 \to 0.7605$ (+0.0048)
     - **Model B Motion**: $0.7749 \to 0.7808$ (+0.0059)
   - These numbers were reported without an explicit statement distinguishing seed macro-averaging from soft-voting ensembling, resulting in an apparent discrepancy between `test_metrics_summary.csv` and `comprehensive_model_comparison.csv`.
3. **Transition Ramp Ablation**:
   - Added `frame_idx_norm` as a 36th feature to Model B. Model B without time (35 feats) achieved $0.9690$ AUROC in Transition, while Model B + time achieved $0.9664$ ($\Delta = -0.0026$), verifying that Model B directly tracks visual degradation rather than elapsed time.

---

## 3. Reconciliation of Aggregation Methods

Both aggregation methods are mathematically valid but answer distinct questions:

| Model | Macro-Average Across Seeds (Mean ± Std) | Soft-Voting Ensemble $\text{AUROC}(y, \bar{p})$ | Difference ($\text{Ens} - \text{Macro}$) |
| :--- | :---: | :---: | :---: |
| **Model A (Visual AUROC)** | $0.8798 \pm 0.0057$ | $0.8838$ | $+0.0040$ |
| **Model A (Motion AUROC)** | $0.7557 \pm 0.0053$ | $0.7605$ | $+0.0048$ |
| **Model B (Visual AUROC)** | $0.9345 \pm 0.0014$ | $0.9365$ | $+0.0020$ |
| **Model B (Motion AUROC)** | $0.7749 \pm 0.0052$ | $0.7808$ | $+0.0059$ |
| **Baseline C (MLP Visual)** | $0.9085 \pm 0.0011$ | $0.9097$ | $+0.0012$ |
| **Baseline C (MLP Motion)** | $0.6513 \pm 0.0054$ | $0.6571$ | $+0.0058$ |
| **Baseline D (MLP Visual)** | $0.9029 \pm 0.0014$ | $0.9035$ | $+0.0006$ |
| **Baseline D (MLP Motion)** | $0.5000 \pm 0.0001$ | $0.5000$ | $+0.0000$ |

**Canonical Definition Selection**:
All primary summary tables are standardized to the **Macro-Average Across Seeds (Mean ± Std across seeds 42–46)** to reflect single-model operational deployment expectations. Soft-voting ensemble numbers are documented explicitly for full transparency.

---

## 4. Bootstrap Resampling Unit: Trial-Level vs. Base-Flight Cluster

In the initial run, the bootstrap resampled 15 visual trials with replacement. However:
- The 15 visual trials are generated by applying 5 degradation regimes to only **3 physical base flights** (`sweep_G_C_R1`, `sweep_G_B_R2`, `sweep_G_S_R2`).
- Resampling trials independently ignores cross-regime correlation within the same underlying flight.
- Correct hierarchical cluster bootstrap resamples the **3 physical base flights** with replacement (size 3, $B=1000$, seed 42) for visual evaluation, and the 14 held-out test flights for motion evaluation.
- All per-trial AUROCs across each regime (3 values per regime) are explicitly published to provide complete visibility into the cluster-level data.
