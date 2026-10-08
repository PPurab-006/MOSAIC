# Research 3: Cross-Mechanism VO Failure Generalization

This repository contains the complete experimental campaign, datasets, evaluation pipelines, and scientific report for **Research 3: Cross-Mechanism Visual Odometry Failure Generalization**.

---

## 🎯 Executive Overview

Research 3 addresses a fundamental scientific question arising from Research 2:
> **Does the frozen Research 2 telemetry-based predictor generalize from motion-induced monocular VO failure to VO failure caused by substantially different visual/observability mechanisms under benign vehicle motion?**

### The Competing Scientific Hypotheses:
- **$H_1$ — Mechanism-General Prediction**: The R2 predictor learned an intrinsic precursor signal of VO failure that transfers across physical mechanisms, retaining discrimination and early-warning fidelity under visual degradation with benign motion.
- **$H_2$ — Motion-Specific Proxy**: The R2 predictor learned a motion-specific proxy conditioned on aggressive maneuvers and collapses toward chance ($\text{AUROC} \approx 0.50$) under non-motion failure.

### 🏆 Scientific Outcome: **A. MECHANISM-GENERAL PREDICTION SUPPORTED**
*(With documented regime-boundary nuances aligning with **B. PARTIAL MECHANISM TRANSFER** regarding frame-level recall)*
- **AUROC Transfer**: Frozen R2 MLP achieves **$\text{AUROC} = 0.7906$** ($95\%$ CI: $[0.7856, 0.7958]$) across 30,545 test frames, compared to R2 test baseline **$0.8046$** ($\Delta = -0.0140$).
- **Hypothesis Test Outcome**: The predicted collapse-to-chance behavior specified by $H_2$ was not observed ($W = 67.0, p = 9.05 \times 10^{-5}$ via one-sample Wilcoxon signed-rank test vs chance).
- **Early-Warning Efficacy**: Successfully warns prior to failure on **$90.58\%$** of 5,964 breakdown episodes (vs $79.33\%$ in R2) with a median lead time of **$0.132\text{ s}$** ($4$ frames) and only **$0.70\%$** missed failures.
- **The Transfer Boundary**: Frame-level classification recall dropped by $23.78\%$ (from $73.36\%$ to $49.58\%$), while precision elevated to $0.9388$. In the absence of aggressive angular velocity cues, the predictor operates conservatively.
- **Empirical Motion Decoupling**: Vehicle yaw rate was substantially less aggressive than R2 ($\bar{\omega}_z = 10.92^\circ/\text{s}$ vs $41.55^\circ/\text{s}$ in R2 test, Cohen's $d = 1.30$, KS $p < 10^{-10}$), confirming that failure was driven by visual degradation.

---

## 🔒 Frozen Research 2 Artifacts

The authoritative Research 2 model weights, scaler, and feature specifications were cryptographically locked and evaluated strictly as a black-box instrument:

| Artifact | Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **PyTorch MLP Checkpoint** | `models/expanded_mlp.pt` | `93acd1048a6954a143aac873a3fa8a4548c8a6230aa574aa1ea3fcce17b78e9f` |
| **StandardScaler Joblib** | `models/expanded_scaler.joblib` | `4e9c6ebacb5ac4a554ed69a56d4c7c48df8ef1036b53907181b6dc221550b105` |
| **Integrity Manifest** | `models/frozen_r2_manifest.json` | Recorded and verified in Phase 1 |

- **Features (15 strictly ordered lagged signals)**: `eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame` at lags $\{0, 1, 2, 3, 5\}$.
- **Prediction Target**: Monocular 5-point Essential matrix pose inliers $< 8$ at horizon $K=5$ ($\tau = 0.165\text{ s}$).
- **Decision Threshold**: $\theta = 0.50$ (uncalibrated, frozen).

---

## 🔬 Visual Degradation Mechanism (As Executed)

Visual tracking failure was induced through deterministic optical contrast attenuation towards a uniform ambient veil luminance:
$$I_{\text{degraded}}(x, y) = \text{clip}\Big((1 - \alpha) \cdot I_{\text{orig}}(x, y) + \alpha \cdot A_{\infty},\; 0,\; 255\Big)$$
with $A_{\infty} = 128.0$.

This transformation uniformly scales spatial gradients:
$$\nabla I_{\text{degraded}} = (1 - \alpha) \cdot \nabla I_{\text{orig}}$$
starving the Lucas-Kanade (KLT) tracker of valid corner features and driving Essential matrix inlier collapse ($\text{num\_inliers\_pose} < 8$) without platform motion excitation.

---

## 📊 Summary of Empirical Results

### Quantitative Metrics Comparison

| Metric | R2 Test Baseline (Motion Regime) | R3 Overall (Visual Degradation) | Absolute Delta ($\Delta$) | 95% Bootstrap Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **AUROC** | 0.8046 | **0.7906** | $-0.0140$ | [0.7856, 0.7958] |
| **AUPRC** | 0.7260 | **0.8618** | $+0.1358$ | [0.8570, 0.8669] |
| **F1 Score** | 0.6484 | **0.6489** | $+0.0005$ | [0.6422, 0.6552] |
| **Precision** | 0.5808 | **0.9388** | $+0.3580$ | [0.9341, 0.9437] |
| **Recall** | 0.7336 | **0.4958** | $-0.2378$ | [0.4886, 0.5029] |
| **Early Warning Rate** | 79.33% | **90.58%** | $+11.25\%$ | — |
| **Missed Failure Rate** | — | **0.70%** | — | — |
| **Median Lead Time** | 0.165 s | **0.132 s** | $-0.033\text{ s}$ | — |

### Kinematic Motion Decoupling

| Metric | R3 Visual Degradation | R2 Test Baseline | R2 Aggressive Sweep | Statistical Separation |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Yaw Rate** | **$10.92^\circ/\text{s}$** | $41.55^\circ/\text{s}$ | $61.09^\circ/\text{s}$ | **Cohen's $d = 1.30$**, KS $p < 10^{-10}$ |
| **Median Yaw Rate** | **$7.31^\circ/\text{s}$** | $35.70^\circ/\text{s}$ | $53.00^\circ/\text{s}$ | **$-79.5\%$ reduction** |
| **P95 Yaw Rate** | **$35.89^\circ/\text{s}$** | $101.72^\circ/\text{s}$ | $109.33^\circ/\text{s}$ | **$-64.7\%$ reduction** |
| **Mean Optical Flow** | **$5.05\text{ px}$** | $18.61\text{ px}$ | $25.13\text{ px}$ | **Cohen's $d = 1.42$**, KS $p < 10^{-10}$ |
| **R-Frame Frequency** | **$9.59\%$** | $70.52\%$ | $84.18\%$ | **$-60.93\%$ absolute drop** |
| **Mean Ground Speed** | **$1.81\text{ m/s}$** | $2.45\text{ m/s}$ | $3.80\text{ m/s}$ | Benign translation |

---

## 📈 Visualizations

The experimental analysis generated four publication figures:
- **`figures/fig1_r2_vs_r3_performance.png`**: Cross-mechanism metric comparison (AUROC, AUPRC, F1, Precision, Recall) across all regimes.
- **`figures/fig2_performance_vs_degradation_level.png`**: Inlier tracking breakdown and model prediction distributions across degradation severity $\alpha \in [0.0, 0.985]$.
- **`figures/fig3_motion_distribution_comparison.png`**: Empirical verification of benign vehicle motion separating R3 from R2.
- **`figures/fig4_early_warning_behavior.png`**: Failure episode segmentation, pre-failure warning timing, and lead-time distributions.

---

## 📁 Repository Structure

```
Research3/
├── configs/
│   └── r3_experiment_config.yaml         # Master configuration for R3 degradation and evaluation
├── data/
│   ├── processed/
│   │   ├── r3_experiment_manifest.csv    # 43 trials with trajectory, seed, and degradation parameters
│   │   ├── r3_eligibility_log.csv        # Detailed eligibility and exclusion documentation
│   │   ├── r3_dataset_k5.csv             # 30,545 active flight frames with 15 lagged features & K=5 labels
│   │   ├── r3_frozen_r2_evaluation_metrics.json # Full evaluation results, confusion matrices, regime metrics
│   │   ├── r3_per_flight_metrics.csv     # Individual per-flight performance (AUROC, F1, Recall)
│   │   ├── r3_motion_statistics.csv      # Per-flight kinematic statistics
│   │   ├── r3_motion_comparison_summary.json # Motion separation statistical tests (Cohen's d, KS test)
│   │   ├── r3_lead_time_early_warnings.json  # Pre-failure warning and lead-time summary
│   │   ├── r3_episode_details.csv        # 5,964 individual failure episode details
│   │   └── r3_statistical_comparison.json# 2,000-bootstrap CIs, Mann-Whitney U, Wilcoxon tests
├── docs/
│   └── r3_final_research_report.md       # Comprehensive scientific research report
├── figures/
│   ├── fig1_r2_vs_r3_performance.png
│   ├── fig2_performance_vs_degradation_level.png
│   ├── fig3_motion_distribution_comparison.png
│   └── fig4_early_warning_behavior.png
├── models/
│   ├── expanded_mlp.pt                   # Frozen R2 PyTorch weights
│   ├── expanded_scaler.joblib            # Frozen R2 StandardScaler
│   └── frozen_r2_manifest.json           # Cryptographic hash and architecture verification manifest
├── scripts/
│   ├── 00_verify_and_freeze_r2_artifacts.py
│   ├── 01_generate_r3_visual_degradation_dataset.py
│   ├── 02_run_r3_offline_vo.py
│   ├── 03_extract_features_and_labels.py
│   ├── 04_evaluate_frozen_r2_transfer.py
│   ├── 05_compute_motion_distribution_comparison.py
│   ├── 06_analyze_lead_time_and_early_warnings.py
│   ├── 07_statistical_significance_tests.py
│   ├── 08_generate_r3_figures.py
│   └── run_r3_analysis.sh                # End-to-end evaluation & figure reproduction runner
└── README.md                             # Repository overview and summary
```

---

## 🔁 Reproduction Instructions

To execute the end-to-end evaluation, statistical analysis, and figure generation pipeline:

```bash
# 1. Activate project environment
source .venv/bin/activate

# 2. Verify frozen R2 artifact hashes
python scripts/00_verify_and_freeze_r2_artifacts.py

# 3. Run full statistical evaluation and figure generation
bash scripts/run_r3_analysis.sh
```

All figures will be output to `figures/` and metrics to `data/processed/`.

---

## 📄 Full Report
For comprehensive methodology, physical degradation modeling, regime breakdowns, and discussion of implications for UAV flight control, see [docs/r3_final_research_report.md](docs/r3_final_research_report.md).
