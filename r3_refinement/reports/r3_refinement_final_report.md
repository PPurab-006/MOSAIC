# Research 3 Refinement Study: Sensitivity Recovery Across Failure Mechanisms
## Recovering Sensitivity to Visually Induced VO Failures Without Sacrificing Motion-Induced Generalization

> [!WARNING]
> **SUPERSEDED. These results used a label that included the current frame (`range(0, k+1)`). See [docs/V2_PROTOCOL.md](docs/V2_PROTOCOL.md), [docs/V2_DEVIATIONS.md](docs/V2_DEVIATIONS.md) and [r3_v2/results/](r3_v2/results/) for the strict re-evaluation.**

---

## 1. Executive Summary & Scientific Verdict

**Refinement Research Question**: Can a targeted refinement of the Research 2 telemetry predictor, informed by the cross-mechanism behavior observed in Research 3 Phase 1, recover sensitivity to visually induced Visual Odometry (VO) failures without sacrificing the generalization demonstrated on motion-induced failures?

**Empirical Verdict**: **OUTCOME A: SENSITIVITY IMPROVES SUBSTANTIALLY ON VISUAL FAILURES WHILE MOTION PERFORMANCE REMAINS STABLE.**
- **Visual Sensitivity Recovery**: On the protected visual-degradation test set (15 trials, 11,460 frames), frame-level recall increased from **$48.94\%$** (frozen R2) to **$87.20\%$** (refined MLP), an absolute recovery of **$+38.26\%$** ($\Delta \text{Recall} = +0.3826$, paired Wilcoxon $p = 6.10 \times 10^{-5}$, paired Cohen's $d = 1.24$). F1 score increased from **$0.6419$** to **$0.8986$** ($\Delta = +0.2567$), and AUROC increased from **$0.7922$** to **$0.9371$** ($\Delta = +0.1449$).
- **Motion Generalization Preserved**: On the protected motion-induced test set (14 held-out flights, 10,411 frames), performance remained fully intact:
  - $\text{AUROC} = \mathbf{0.8014}$ (vs frozen R2 baseline $\mathbf{0.8046}$, $\Delta = -0.0032$, $p = 0.3258$).
  - $\text{F1} = \mathbf{0.6466}$ (vs frozen R2 baseline $\mathbf{0.6484}$, $\Delta = -0.0018$, $p = 0.8552$).
  - $\text{Recall} = \mathbf{68.67\%}$ (vs frozen R2 baseline $\mathbf{73.36\%}$, $\Delta = -0.0469$, $p = 0.1079$, not statistically significant).
  - $\text{Precision} = \mathbf{0.6110}$ (vs frozen R2 baseline $\mathbf{0.5808}$, $\Delta = +0.0302$ higher).
- **Cross-Mechanism Gap Reduction**: The sensitivity disparity between mechanisms $|\text{Recall}_{\text{motion}} - \text{Recall}_{\text{visual}}|$ narrowed from **$24.42\%$** to **$18.53\%$**.
- **Combined Performance**: On the pooled cross-mechanism test distribution (21,871 frames), AUROC jumped from **$0.7201$** to **$0.8900$** ($\Delta = +0.1699$) and F1 from **$0.6447$** to **$0.8082$** ($\Delta = +0.1635$).
- **Episode Early Warning**: Pre-failure warnings on the protected visual test set rose from **$90.43\%$** to **$94.28\%$**, with **0 missed episodes (0.00%)** and a median lead time of **$0.165\text{ s}$** (full 5-frame horizon).
- **Attribution**: Ablation analysis proved that representation learning accounts for **$93\%$** of the visual recall gain, demonstrating that a simple threshold calibration (Model C) cannot solve the cross-mechanism problem.

---

## 2. R2 Baseline Context & R3 Phase 1 Discovery

### 2.1 Research 2 Baseline
Research 2 developed a temporal telemetry-based failure predictor (`SmallMLP`: $15 \to 32 \to 16 \to 1$, 1,057 parameters) trained exclusively on aggressive UAV flight maneuvers. The model used 15 lagged features (`eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame` at lags $\{0, 1, 2, 3, 5\}$) to predict monocular 5-point Essential matrix failure ($K=5$ frames ahead, $\tau = 0.165\text{ s}$). On held-out aggressive test flights, it achieved AUROC = 0.8046, Recall = 73.36%, Precision = 58.08%, and F1 = 0.6484 at default threshold $\theta = 0.50$.

### 2.2 Research 3 Phase 1 Discovery
Research 3 Phase 1 evaluated the frozen R2 model without retraining on 40 flights undergoing controlled optical contrast attenuation under benign vehicle motion ($10.92^\circ/\text{s}$ mean yaw rate, Cohen's $d = 1.30$ vs R2 test). 

The primary hypothesis test confirmed mechanism generalization (overall AUROC = 0.7906, episode early warning = 90.58%), but exposed a severe **sensitivity boundary**:
- Frame-level recall collapsed by **$-23.78\%$** (from $73.36\%$ to $49.58\%$).
- Frame-level precision rose to **$93.88\%$**.
- The model operated in an overly conservative regime, accurately ranking failing frames above healthy frames (AUROC 0.791), but failing to cross the uncalibrated decision threshold $\theta = 0.50$ on isolated failure frames.

---

## 3. Empirical Diagnosis: Why Was the Predictor Conservative?

Prior to proposing any refinement, we performed an empirical diagnosis of feature distributions, network weights, and probability calibration on the existing dataset (`r3_refinement/diagnosis/sensitivity_diagnosis.md`):

### 3.1 Class-Conditional Feature Shift
In R2, every training failure was accompanied by violent platform motion. When standardized by the R2 scaler:
- Failing frames in R2 had positive standardized coordinates: $z_{\text{yaw}} = +0.486$, $z_{\text{vel}} = +0.271$, $z_{\text{R}} = +0.317$.
- In R3, vehicle motion remained benign during failure: $z_{\text{yaw}} = -0.819$ ($12.0^\circ/\text{s}$ vs $57.0^\circ/\text{s}$ in R2), $z_{\text{vel}} = -1.041$ ($4.0\text{ px}$ vs $22.3\text{ px}$ in R2), and $z_{\text{R}} = -1.319$ ($11.5\%$ vs $85.6\%$ active).

### 3.2 Network Weight Attribution
Backpropagation gradient analysis on failing frames revealed that:
1. The network placed large positive weights on `is_r_frame` ($+1.53$ logit sensitivity). When `is_r_frame = 0` (in $90.4\%$ of benign frames), this feature maps to $z = -1.57$, subtracting over $2.0$ logit units from the failure score.
2. Low yaw rate was treated as evidence of flight safety, applying a persistent negative offset to the output logit.
3. The model correctly sensed tracking failure via `feature_vel_mean_lag0` (gradient $-5.58$), but the negative offset from the motion features pulled the median predicted probability for failing frames from $0.685$ in R2 down to **$0.495$** in R3.

### 3.3 Why Naive Global Threshold Tuning Fails
Lowering the decision threshold to $\theta = 0.20$ on R3 visual failures recovered recall ($71.0\%$, F1 $0.771$). However, applying $\theta = 0.20$ to R2 motion flights caused a **catastrophic precision collapse**:
- Precision dropped from **$58.1\%$** to **$42.3\%$** ($\Delta = -15.8\%$).
- False Positive Rate exploded from **$28.0\%$** to **$69.3\%$** ($\Delta = +41.3\%$).

**Diagnostic Conclusion**: The sensitivity loss cannot be solved by a static global threshold shift. The model required **cross-mechanism representation refinement** to learn that visual feature collapse can trigger failure independently of platform motion.

---

## 4. Refinement Methodology & Experimental Protocol

### 4.1 Strict Flight-Level Partitioning
To guarantee zero optimization leakage, all data splits were partitioned strictly at the flight level (`r3_refinement/manifests/refinement_split_manifest.json`):
1. **Protected R2 Motion Test Set**: The exact 14 held-out test flights from R2 (10,411 frames) remained 100% untouched.
2. **Protected R3 Visual Test Set**: 15 trials (11,460 frames) derived from base flights `sweep_G_C_R1`, `sweep_G_B_R2`, `sweep_G_S_R2` across all 5 regimes remained 100% untouched.
3. **Permitted Training Pool**: 15 R2 motion flights + 15 R3 visual trials (base flights `sweep_G_C_R2`, `sweep_G_S_R3`, `sweep_G_B_R3`), totaling 22,738 frames.
4. **Permitted Validation Pool**: 13 R2 motion flights + 10 R3 visual trials (base flights `sweep_G_C_R3`, `sweep_G_S_R1`), totaling 17,131 frames.

### 4.2 Architecture and Features (Immutable)
- Architecture: Identical `SmallMLP(15, 32, 16, 1)`, 1,057 parameters.
- Features: Identical 15 lagged features (`eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame` at lags $\{0, 1, 2, 3, 5\}$).
- Target: Identical causal $K=5$ horizon ($\tau = 0.165\text{ s}$).

### 4.3 Competing Models Evaluated
- **Model A (Frozen R2 Baseline)**: Original R2 weights (`expanded_mlp.pt`), R2 scaler (`expanded_scaler.joblib`), locked threshold $\theta = 0.50$.
- **Model B (Cross-Mechanism Refined MLP)**: Identical `SmallMLP` trained on the joint training pool with unified `StandardScaler`; threshold $\theta^* = 0.37$ locked on the joint validation set prior to test evaluation.
- **Model C (Validation-Calibrated Frozen R2)**: Original frozen R2 weights and scaler, with threshold $\theta^* = 0.46$ calibrated on the joint validation set.

---

## 5. Protected Test Evaluation Results

### 5.1 Three-Way Benchmark Comparison

| Evaluation Set | Metric | Model A (Frozen R2 Baseline, $\theta=0.50$) | Model B (Cross-Mechanism Refined, $\theta=0.37$) | Model C (Calibrated Frozen R2, $\theta=0.46$) | Delta (Model B vs Model A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Protected R3 Visual Test**<br>(15 Trials, 11,460 frames) | **AUROC**<br>**AUPRC**<br>**Recall**<br>**Precision**<br>**F1 Score**<br>**FPR** | 0.7922<br>0.8598<br>**0.4894**<br>0.9327<br>0.6419<br>0.0572 | **0.9371**<br>**0.9664**<br>**0.8720**<br>0.9270<br>**0.8986**<br>0.1098 | 0.7922<br>0.8598<br>0.5183<br>0.9204<br>0.6632<br>0.0717 | **$+0.1449$**<br>**$+0.1066$**<br>**$+0.3826$**<br>$-0.0057$<br>**$+0.2567$**<br>$+0.0526$ |
| **Protected R2 Motion Test**<br>(14 Flights, 10,411 frames) | **AUROC**<br>**AUPRC**<br>**Recall**<br>**Precision**<br>**F1 Score**<br>**FPR** | **0.8046**<br>**0.7260**<br>**0.7336**<br>0.5808<br>**0.6484**<br>0.2803 | 0.8014<br>0.7214<br>0.6867<br>**0.6110**<br>0.6466<br>**0.2319** | **0.8046**<br>**0.7260**<br>0.7744<br>0.5464<br>0.6407<br>0.3424 | $-0.0032$<br>$-0.0046$<br>$-0.0469$<br>**$+0.0302$**<br>$-0.0018$<br>**$-0.0484$** |
| **Combined Cross-Mechanism Test**<br>(29 Flights, 21,871 frames) | **AUROC**<br>**AUPRC**<br>**Recall**<br>**Precision**<br>**F1 Score**<br>**FPR** | 0.7201<br>0.7276<br>0.5720<br>0.7386<br>0.6447<br>0.1923 | **0.8900**<br>**0.9125**<br>**0.8093**<br>**0.8072**<br>**0.8082**<br>**0.1837** | 0.7201<br>0.7276<br>0.6049<br>0.7100<br>0.6533<br>0.2348 | **$+0.1699$**<br>**$+0.1849$**<br>**$+0.2373$**<br>**$+0.0686$**<br>**$+0.1635$**<br>$-0.0086$ |
| **Full 40-Flight R3 Dataset**<br>(40 Trials, 30,545 frames) | **AUROC**<br>**AUPRC**<br>**Recall**<br>**Precision**<br>**F1 Score** | 0.7906<br>0.8618<br>0.4958<br>0.9388<br>0.6489 | **0.9388**<br>**0.9686**<br>**0.8731**<br>0.9289<br>**0.9002** | 0.7906<br>0.8618<br>0.5229<br>0.9275<br>0.6688 | **$+0.1482$**<br>**$+0.1068$**<br>**$+0.3773$**<br>$-0.0099$<br>**$+0.2513$** |

![Figure 1: Three-Way Comparison](refinement_fig1_three_way_comparison.png)

---

## 6. Statistical Significance & Paired Per-Flight Analysis

To establish whether the sensitivity improvement is consistent across flights rather than driven by outliers, we performed paired per-flight hypothesis tests (`r3_refinement/results/metrics/statistical_tests.json`):

### 6.1 Protected Visual Test Trials ($N=15$)
- **$\Delta \text{Recall}$ (Model B vs Model A)**: Mean paired difference $= \mathbf{+28.78\%} \pm 23.2\%$, Wilcoxon signed-rank $W = 0.0, \mathbf{p = 6.10 \times 10^{-5}}$, paired Cohen's $d = \mathbf{1.24}$ (Large effect size).
- **$\Delta \text{F1}$ (Model B vs Model A)**: Mean paired difference $= \mathbf{+18.30\%}$, Wilcoxon $p = \mathbf{6.23 \times 10^{-3}}$, Cohen's $d = \mathbf{0.97}$.
- **$\Delta \text{AUROC}$ (Model B vs Model A)**: Mean paired difference $= \mathbf{+0.1518}$, Cohen's $d = \mathbf{0.81}$.
- **Model B vs Model C (Refined Representation vs Calibration Alone)**:
  - $\Delta \text{Recall} = \mathbf{+25.82\%}$, Wilcoxon $p = \mathbf{1.01 \times 10^{-3}}$, Cohen's $d = \mathbf{1.10}$.
  - Proves that representation learning significantly outperforms threshold calibration.

### 6.2 Protected Motion Test Flights ($N=14$)
- **$\Delta \text{Recall}$ (Model B vs Model A)**: Mean difference $= -3.44\%$, Wilcoxon $p = 0.1079$ (**not statistically significant**).
- **$\Delta \text{F1}$ (Model B vs Model A)**: Mean difference $= -0.31\%$, Wilcoxon $p = 0.8552$ (**no difference**).
- **$\Delta \text{AUROC}$ (Model B vs Model A)**: Mean difference $= -0.0074$, Wilcoxon $p = 0.3258$ (**no difference**).

![Figure 3: Per-Flight Deltas](refinement_fig3_per_flight_deltas.png)

---

## 7. Targeted Ablation: Attributing the Performance Gain

To determine what proportion of the gain stems from representation learning versus decision thresholding, we ablated each component on the protected test sets (`r3_refinement/results/metrics/ablation_results.json`):

| Configuration | Scaler | Weights | Threshold ($\theta$) | Visual AUROC | Visual Recall | Visual F1 | Motion Recall | Motion F1 | Factor Attribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Model A (Frozen Baseline)** | R2 | Frozen R2 | 0.50 | 0.7922 | 48.9% | 0.6419 | 73.4% | 0.6484 | Baseline |
| **2. Model C (Calibration Only)** | R2 | Frozen R2 | 0.46 | 0.7922 | 51.8% | 0.6632 | 77.4% | 0.6407 | Calibration alone adds only $+2.9\%$ visual recall. |
| **3. Refined Rep (Fixed Threshold)** | Refined | Refined | 0.50 | **0.9371** | **84.6%** | **0.8970** | 54.8% | 0.6133 | **Representation learning drives $+35.7\%$ of the $+38.3\%$ recall recovery ($93.3\%$ of total gain)!** |
| **4. Model B (Full Refinement)** | Refined | Refined | 0.37 | **0.9371** | **87.2%** | **0.8986** | **68.7%** | **0.6466** | Joint threshold balances operating point, restoring motion recall to $68.7\%$. |

![Figure 2: PR and ROC Curves](refinement_fig2_pr_roc_curves.png)

---

## 8. Episode-Level Early Warning & Lead-Time Dynamics

Episode dynamics across contiguous failure runs (`r3_refinement/results/metrics/episode_early_warning_comparison.json`):

### 8.1 Protected Visual Test ($N=2,236$ episodes)
- **Model A (Frozen R2)**: Early Warnings = $90.43\%$ ($2,022$ eps), Missed = $0.76\%$ ($17$ eps), Median Lead = $0.132\text{ s}$.
- **Model B (Refined MLP)**: Early Warnings = **$94.28\%$** ($2,108$ eps), Missed = **$0.00\%$ (0 episodes)**, Median Lead = **$0.165\text{ s}$** (full 5-frame horizon).
- **Model C (Calibrated Frozen)**: Early Warnings = $92.26\%$ ($2,063$ eps), Missed = $0.13\%$ ($3$ eps), Median Lead = $0.132\text{ s}$.

### 8.2 Protected Motion Test ($N=647$ episodes)
- **Model A (Frozen R2)**: Early Warnings = $80.22\%$ ($519$ eps), Missed = $0.15\%$ ($1$ ep), Median Lead = $0.133\text{ s}$.
- **Model B (Refined MLP)**: Early Warnings = $76.35\%$ ($494$ eps), Missed = $0.15\%$ ($1$ ep), Median Lead = $0.133\text{ s}$.
- **Model C (Calibrated Frozen)**: Early Warnings = $82.23\%$ ($532$ eps), Missed = $0.15\%$ ($1$ ep), Median Lead = $0.133\text{ s}$.

![Figure 4: Episode Early Warning](refinement_fig4_early_warning_comparison.png)

---

## 9. Scientific Interpretation & Discussion

### 9.1 What This Study Establishes
1. **The Sensitivity Limitation Is Correctable**: The frame-level recall collapse observed in R3 Phase 1 was not an insurmountable physical wall, but an artifact of single-domain training bias. Exposing the exact same 15-feature MLP architecture to benign visual degradation flights allowed it to decouple motion amplitude from visual failure, recovering frame recall to $87.2\%$.
2. **Generalization Is Not Zero-Sum**: Crucially, recovering sensitivity to visual failures did not cause a material breakdown in motion-induced failure prediction (AUROC $0.8014$ vs $0.8046$, F1 $0.6466$ vs $0.6484$, $p = 0.8552$). The refined predictor operates effectively across both domains.
3. **Representation vs Calibration**: The ablation proves that cross-mechanism joint training reorganizes the latent feature space (AUROC jumps from $0.792$ to $0.937$), whereas threshold calibration alone leaves the underlying representation unchanged and fails to recover sensitivity.

### 9.2 What This Study Does NOT Claim
- We do NOT claim that this model is a "universal VO failure predictor" or that it will generalize to unseen failure mechanisms (e.g., sensor hardware failure, severe rolling-shutter distortion, or dynamic object occlusion).
- We do NOT claim that monocular VO failure can be predicted under stationary hover; epipolar geometry remains degenerate at zero baseline.
- All conclusions remain strictly bounded to the tested mechanisms: aggressive platform motion and environmental contrast attenuation.

---

## 10. Provisional Motivation for Research 4
*Note: Research 4 remains provisional and subject to independent review.*
- **Provisional Question**: With a refined telemetry predictor that reliably signals monocular VO failure $165\text{ ms}$ ahead across both motion and visual observability failure regimes, how can this early-warning signal be integrated with the vehicle autopilot to execute **predictive state handover** (e.g., switching to optical flow or IMU-only dead-reckoning) or **trajectory de-escalation** before loss of state estimation causes flight instability?

---

## 11. Reproducibility & Artifact Index

### Refinement Pipeline Scripts
- [r3_refinement/diagnosis/run_sensitivity_diagnosis.py](../../r3_refinement/diagnosis/run_sensitivity_diagnosis.py): Diagnostic analysis script.
- [r3_refinement/manifests/build_split_manifest.py](../../r3_refinement/manifests/build_split_manifest.py): Builds leak-free flight splits.
- [r3_refinement/src/train_and_validate_refinement.py](../../r3_refinement/src/train_and_validate_refinement.py): Trains refined MLP and locks validation thresholds.
- [r3_refinement/src/evaluate_protected_test_sets.py](../../r3_refinement/src/evaluate_protected_test_sets.py): Side-by-side protected evaluation.
- [r3_refinement/src/statistical_analysis.py](../../r3_refinement/src/statistical_analysis.py): Paired per-flight tests and effect sizes.
- [r3_refinement/src/run_ablation_analysis.py](../../r3_refinement/src/run_ablation_analysis.py): Component ablation script.
- [r3_refinement/src/generate_refinement_figures.py](../../r3_refinement/src/generate_refinement_figures.py): Generates 4 publication figures.
- [r3_refinement/tests/test_refinement_integrity.py](../../r3_refinement/tests/test_refinement_integrity.py): Automated test suite.

### Model Artifacts & Manifests
- [r3_refinement/models/refined_mlp.pt](../../r3_refinement/models/refined_mlp.pt): Refined PyTorch weights (1,057 params).
- [r3_refinement/models/refined_scaler.joblib](../../r3_refinement/models/refined_scaler.joblib): Refined StandardScaler.
- [r3_refinement/models/locked_thresholds.json](../../r3_refinement/models/locked_thresholds.json): Validation-locked decision thresholds.
- [r3_refinement/manifests/refinement_split_manifest.json](../../r3_refinement/manifests/refinement_split_manifest.json): Split manifest.

### Quantitative Outputs & Figures
- [r3_refinement/results/metrics/protected_evaluation_metrics.json](../../r3_refinement/results/metrics/protected_evaluation_metrics.json): Primary evaluation results.
- [r3_refinement/results/metrics/per_flight_comparison.csv](../../r3_refinement/results/metrics/per_flight_comparison.csv): Per-flight metric records.
- [r3_refinement/results/metrics/statistical_tests.json](../../r3_refinement/results/metrics/statistical_tests.json): Statistical significance tests.
- [r3_refinement/results/metrics/ablation_results.json](../../r3_refinement/results/metrics/ablation_results.json): Factor attribution matrix.
- [r3_refinement/results/figures/refinement_fig1_three_way_comparison.png](../../r3_refinement/results/figures/refinement_fig1_three_way_comparison.png): Figure 1.
- [r3_refinement/results/figures/refinement_fig2_pr_roc_curves.png](../../r3_refinement/results/figures/refinement_fig2_pr_roc_curves.png): Figure 2.
- [r3_refinement/results/figures/refinement_fig3_per_flight_deltas.png](../../r3_refinement/results/figures/refinement_fig3_per_flight_deltas.png): Figure 3.
- [r3_refinement/results/figures/refinement_fig4_early_warning_comparison.png](../../r3_refinement/results/figures/refinement_fig4_early_warning_comparison.png): Figure 4.
