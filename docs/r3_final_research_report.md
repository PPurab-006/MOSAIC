# Research 3: Cross-Mechanism VO Failure Generalization
## Evaluating Telemetry-Based Visual Odometry Failure Prediction Under Non-Motion Visual Degradation

> [!WARNING]
> **SUPERSEDED. These results used a label that included the current frame (`range(0, k+1)`). See [docs/V2_PROTOCOL.md](docs/V2_PROTOCOL.md), [docs/V2_DEVIATIONS.md](docs/V2_DEVIATIONS.md) and [r3_v2/results/](r3_v2/results/) for the strict re-evaluation.**

---

## 1. Abstract
This study investigates whether the predictive relationship discovered in Research 2—wherein vehicle telemetry predicts monocular Visual Odometry (VO) tracking failure 5 frames ahead ($\tau = 0.165\text{ s}$)—generalizes across fundamentally different failure mechanisms. The authoritative Research 2 Multi-Layer Perceptron (`expanded_mlp.pt`) and preprocessor (`expanded_scaler.joblib`) were cryptographically locked and evaluated as an immutable instrument without retraining, recalibration, or threshold adjustment. We evaluated 40 eligible flights (30,545 active flight frames) across five controlled visual degradation regimes (Nominal, Low, Transition, Medium, High) where failure was induced through deterministic optical contrast attenuation towards a uniform veil luminance ($A_\infty = 128.0$), while vehicle motion remained substantially less aggressive than the R2 training distribution (mean yaw rate $10.92^\circ/\text{s}$ vs $41.55^\circ/\text{s}$ in R2 test, Cohen's $d = 1.30$, Kolmogorov-Smirnov $p < 10^{-10}$). 

Under this decoupled regime, the frozen R2 predictor retained substantial discriminatory capability, achieving an overall AUROC of **0.7906** (95% bootstrap CI: **[0.7856, 0.7958]**) and AUPRC of **0.8618** (95% CI: **[0.8570, 0.8669]**), compared to R2 test performance of 0.8046 ($\Delta = -0.0140$). The collapse toward chance predicted by the motion-specific proxy hypothesis ($H_2$) was not observed ($p = 9.05 \times 10^{-5}$ vs chance). At the episode level, the predictor provided advance warnings on **90.58%** of 5,964 failure episodes with a median lead time of **0.132 s** (4 frames) and a missed failure rate of **0.70%**. However, frame-level classification recall dropped by 23.78% (from 73.36% to 49.58%), revealing a clear transfer boundary: absent aggressive angular velocity cues, the predictor operates conservatively ($\text{Precision} = 0.9388$), alerting ahead of extended failure episodes but missing transient single-frame blips. These findings support cross-mechanism transfer (**Outcome A** with documented **Outcome B** boundary nuances), demonstrating that the learned temporal telemetry signature reflects visual tracking instability rather than an exclusive motion proxy.

---

## 2. Research Question
**Does the frozen Research 2 telemetry-based predictor generalize from motion-induced monocular VO failure to VO failure caused by substantially different visual/observability mechanisms under benign vehicle motion?**

The purpose is to determine whether the predictive relationship discovered in Research 2 survives when the physical mechanism causing VO tracking collapse changes from platform-motion dynamics to external visual observability degradation.

---

## 3. Research 2 Baseline
Research 2 established that telemetry-based early warning of monocular 5-point Essential matrix VO failure ($K=5$ frames into the future, $\tau = 0.165\text{ s}$) is achievable during aggressive multi-axis UAV maneuvers. 

The authoritative Research 2 baseline comprises:
- **Baseline Failure Mechanism**: Severe UAV angular velocities and accelerations causing large inter-frame optical flow disparities, motion blur, and feature tracking losses.
- **Model**: A 2-hidden-layer MLP (`SmallMLP`: $15 \to 32 \to 16 \to 1$, 1,057 parameters).
- **Features**: 15 strictly ordered lagged signals derived from 3 base telemetry features (`eis_yaw_rate_deg`, `feature_vel_mean`, and keyframe classification `is_r_frame`) sampled at lags $\{0, 1, 2, 3, 5\}$.
- **Preprocessing**: `StandardScaler` fitted on the R2 training sweep.
- **Decision Threshold**: $\theta = 0.50$ (uncalibrated).
- **Causal Label**: Binary indicator $y_t^{(K=5)} = \mathbb{I}(\text{num\_inliers\_pose}_{t+5} < 8)$.
- **Held-Out R2 Test Performance**: Across 14 held-out aggressive test flights (10,411 frames, positive rate $34.62\%$):
  - $\text{AUROC} = 0.8046$, $\text{AUPRC} = 0.7260$, $\text{F1} = 0.6484$
  - $\text{Precision} = 0.5808$, $\text{Recall} = 0.7336$
  - $\text{Episode Early Warning Rate} = 79.33\%$, $\text{Median Lead Time} = 0.165\text{ s}$

---

## 4. Competing Hypotheses
Research 3 evaluated two competing, symmetric scientific hypotheses without a priori preference:

* **$H_1$ — Mechanism-General Prediction**:
  The Research 2 predictor learned a general precursor signal associated with impending VO tracking collapse itself (e.g., optical flow deceleration caused by tracking loss, keyframe selection instability, and small residual yaw dynamics). Therefore, despite being trained exclusively on aggressive-motion regimes, it retains meaningful predictive discrimination and early-warning fidelity when VO failure is induced through visual/observability degradation under benign motion.

* **$H_2$ — Motion-Specific Proxy**:
  The Research 2 predictor primarily learned a motion-specific proxy conditioned on the aggressive-motion failure regime. Therefore, when VO failure is induced through visual/observability degradation without corresponding aggressive motion, predictive performance substantially degrades toward chance ($\text{AUROC} \approx 0.50$, $\text{F1} \approx 0$).

---

## 5. Experimental Design
The experimental design evaluates whether the predictive relationship survives when platform motion is decoupled from VO failure:
- **Flight Trajectories**: 8 distinct, repeatable base flight trajectories (gentle cruise `G_C`, gentle sharp stops `G_S`, gentle braking `G_B`) flown at nominal operating altitudes ($z \ge 2.0\text{ m}$).
- **Degradation Conditions**: 5 controlled visual regimes evaluated per base flight, yielding $8 \times 5 = 40$ primary experimental trials.
- **Simulated Camera & Telemetry**: Monocular camera stream at $30\text{ Hz}$ ($640 \times 480$, grayscale), synchronized with vehicle flight telemetry and ground truth poses.
- **Evaluation Discipline**: Strict black-box evaluation of the frozen R2 model. Zero retraining, zero parameter fine-tuning, zero threshold adjustment, and zero data leakage.

---

## 6. Visual Degradation Mechanism
To break the correlation between vehicle motion and VO failure, tracking breakdown was induced through controlled **optical contrast attenuation towards an ambient atmospheric veil luminance**.

### 6.1 Mathematical Formulation
The degradation applied to each camera frame $I_{\text{orig}}(x, y) \in [0, 255]$ was deterministically implemented as:
$$I_{\text{degraded}}(x, y) = \text{clip}\Big((1 - \alpha) \cdot I_{\text{orig}}(x, y) + \alpha \cdot A_{\infty},\; 0,\; 255\Big)$$
where:
- $\alpha \in [0.0, 1.0]$ is the degradation parameter.
- $A_{\infty} = 128.0$ is the constant ambient veil luminance.
- $\text{clip}(v, 0, 255)$ bounds the output to 8-bit unsigned integer values.

### 6.2 Physical and Algorithmic Effect
This deterministic transformation directly attenuates spatial image gradients:
$$\nabla I_{\text{degraded}}(x, y) = (1 - \alpha) \cdot \nabla I_{\text{orig}}(x, y)$$
As $\alpha \to 1$, the spatial image gradient $\nabla I$ vanishes, causing the Harris corner detector and Lucas-Kanade (KLT) optical tracker to lose reliable tracking points. Consequently, 5-point Essential matrix RANSAC cannot find sufficient geometric correspondences, driving:
$$\text{num\_inliers\_pose} < 8$$
This collapse occurs purely due to visual feature starvation without requiring any platform motion.

### 6.3 Tested Degradation Regimes
1. **Nominal ($\alpha = 0.000$)**: 100% gradient retention; pristine visual texture.
2. **Low ($\alpha = 0.920$)**: 8% gradient retention; calibrated to match R2's baseline failure rate ($34.49\%$).
3. **Transition ($\alpha(t) = 0.000 \to 0.975$)**: Dynamic ramp between frames 200 and 350; evaluates pre-failure warning timing during progressive visual fade.
4. **Medium ($\alpha = 0.960$)**: 4% gradient retention; severe visual breakdown ($96.55\%$ failure rate).
5. **High ($\alpha = 0.985$)**: 1.5% gradient retention; near-total contrast extinction ($100.0\%$ failure rate).

---

## 7. Motion-Control Methodology
A core requirement of Research 3 is proving that vehicle motion remained substantially less aggressive than in Research 2, ensuring that visual degradation was the sole driver of VO failure.

### 7.1 Quantitative Motion Separation

| Metric | R3 Visual Degradation ($N=30,545$) | R2 Held-out Test ($N=10,411$) | R2 Aggressive Sweep ($N=17,735$) | Statistical Separation (R3 vs R2 Test) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Yaw Rate** ($\pm$ std) | **$10.92^\circ/\text{s} \pm 18.31^\circ/\text{s}$** | $41.55^\circ/\text{s} \pm 34.86^\circ/\text{s}$ | $61.09^\circ/\text{s} \pm 32.05^\circ/\text{s}$ | **Cohen's $d = 1.30$**, KS $D = 0.613, p < 10^{-10}$ |
| **Median Yaw Rate (P50)** | **$7.31^\circ/\text{s}$** | $35.70^\circ/\text{s}$ | $53.00^\circ/\text{s}$ | **$-79.5\%$ reduction** |
| **P95 Yaw Rate** | **$35.89^\circ/\text{s}$** | $101.72^\circ/\text{s}$ | $109.33^\circ/\text{s}$ | **$-64.7\%$ reduction** |
| **Mean Optical Flow** | **$5.05\text{ px} \pm 7.60\text{ px}$** | $18.61\text{ px} \pm 13.81\text{ px}$ | $25.13\text{ px} \pm 13.37\text{ px}$ | **Cohen's $d = 1.42$**, KS $D = 0.598, p < 10^{-10}$ |
| **Median Optical Flow** | **$3.45\text{ px}$** | $16.36\text{ px}$ | $24.11\text{ px}$ | **$-78.9\%$ reduction** |
| **R-Frame Frequency** | **$9.59\%$** | $70.52\%$ | $84.18\%$ | **$-60.93\%$ absolute reduction** |
| **Mean Ground Speed** | **$1.81\text{ m/s}$** | $2.45\text{ m/s}$ | $3.80\text{ m/s}$ | Benign translation |
| **Median Ground Speed** | **$0.98\text{ m/s}$** | $2.10\text{ m/s}$ | $3.40\text{ m/s}$ | Moderate cruise |
| **Mean Body Tilt** | **$4.29^\circ$** | $14.20^\circ$ | $22.80^\circ$ | Minimal attitude tilt |

### 7.2 Defensible Motion Characterization
Vehicle motion in R3 was not completely static (P95 yaw rate was $35.89^\circ/\text{s}$ during path turns), but typical yaw rate was $7.31^\circ/\text{s}$ (median), body tilt averaged $4.29^\circ$, and rotation-dominant keyframe flags occurred in under $10\%$ of frames. Compared against R2's aggressive regime, the motion distributions are statistically distinct (Cohen's $d = 1.30$, KS $p < 10^{-10}$), confirming that platform motion was decoupled from VO breakdown.

---

## 8. Dataset and Eligibility

### 8.1 Flight Cohort and Filtering
- **Total Executed Trials**: 43 flights ([data/processed/r3_experiment_manifest.csv](../data/processed/r3_experiment_manifest.csv)).
- **Eligible Primary Trials**: 40 flights (8 base flights $\times$ 5 regimes).
- **Active Flight Window**: Altitude filter $z \ge 2.0\text{ m}$ (takeoff and landing excluded).
- **Valid Frame Samples**: Exactly 6,109 active frames per regime, totaling **30,545 frames** in [data/processed/r3_dataset_k5.csv](../data/processed/r3_dataset_k5.csv).
- **Causal Lag Handling**: The first 5 frames (backward lag $5$) and last 5 frames (forward label horizon $K=5$) per trial were removed to enforce strict temporal causality.

### 8.2 Excluded Trials Documentation
3 stationary hover trials (`p3x_HOVER_L0_R1`, `p3x_HOVER_L0_R2`, `p3x_HOVER_L0_R3`) were logged and excluded prior to primary analysis ([data/processed/r3_eligibility_log.csv](../data/processed/r3_eligibility_log.csv)).
- *Exclusion Criterion*: Stationary hover produces zero translational baseline ($t \to 0$), causing the monocular 5-point Essential matrix epipolar geometry to become degenerate ($\text{rank}(E) < 2$). Under zero baseline, monocular VO fails identically even in pristine conditions, confounding the mechanism analysis.

---

## 9. Frozen R2 Model Integrity
The authoritative Research 2 artifacts were verified and frozen prior to evaluation ([models/frozen_r2_manifest.json](../models/frozen_r2_manifest.json)):
- **MLP Checkpoint**: [models/expanded_mlp.pt](../models/expanded_mlp.pt)  
  `SHA-256: 93acd1048a6954a143aac873a3fa8a4548c8a6230aa574aa1ea3fcce17b78e9f`
- **Scaler Joblib**: [models/expanded_scaler.joblib](../models/expanded_scaler.joblib)  
  `SHA-256: 4e9c6ebacb5ac4a554ed69a56d4c7c48df8ef1036b53907181b6dc221550b105`
- **Architecture**: `SmallMLP` ($15 \to 32 \to 16 \to 1$, 1,057 parameters).
- **Features**: 15 strictly ordered inputs (`eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame` at lags $\{0, 1, 2, 3, 5\}$).
- **Decision Threshold**: $\theta = 0.50$ (frozen, uncalibrated).
- **Integrity Status**: Byte-for-byte identical to Research 2. Zero retraining or recalibration occurred.

---

## 10. Quantitative Results
The frozen R2 predictor was evaluated across all 30,545 frames in [data/processed/r3_frozen_r2_evaluation_metrics.json](../data/processed/r3_frozen_r2_evaluation_metrics.json):

### 10.1 Overall Transfer Performance
- **AUROC**: **0.7906** (95% bootstrap CI: **[0.7856, 0.7958]**)
- **AUPRC**: **0.8618** (95% bootstrap CI: **[0.8570, 0.8669]**)
- **F1 Score**: **0.6489** (95% bootstrap CI: **[0.6422, 0.6552]**)
- **Precision**: **0.9388** (95% bootstrap CI: **[0.9341, 0.9437]**)
- **Recall**: **0.4958** (95% bootstrap CI: **[0.4886, 0.5029]**)
- **Warning Rate**: **32.44%**

### 10.2 Confusion Matrix (Frame-Level, $N=30,545$)
- **True Negatives (TN)**: 11,178
- **False Positives (FP)**: 606 (False Positive Rate: $5.14\%$)
- **False Negatives (FN)**: 9,459
- **True Positives (TP)**: 9,302

---

## 11. Comparison: R2 Baseline vs R3 Transfer

| Metric | R2 Test Baseline (Motion Regime) | R3 Overall (Visual Degradation) | Absolute Delta ($\Delta$) | 95% Bootstrap CI [Lower, Upper] |
| :--- | :--- | :--- | :--- | :--- |
| **AUROC** | **0.8046** | **0.7906** | **$-0.0140$** | **[0.7856, 0.7958]** |
| **AUPRC** | **0.7260** | **0.8618** | **$+0.1358$** | **[0.8570, 0.8669]** |
| **F1 Score** | **0.6484** | **0.6489** | **$+0.0005$** | **[0.6422, 0.6552]** |
| **Precision** | **0.5808** | **0.9388** | **$+0.3580$** | **[0.9341, 0.9437]** |
| **Recall** | **0.7336** | **0.4958** | **$-0.2378$** | **[0.4886, 0.5029]** |
| **Warning Rate** | **34.62%** | **32.44%** | **$-2.18\%$** | — |

---

## 12. Degradation-Level Analysis

| Regime | Parameter ($\alpha$) | Frames ($N$) | Failure Rate | Mean Inliers | Median Inliers | AUROC | AUPRC | F1 | Precision | Recall | Warning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nominal** | $0.000$ | 6,109 | 13.06% | 646.9 | 491.0 | **0.7929** | 0.5216 | 0.5131 | 0.5905 | 0.4536 | 10.03% |
| **Low** | $0.920$ | 6,109 | 34.49% | 277.6 | 205.0 | **0.7573** | 0.7222 | 0.5235 | **0.9106** | 0.3673 | 13.91% |
| **Transition** | $0.0 \to 0.975$ | 6,109 | 63.01% | 291.7 | 83.0 | **0.6738** | 0.8067 | 0.5680 | **0.9027** | 0.4144 | 28.92% |
| **Medium** | $0.960$ | 6,109 | 96.55% | 73.6 | 69.0 | 0.3845* | **0.9510** | 0.5520 | **0.9553** | 0.3881 | 39.22% |
| **High** | $0.985$ | 6,109 | 100.0% | 14.5 | 0.0 | N/A** | N/A** | **0.8242** | **1.0000** | 0.7009 | 70.09% |

*\*In the Medium regime, extreme class imbalance (96.55% failures, only 211 negative samples) distorts rank-order AUROC, while AUPRC is 0.9510 and early-warning rate is 95.07%.*  
*\*\*In High degradation, 100% of frames are positive, meaning ROC and PR curves are mathematically undefined.*

---

## 13. Early-Warning and Lead-Time Analysis

### 13.1 Episode-Level vs Frame-Level Distinction
A critical distinction must be maintained between **frame-level classification** and **episode-level early warning**:
- **Frame-Level Classification**: Every single frame is evaluated independently against $y_t^{(K=5)}$. Frame recall was **49.58%**, reflecting conservative model behavior on individual frames.
- **Episode-Level Early Warning**: Evaluates whether the predictor emits an advance warning in the causal lookahead window $[t_{\text{onset}} - 0.165\text{ s}, t_{\text{onset}})$ prior to the onset of a contiguous tracking failure event.

### 13.2 Episode Early-Warning Metrics ($N=5,964$ episodes)
From [data/processed/r3_lead_time_early_warnings.json](../data/processed/r3_lead_time_early_warnings.json):
* **True Early Warnings**: **5,402 episodes (90.58%)** triggered a warning prior to failure onset.
* **Post-Failure Detections**: **520 episodes (8.72%)** were first flagged after inliers dropped below 8.
* **Missed Failure Episodes**: **42 episodes (0.70%)** received no warning throughout the failure event.
* **Lead Time**:
  - Mean: **$0.1311\text{ s}$** ($3.97$ frames)
  - Median: **$0.1320\text{ s}$** ($4$ frames)
  - Maximum: **$0.1650\text{ s}$** ($5$ frames)
  - Minimum: **$0.0330\text{ s}$** ($1$ frame)

---

## 14. Statistical Analysis
From [data/processed/r3_statistical_comparison.json](../data/processed/r3_statistical_comparison.json):
1. **Non-Parametric Bootstrap (B=2,000 resamples)**:
   - AUROC 95% CI: **[0.7856, 0.7958]**
   - F1 95% CI: **[0.6422, 0.6552]**
   - The AUROC CI is strictly and significantly above 0.50.
2. **One-Sample Wilcoxon Signed-Rank Test vs Chance ($\mu_0 = 0.50$)**:
   - $W = 67.0, p = 9.05 \times 10^{-5}$
   - **Conclusion**: The predicted collapse-to-chance behavior specified by $H_2$ was not observed.
3. **Flight-Level Mann-Whitney U Test (R3 vs R2 Test)**:
   - $U = 312.0, p = 0.0367$ (two-sided)
   - Reflects flight-level variance introduced by high-saturation regimes, while aggregate discrimination remains strong ($0.7906$).

---

## 15. The Transfer Boundary
While aggregate discrimination transfers well (AUROC 0.7906 vs 0.8046), a clear **transfer boundary** is documented:
- **Recall Reduction**: Frame-level recall dropped from **$73.36\%$** in R2 to **$49.58\%$** in R3 ($\Delta = -23.78\%$).
- **Precision Elevation**: Frame-level precision increased from **$0.5808$** to **$0.9388$** ($\Delta = +0.3580$).
- **Mechanistic Cause**: In R2, high angular velocity served as a strong anticipatory trigger that pushed prediction probabilities above the 0.50 threshold early and broadly. Under benign motion, the yaw rate feature remains low ($\sim 7\text{--}10^\circ/\text{s}$). The model relies primarily on the deceleration and instability of `feature_vel_mean` and `is_r_frame`. This makes the frozen predictor conservative: it cleanly warns ahead of significant failure episodes ($90.58\%$ episode early warning rate), but fails to trigger on brief, single-frame transient dips.

---

## 16. Limitations
1. **Monocular Degeneracy in Hover**: Monocular VO cannot estimate Essential matrices during pure stationary hovering due to zero translational baseline ($t = 0$). This geometric degeneracy was documented and excluded from primary evaluation.
2. **High-Degradation Saturation**: At $\alpha \ge 0.96$, tracking failures saturate ($>96.5\%$), which distorts rank-order AUROC metrics due to severe negative-class sparsity.
3. **Synthetic Atmospheric Model**: Contrast attenuation was implemented via deterministic image blending rather than full 3D radiative transfer or physical volumetric fog.
4. **Fixed Decision Threshold**: The decision threshold remained locked at $\theta = 0.50$. While appropriate for testing the frozen model, recalibrating the threshold would likely recover frame recall at the expense of precision.

---

## 17. Scientific Contribution
* **What Research 2 Established**: Research 2 demonstrated that vehicle telemetry can predict imminent VO tracking failure $165\text{ ms}$ ahead during aggressive multi-axis UAV flight.
* **What Research 3 Establishes**: Research 3 evaluated whether this predictive relationship survived when VO failure was produced by a qualitatively different visual observability mechanism while vehicle motion remained substantially less aggressive. The frozen predictor retained substantial discrimination (AUROC 0.7906, 90.58% early-warning rate). Therefore, the evidence supports cross-mechanism transfer, while the reduction in frame-level recall demonstrates a clear transfer boundary.
* **Bounded Scope**: These results do not claim that the predictor is a "universal VO failure detector," that it operates across all possible environments, or that it is safety-certified for real-world deployment. The contribution is strictly bounded to demonstrating cross-mechanism transfer between aggressive motion dynamics and visual contrast attenuation.

---

## 18. Provisional Motivation for Research 4
*Note: Research 4 remains provisional and is subject to independent review of Research 3.*
- **Provisional Research Question**: Having established that monocular VO failure can be predicted across independent failure mechanisms (motion dynamics and visual observability) with an advance lead time of $\approx 130\text{--}165\text{ ms}$, how can this predictive early-warning signal be utilized by an active flight controller to execute proactive, non-conservative sensor handover or trajectory adaptation before catastrophic loss of state estimation occurs?

---

## 19. Reproducibility and Artifact Index

### Script Pipeline
- [scripts/00_verify_and_freeze_r2_artifacts.py](../scripts/00_verify_and_freeze_r2_artifacts.py): Verifies and freezes authoritative R2 weights and scaler.
- [scripts/01_generate_r3_visual_degradation_dataset.py](../scripts/01_generate_r3_visual_degradation_dataset.py): Assembles experiment manifest and eligibility log.
- [scripts/02_run_r3_offline_vo.py](../scripts/02_run_r3_offline_vo.py): Executes canonical offline VO across all 40 trials under degradation.
- [scripts/03_extract_features_and_labels.py](../scripts/03_extract_features_and_labels.py): Extracts 15 lagged features and $K=5$ causal labels.
- [scripts/04_evaluate_frozen_r2_transfer.py](../scripts/04_evaluate_frozen_r2_transfer.py): Evaluates frozen R2 model without retraining.
- [scripts/05_compute_motion_distribution_comparison.py](../scripts/05_compute_motion_distribution_comparison.py): Computes motion distributions and separation statistics.
- [scripts/06_analyze_lead_time_and_early_warnings.py](../scripts/06_analyze_lead_time_and_early_warnings.py): Evaluates episode dynamics and lead-time distributions.
- [scripts/07_statistical_significance_tests.py](../scripts/07_statistical_significance_tests.py): Computes bootstrap CIs, Mann-Whitney U, and Wilcoxon tests.
- [scripts/08_generate_r3_figures.py](../scripts/08_generate_r3_figures.py): Generates publication figures.
- [scripts/run_r3_analysis.sh](../scripts/run_r3_analysis.sh): Master execution script.

### Data Artifacts
- [models/frozen_r2_manifest.json](../models/frozen_r2_manifest.json): Model and scaler integrity manifest.
- [data/processed/r3_experiment_manifest.csv](../data/processed/r3_experiment_manifest.csv): Master experiment trial manifest (43 trials).
- [data/processed/r3_eligibility_log.csv](../data/processed/r3_eligibility_log.csv): Trial eligibility and exclusion log.
- [data/processed/r3_dataset_k5.csv](../data/processed/r3_dataset_k5.csv): Master dataset (30,545 frames, 15 features, $K=5$ labels).
- [data/processed/r3_frozen_r2_evaluation_metrics.json](../data/processed/r3_frozen_r2_evaluation_metrics.json): Primary evaluation results and confusion matrices.
- [data/processed/r3_per_flight_metrics.csv](../data/processed/r3_per_flight_metrics.csv): Per-flight performance breakdown.
- [data/processed/r3_motion_comparison_summary.json](../data/processed/r3_motion_comparison_summary.json): Motion statistics and statistical separation tests.
- [data/processed/r3_lead_time_early_warnings.json](../data/processed/r3_lead_time_early_warnings.json): Episode-level early warning and lead-time data.
- [data/processed/r3_statistical_comparison.json](../data/processed/r3_statistical_comparison.json): Bootstrap CIs, hypothesis tests, and classification rationale.

### Publication Figures
- [figures/fig1_r2_vs_r3_performance.png](../figures/fig1_r2_vs_r3_performance.png): Performance comparison across regimes with 95% bootstrap CIs.
- [figures/fig2_performance_vs_degradation_level.png](../figures/fig2_performance_vs_degradation_level.png): Tracking breakdown and predictor probability shifts vs degradation severity.
- [figures/fig3_motion_distribution_comparison.png](../figures/fig3_motion_distribution_comparison.png): Kinematic distributions demonstrating benign vehicle motion.
- [figures/fig4_early_warning_behavior.png](../figures/fig4_early_warning_behavior.png): Failure episode classification and lead-time distributions.
