# Research 3 Refinement: Diagnostic Analysis of the Sensitivity Loss

---

## 1. Context and Problem Statement
In Research 3 Phase 1, the frozen Research 2 telemetry-based Visual Odometry (VO) failure predictor (`models/expanded_mlp.pt`, `models/expanded_scaler.joblib`) was evaluated without retraining on 40 flights undergoing controlled visual contrast degradation under benign platform motion.

While the primary hypothesis test confirmed that the learned relationship transferred across physical mechanisms (overall AUROC = **0.7906**, matching R2 test baseline **0.8046** within $\Delta = -0.0140$; early-warning rate on failure episodes = **90.58%**), it revealed a pronounced **sensitivity asymmetry**:
- **Frame-Level Recall**: Dropped from **$73.36\%$** in R2 to **$49.58\%$** in R3 ($\Delta = -23.78\%$).
- **Frame-Level Precision**: Elevated from **$58.08\%$** in R2 to **$93.88\%$** in R3 ($\Delta = +35.80\%$).
- **F1 Score**: Remained essentially identical (**$0.6489$** in R3 vs **$0.6484$** in R2).

The model did not fail to detect visual degradation episodes, but it operated in an excessively conservative regime at the frame level. The purpose of this diagnostic study is to uncover the exact physical and mathematical mechanism responsible for this conservatism prior to proposing any refinement.

---

## 2. Empirical Findings: Why is the Predictor Conservative?

### 2.1 Class-Conditional Feature Distribution Shift
We compared the standardized feature representations $Z = (X - \mu_{\text{scaler}}) / \sigma_{\text{scaler}}$ across failing frames ($y=1$) between the R2 held-out motion test set and the R3 visual degradation dataset:

| Feature | R2 Failing Frames ($y=1$) | R3 Failing Frames ($y=1$) | Difference ($\Delta z$) | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `eis_yaw_rate_deg_lag0` | $z = +0.486$ ($57.0^\circ/\text{s}$) | $z = -0.819$ ($12.0^\circ/\text{s}$) | $-1.305\sigma$ | Benign vehicle motion suppresses yaw rate. |
| `eis_yaw_rate_deg_lag5` | $z = +0.481$ ($56.8^\circ/\text{s}$) | $z = -0.822$ ($11.9^\circ/\text{s}$) | $-1.303\sigma$ | Persistent low yaw rate across lag window. |
| `feature_vel_mean_lag0` | $z = +0.271$ ($22.3\text{ px}$) | $z = -1.041$ ($4.0\text{ px}$) | $-1.312\sigma$ | Optical tracking collapse reduces tracked corner velocity. |
| `is_r_frame_lag0` | $z = +0.317$ ($85.6\%$ active) | $z = -1.319$ ($11.5\%$ active) | $-1.636\sigma$ | Gentle flight rarely triggers rotation keyframes. |

In the training distribution of Research 2, every failure was driven by aggressive platform motion, causing high yaw rates, high optical flow velocities, and dense rotation keyframe generation (`is_r_frame = 1`). Consequently, the model's feature space shifted by over **$-1.3$ to $-1.6$ standard deviations** along every single input axis under visual degradation.

### 2.2 Network Weight and Gradient Attribution
Analysis of the frozen MLP weights ($15 \to 32 \to 16 \to 1$) and backpropagated input gradients $\nabla_X \text{logit}$ reveals how the network responds to this shift:
1. **Rotation Keyframe Bias**: The effective linear attribution for `is_r_frame` at lags $\{0, 1, 3, 5\}$ is strongly positive ($+1.42, +0.96, +0.85, +0.14$). In benign flight, `is_r_frame = 0` ($90.4\%$ of frames), which maps to $z = -1.57$ under the R2 scaler. Multiplying by positive weights, the absence of rotation keyframes subtracts over $2.0$ logit units from the failure logit.
2. **Yaw Rate Bias**: The network learned that low angular velocity indicates healthy, stable flight. In R3, yaw rates are small ($10^\circ/\text{s}$), reinforcing the network's belief that flight is safe.
3. **Decoupled Optical Flow Response**: The network's primary signal for visual degradation failure is `feature_vel_mean_lag0` (input gradient $\approx -5.58$). While the sharp drop in feature velocity correctly ranks failure samples above non-failure samples (yielding AUROC = $0.7906$), the massive negative offset from the motion features pushes the entire output distribution downwards.

### 2.3 Output Probability Shift
Because of this negative logit offset:
- In R2 test, failing frames have **Mean $p = 0.660$**, **Median $p = 0.685$** (only $26.6\%$ below $0.50$).
- In R3, failing frames have **Mean $p = 0.431$**, **Median $p = 0.495$** (**$50.4\%$ below $0.50$**).
- For non-failing frames in R3, the probabilities are very low (**Mean $p = 0.186$**, **Median $p = 0.130$**).

Thus, exactly half ($50.4\%$) of all failing frames in R3 hover in the narrow interval $[0.20, 0.50)$, just below the fixed decision threshold $\theta = 0.50$.

### 2.4 Why Global Threshold Tuning Fails
If we simply lower the global decision threshold $\theta$ to $0.20$:
- In R3: Frame recall jumps from $49.6\%$ to **$71.0\%$** (matching R2 baseline), with F1 increasing from $0.649$ to **$0.771$**.
- In R2: Applying $\theta = 0.20$ to the motion test set causes a **catastrophic precision collapse**:
  - Precision drops from **$58.1\%$** to **$42.3\%$** ($\Delta = -15.8\%$).
  - False Positive Rate explodes from **$28.0\%$** to **$69.3\%$** ($\Delta = +41.3\%$).
  - Over two-thirds of healthy frames in aggressive flight trigger false failure warnings!

This proves conclusively that:
> **The sensitivity loss in R3 cannot be resolved by a naive global threshold adjustment.**
> Lowering the threshold to rescue visual recall destroys motion-regime precision.

---

## 3. Root Cause Classification

We classify the observed behavior into the taxonomy requested:
1. **Class-Conditional Distribution Shift (Primary)**: In aggressive flight, VO failure is correlated with high angular velocity. In visual degradation, VO failure occurs under low angular velocity. The model learned a class-conditional relationship that assumes high motion is a prerequisite for failure.
2. **Representation Conditioning (Secondary)**: The single-domain training distribution of R2 caused the MLP to treat motion features as gatekeepers for failure, rather than learning a multi-cue representation where visual tracking decay alone is sufficient to declare failure when motion is benign.
3. **Threshold Calibration Mismatch (Tertiary)**: While threshold tuning can shift the operating point within a single domain, a static threshold cannot reconcile the two mechanisms without cross-mechanism training.

---

## 4. Scientifically Justified Refinement Proposal

To resolve this limitation without methodology drift, we formulate the **minimal, scientifically justified refinement**:

### Model Architecture and Features (Immutable)
- Architecture: Identical `SmallMLP(15, 32, 16, 1)` (1,057 parameters).
- Features: Identical 15 lagged features (`eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame` at lags $\{0, 1, 2, 3, 5\}$).
- Target: Identical causal $K=5$ horizon ($\tau = 0.165\text{ s}$).

### The Refinement: Cross-Mechanism Joint Training (Model B)
- Train the `SmallMLP` on the pooled **permitted training flights** from both mechanisms:
  - 15 R2 motion-induced training flights.
  - 15 R3 visually degraded training trials (derived strictly from base flights `sweep_G_C_R2`, `sweep_G_S_R3`, `sweep_G_B_R3`).
- Fit a unified `StandardScaler` on the combined training distribution.
- Optimize using standard binary cross-entropy loss with Adam optimizer.
- Select the decision threshold $\theta^*$ exclusively on the **permitted validation flights**:
  - 13 R2 motion validation flights.
  - 10 R3 visual validation trials (derived strictly from base flights `sweep_G_C_R3`, `sweep_G_S_R1`).
- Keep **Protected Test Sets (14 R2 test flights + 15 R3 test trials)** 100% untouched until final evaluation.

### The Essential Baseline: Validation-Calibrated Frozen R2 (Model C)
- Evaluate whether calibrating the threshold on the joint validation set using Platt scaling / validation tuning on the frozen R2 model can achieve sensitivity recovery without retraining.

---

## 5. Success Criteria Pre-Registration
A successful refinement must satisfy:
1. **Sensitivity Recovery**: Frame-level recall on the protected R3 visual test set increases from $\approx 50\%$ towards $\ge 65\%$.
2. **Motion Generalization Preservation**: Performance on the protected R2 motion test set does not collapse (AUROC $\ge 0.78$, Recall $\ge 68\%$, Precision $\ge 52\%$, F1 $\ge 0.60$).
3. **Cross-Mechanism Gap Reduction**: The sensitivity gap $|\text{Recall}_{\text{motion}} - \text{Recall}_{\text{visual}}|$ is substantially reduced.
