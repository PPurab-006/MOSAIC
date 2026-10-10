# MOSAIC (Research 3) — V2 Leakage-Free Experimental Protocol

**Document Status:** LOCKED PRIOR TO EXECUTION  
**Version:** 2.0.0  
**Date:** 2026-10-10  
**Target Repository:** PPurab-006/MOSAIC (Research 3)

---

## 1. Executive Summary & Objective

The objective of this protocol is to establish a rigorous, leakage-free benchmark (v2) for cross-mechanism visual odometry (VO) failure prediction. In the initial v1 experiments, the forward-looking failure window target $y_t$ evaluated `range(0, k + 1)`, which inadvertently included the current frame $t$ ($F_t = 1$). 

This v2 protocol enforces a strict forward-looking failure prediction label:
$$y_t = \max(F_{t+1}, F_{t+2}, \dots, F_{t+5})$$
and purges all frames where failure has already occurred ($F_t = 1$). All experiments, models, and statistical tests in v2 operate alongside the existing archival v1 results without modifying previously reported numbers.

---

## 2. Mathematical Specification of the Strict Label

For a sequence of video frames and visual odometry telemetry $\{I_t, \mathbf{x}_t\}_{t=0}^{N-1}$ within an active flight window:
1. **Raw Frame Failure Indicator ($F_t$):**
   $$F_t = \mathbb{I}(\text{num\_inliers\_pose}_t < 8)$$
2. **Strict Forward Prediction Horizon ($K=5$):**
   $$y_t = \max_{s \in \{1, 2, 3, 4, 5\}} F_{t+s}$$
   The current frame $F_t$ is **strictly excluded** from the prediction window.
3. **Current-Frame Failure Purge:**
   All frames where $F_t = 1$ are dropped from training, validation, and testing:
   $$\mathcal{D}_{\text{valid}} = \{ t \mid F_t = 0 \}$$
   The model is strictly evaluated on its capacity to predict impending failure given that the platform is currently operating normally.
4. **Temporal Boundary Drops:**
   - The first 5 frames ($t < 5$) of each flight/trial are dropped due to the maximum backward lag ($L = 5$).
   - The last 5 frames ($t \ge N - 5$) of each flight/trial are dropped due to the forward prediction horizon ($K = 5$).

---

## 3. Data Partitions & Split Isolation Guarantees

All flight partitions match `r3_refinement/manifests/refinement_split_manifest.json` exactly:

| Partition | R2 Motion Flights | R3 Visual Trials | Total Sequences |
|---|---|---|---|
| **Training Pool** | 15 flights | 15 trials (3 base flights × 5 regimes) | 30 sequences |
| **Validation Pool** | 13 flights | 10 trials (2 base flights × 5 regimes) | 23 sequences |
| **Protected Test Set** | 14 flights | 15 trials (3 base flights × 5 regimes) | 29 sequences |
| **Total** | 42 flights | 40 trials | 82 sequences |

### Isolation Rules:
- **Zero Flight Overlap:** Trajectory families and base flights are strictly partitioned at the flight level.
- **Single-Use Protected Test Set:** The protected test partition is touched **strictly once** for final reporting.
- **No Test Leakage in Model Decisions:** All feature scaling, neural network training, early stopping, hyperparameter choices, and decision thresholds ($\theta^*$) are determined exclusively using the training and validation partitions.

---

## 4. Multi-Seed Training & Model Selection

1. **Architecture:** Multi-Layer Perceptron (`SmallMLP`)
   $$\mathbf{x} \in \mathbb{R}^D \xrightarrow{\text{Linear}(D, 32)} \text{ReLU} \xrightarrow{\text{Linear}(32, 16)} \text{ReLU} \xrightarrow{\text{Linear}(16, 1)} \hat{z} \in \mathbb{R}$$
   where $D=15$ for Model A and $D=35$ for Model B.
2. **Optimizer & Regularization:**
   - Optimizer: Adam ($\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)
   - Weight Decay: $10^{-4}$
   - Loss Function: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
   - Batch Size: 256
   - Maximum Epochs: 50
   - Early Stopping: Patience of 10 epochs monitored on validation loss.
3. **Random Seeds:**
   Models are retrained across 5 distinct random seeds: $\{42, 43, 44, 45, 46\}$.
   All reported metrics include the sample mean and sample standard deviation across seeds.
4. **Validation Threshold Selection:**
   Decision threshold $\theta^*$ is selected per seed on the joint validation split across grid $[0.05, 0.95]$ with step $0.01$ by maximizing the F1 score:
   $$\theta^* = \arg\max_{\theta \in [0.05, 0.95]} F_1(\mathbf{y}_{\text{val}}, \mathbb{I}(\sigma(\hat{\mathbf{z}}_{\text{val}}) \ge \theta))$$

---

## 5. Feature Sets & Causality Verification

### Feature Set A (Baseline 15 Features):
Derived strictly from platform motion and visual odometry solver state across backward lags $\{0, 1, 2, 3, 5\}$:
- `eis_yaw_rate_deg_lag{0,1,2,3,5}`
- `feature_vel_mean_lag{0,1,2,3,5}`
- `is_r_frame_lag{0,1,2,3,5}`

### Feature Set B (A + 20 Image-Quality Features):
Set A (15 features) supplemented by 4 image-quality features computed on the degraded image stream at lags $\{0, 1, 2, 3, 5\}$:
1. `mean_grad_mag_lag{0,1,2,3,5}`: Mean Sobel gradient magnitude: $\frac{1}{HW}\sum \sqrt{G_x^2 + G_y^2}$
2. `laplacian_var_lag{0,1,2,3,5}`: Variance of Laplacian: $\text{Var}(\nabla^2 I)$
3. `pixel_std_lag{0,1,2,3,5}`: Standard deviation of pixel intensities: $\text{Std}(I)$
4. `shi_tomasi_corners_lag{0,1,2,3,5}`: Count of Shi-Tomasi corners detected (`cv2.goodFeaturesToTrack(I, maxCorners=1000, qualityLevel=0.01, minDistance=10)`)

### Optical Degradation Formula:
For trial frames, degraded images are deterministically reconstructed via:
$$I_{\text{degraded}}(x, y) = \text{clip}\left((1 - \alpha) \cdot I_{\text{clean}}(x, y) + \alpha \cdot 128.0, \, 0, \, 255\right)$$
where $\alpha \in [0.0, 1.0]$ is the documented degradation parameter. For R2 motion flights, $\alpha = 0.0$ ($I_{\text{degraded}} = I_{\text{clean}}$).

### Strict Causality Invariance Criterion:
Every feature at time $t$ must depend strictly on information available at or before time $t$:
$$\mathbf{x}_t = f(\{I_s\}_{s \le t}, \{\mathbf{m}_s\}_{s \le t})$$
A dedicated unit test asserts that modifying frame $I_{t+1}$ or any subsequent frame produces zero change in $\mathbf{x}_t$.

---

## 6. Statistical Inference & Bootstrap Protocol

1. **Hierarchical / Clustered Bootstrap:**
   Because intra-flight frames exhibit strong temporal correlation, bootstrap resampling is conducted at the **trial/flight level**:
   - For R3 visual evaluation: 15 protected visual trials are resampled with replacement 1,000 times (random seed 42).
   - For R2 motion evaluation: 14 protected motion flights are resampled with replacement 1,000 times (random seed 42).
2. **Confidence Intervals:**
   Report 95% non-parametric percentile bootstrap confidence intervals $[q_{0.025}, q_{0.975}]$.
3. **Paired Differences:**
   For any comparative metric $\Delta = M_B - M_A$ or $M_B - M_{\text{Frozen}}$, calculate $\Delta^{(b)}$ on each bootstrap resample $b \in \{1, \dots, 1000\}$ to establish the paired 95% bootstrap confidence interval.

---

## 7. Reporting & Artifact Outputs

All outputs are saved to `r3_v2/results/` alongside existing records:
- `test_metrics_summary.csv`: Aggregated test performance (mean, std, 95% CIs).
- `per_seed_metrics.csv`: Detailed evaluation for all 5 seeds.
- `bootstrap_confidence_intervals.csv`: Percentile bootstrap distributions.
- `paired_bootstrap_differences.csv`: Paired bootstrap deltas ($B - A$, $A - \text{Frozen}$, $B - \text{Frozen}$).
- `per_regime_auroc.csv`: Regime-stratified AUROC (Nominal, Low, Transition, Medium).
- `per_flight_auroc.csv`: Flight-level AUROC distribution.
- `single_feature_directions.csv`: AUROC and directionality of individual features on R2 vs R3.
- `v2_summary.json`: Comprehensive structured summary of all findings.
