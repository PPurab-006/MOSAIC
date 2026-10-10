# MOSAIC: Visual Odometry Failure Prediction Under Non-Motion Visual Degradation

> [!WARNING]
> ## Methodological Retraction & V2 Supersession Notice
> **The v1 experimental results in this repository are SUPERSEDED.**
> In the original v1 pipeline, the forward-looking label window was defined as $y_t = \max(F_t, F_{t+1}, \dots, F_{t+5})$ (`range(0, k+1)`), inadvertently including the current frame's tracking state ($F_t$). 
> Forensic analysis revealed that **83% of the frozen-model true positives under visual degradation were frames where VO tracking had already failed at time $t$ ($F_t = 1$)**. Consequently, the system was performing contemporaneous failure detection rather than prospective early warning.
> 
> **Retracted V1 Headline Claims:**
> - The reported visual test AUROC of **0.9371**, recall of **87.20%**, precision of **0.94**, and the claim of unconditional **"mechanism-general"** failure prediction are formally retracted.
> 
> **V2 Strict Re-Evaluation Protocol:**
> A leak-free v2 re-evaluation was locked before execution ([docs/V2_PROTOCOL.md](docs/V2_PROTOCOL.md), [docs/V2_DEVIATIONS.md](docs/V2_DEVIATIONS.md)):
> 1. **Strict Prospective Horizon**: $y_t = \max(F_{t+1}, \dots, F_{t+5})$ (`range(1, k+1)`), strictly prospective over the next 165 ms ($\tau = 0.165\text{ s}$).
> 2. **Purging Active Failures**: All frames with active VO collapse ($F_t = 1$) are purged, eliminating contemporaneous contamination.
> 3. **Boundary Drops**: The first 5 and last 5 frames of each flight/trial are dropped.
> 4. **Protected Split Evaluation**: Test set evaluated strictly once across 5 deterministic seeds (42..46).
> 5. **Cluster Realism**: Visual within-regime metrics are reported as **Per-Flight Ranges ($n=3$)**, reflecting the 3 physical base flights rather than pseudocontinuous confidence intervals.

---

| Metadata | Details |
|---|---|
| **Release** | **v2.0.0** (Strict Prospective Re-Evaluation) |
| **Research Area** | Autonomous UAV Navigation · Visual Odometry · Domain Shift & Failure Prediction |
| **Core Stack** | Python 3.11 · PyTorch · Scikit-Learn · SciPy · PX4 SITL · Gazebo Sim |
| **Strict Dataset** | 51,325 frames across 82 sequences (`r3_v2/data/r3_v2_strict_dataset.csv`) |
| **Core Architecture**| `SmallMLP` ($15 \to 32 \to 16 \to 1$, **1,057 parameters**) |
| **Primary Takeaway** | Frozen model fails under strict visual shift ($0.6545$ pooled; $<0.50$ in Transition/Medium). Adding 20 image features (Model B) recovers pooled AUROC to $0.9345$ ($+0.0547$ over motion-only Model A) and Transition to $0.9690$ without degrading motion performance ($0.7749$ vs $0.7665$). |

---

## Reconciled V2 Model Comparison Table

Master comparison across 5 seeds (42–46) on the protected test sets (14 R2 motion flights + 15 R3 visual trials).
- **Pooled Visual AUROC & Motion AUROC**: Reported as **Mean ± Sample Std across seeds 42–46 ($ddof=1$)**.
- **Within-Regime Visual Metrics**: Reported as **Overall Ensemble [Per-Flight Range, $n=3$]** across base flights `sweep_G_C_R1`, `sweep_G_B_R2`, and `sweep_G_S_R2`. Because $n=3$, percentile bootstrap CIs collapse to the per-flight minimum and maximum; they are reported as empirical ranges.
- **High Regime**: 100% positive rate across all trials; AUROC is mathematically undefined (single-class).

| Model | Architecture & Features | Pooled Visual AUROC (Seeds 42–46) | Nominal ($n=3$) | Low ($n=3$) | Transition ($n=3$) | Medium ($n=3$) | High | Motion AUROC (Seeds 42–46) | Motion Cluster 95% CI ($n=14$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen R2 Baseline** | SmallMLP (15 Motion) | $0.6545 \pm 0.0000$ | $0.7661$ [$0.7420, 0.7886$] | $0.6935$ [$0.5814, 0.7676$] | $0.4711$ [$0.4561, 0.4787$] | $0.2724$ [$0.1025, 0.4022$] | Undefined | $0.7665 \pm 0.0000$ | $0.7665$ [$0.7169, 0.8054$] |
| **Model A** | Retrained MLP (15 Motion) | $0.8798 \pm 0.0057$ | $0.7356$ [$0.6919, 0.7917$] | $0.6711$ [$0.5756, 0.7332$] | $0.9279$ [$0.9217, 0.9389$] | $0.6169$ [$0.5634, 0.7072$] | Undefined | $0.7556 \pm 0.0053$ | $0.7605$ [$0.7151, 0.7964$] |
| **Model B** | Refined MLP (15 Motion + 20 Image) | **$0.9345 \pm 0.0014$** | $0.7473$ [$0.7205, 0.7786$] | $0.6816$ [$0.5956, 0.7270$] | **$0.9690$** [$0.9656, 0.9729$] | **$0.7159$** [$0.6996, 0.7452$] | Undefined | **$0.7749 \pm 0.0052$** | $0.7808$ [$0.7366, 0.8173$] |
| **Baseline C (MLP)** | MLP (20 Image Features Only) | $0.9085 \pm 0.0010$ | $0.6144$ [$0.5472, 0.6745$] | $0.5187$ [$0.4403, 0.5648$] | $0.9655$ [$0.9629, 0.9675$] | $0.4835$ [$0.3953, 0.5883$] | Undefined | $0.6513 \pm 0.0053$ | $0.6571$ [$0.5872, 0.7077$] |
| **Baseline C (LogReg)** | LogReg (20 Image Features Only) | $0.8905 \pm 0.0000$ | $0.4856$ [$0.4423, 0.5155$] | $0.4766$ [$0.4546, 0.4976$] | $0.8907$ [$0.8651, 0.9217$] | $0.4109$ [$0.3655, 0.5012$] | Undefined | $0.5469 \pm 0.0000$ | $0.5469$ [$0.4940, 0.5895$] |
| **Baseline D (MLP)** | MLP (Regime ID + $\alpha$) | $0.9029 \pm 0.0014$ | $0.5000$ [$0.5000, 0.5000$] | $0.5000$ [$0.5000, 0.5000$] | $0.9414$ [$0.9331, 0.9556$] | $0.5000$ [$0.5000, 0.5000$] | Undefined | $0.5000 \pm 0.0001$ | $0.5000$ [$0.5000, 0.5000$] |
| **Baseline D (LogReg)** | LogReg (Regime ID + $\alpha$) | $0.8826 \pm 0.0000$ | $0.5000$ [$0.5000, 0.5000$] | $0.5000$ [$0.5000, 0.5000$] | $0.9000$ [$0.8692, 0.9164$] | $0.5000$ [$0.5000, 0.5000$] | Undefined | $0.5000 \pm 0.0000$ | $0.5000$ [$0.5000, 0.5000$] |

---

## What the Evidence Actually Claims

1. **Frozen Model Collapse Under Strict Labels**:
   Under strict prospective labels ($F_t = 1$ purged), the frozen R2 model drops to $0.6545$ pooled visual AUROC, falling below chance on all 3 base flights in Transition ($0.4561$–$0.4787$) and Medium ($0.1025$–$0.4022$). The motion-only model completely fails to anticipate visual tracking collapse.
2. **Retraining & Regime Recognition**:
   Retraining on strict labels with motion features alone achieves $0.8798$ pooled AUROC (Model A). However, a model with access only to regime identity (Baseline D) reaches $0.9029$. Most of the pooled visual AUROC represents coarse regime recognition rather than fine-grained frame-level dynamics.
3. **Value of Image-Quality Features**:
   Adding 20 visual quality features (Model B) yields an honest $+0.0526$ pooled gain over motion-only (Cluster 95% CI: $[+0.0433, +0.0592]$). It adds a consistent $+0.0407$ in Transition ($[+0.0267, +0.0470]$, beating Model A on all 3 flights). In Nominal ($+0.0106$) and Low ($+0.0087$), image features add negligible discriminatory benefit.
4. **Medium Regime Assessment**:
   $B - A$ in Medium is $+0.0974$ ($[-0.0076, +0.1818]$). Model B beats Model A on 2 of 3 flights ($0.7452$ vs $0.5634$; $0.7016$ vs $0.5932$), and is essentially tied on the third ($0.6996$ vs $0.7072$). This is **suggestive**. What is completely robust is Model B over the frozen model: Frozen is $< 0.50$ on all 3 Medium flights ($0.1025$, $0.4022$, $0.3146$) while Model B is $\ge 0.70$ on all 3 flights.
5. **Preserved Motion Generalization**:
   Motion domain performance is fully preserved without suffering: Model B achieves $0.7749 \pm 0.0052$ ($0.7808$ ensemble) vs $0.7665$ for the frozen baseline.
6. **Honest Limitations**:
   The evaluation is bounded by 3 physical visual test flights (`sweep_G_C_R1`, `sweep_G_B_R2`, `sweep_G_S_R2`), one synthetic degradation type (uniform veil contrast attenuation), and a single-class High regime where continuous 100% failure prevents AUROC computation.

---

## Corrections to Prior Reporting Errata

In the process of conducting this release audit, the following historical documentation errors were identified and corrected across the entire repository:
- **Transition Failure Rate**: Corrected from erroneous $66.8\%$ to actual legacy rate of **$63.0\%$** ($3,849 / 6,109$ active frames).
- **Angular Velocity Distribution**: Corrected from "Peak $< 35^\circ/\text{s}$" to **95th percentile $= 35.89^\circ/\text{s}$** (maximum observed yaw rate is $167.4^\circ/\text{s}$, with mean $10.92^\circ/\text{s}$).
- **Legacy Base Rate**: Corrected from $0.647$ to actual legacy base rate of **$0.6142$** ($18,761 / 30,545$ active frames).
- **Model Parameter Count**: Corrected from $1,041$ to **1,057 parameters** (verified by explicit count: Layer 1 $15 \times 32 + 32 = 512$; Layer 2 $32 \times 16 + 16 = 528$; Layer 3 $16 \times 1 + 1 = 17$; Total = $1,057$).

---

## Experimental Setup & Optical Degradation Model

### Degradation Regimes
To decouple VO tracking failure from vehicle dynamics, failures were induced through deterministic optical contrast attenuation towards an ambient atmospheric veil luminance ($A_\infty = 128.0$):
$$I_{\text{degraded}}(x, y) = (1 - \alpha) \cdot I_{\text{clean}}(x, y) + \alpha \cdot A_\infty$$

Five degradation severities were evaluated:
1. **Nominal ($\alpha = 0.000$)**: Unattenuated baseline camera stream (VO failure rate: 13.1%).
2. **Low ($\alpha = 0.920$)**: High optical attenuation yet remaining below VO tracking collapse (VO failure rate: 34.5%).
3. **Transition ($\alpha: 0.000 \to 0.975$)**: Dynamic linear ramp crossing the tracking collapse boundary (VO failure rate: 63.0%).
4. **Medium ($\alpha = 0.960$)**: Severe contrast attenuation with intermittent solver failure (VO failure rate: 96.5%).
5. **High ($\alpha = 0.985$)**: Extreme contrast attenuation inducing continuous Essential matrix tracking collapse (VO failure rate: 100.0%).

### Kinematic Decoupling Envelope
- **Mean Yaw Rate**: $10.92^\circ/\text{s}$ (vs $41.55^\circ/\text{s}$ in R2 test, Cohen's $d = 1.30$, Kolmogorov-Smirnov $p < 10^{-10}$).
- **95th Percentile Yaw Rate**: $35.89^\circ/\text{s}$ (Max: $167.4^\circ/\text{s}$).
- **Translational Speed**: Cruise speed $1.81\text{ m/s}$ (median $0.98\text{ m/s}$).
- **Body Attitude Tilt**: Mean $4.29^\circ$ (strictly benign level flight).

---

## Data Dependency & Reproduction Notice

> [!NOTE]
> **Data Scope**:
> - **Reproducing Tables & Figures**: Needs **ONLY** the strict dataset (`r3_v2/data/r3_v2_strict_dataset.csv`) and the saved model weights (`r3_v2/models/`). Everything runs self-contained from this repository.
> - **Regenerating Raw Data**: Regenerating raw visual degradation images or rerunning offline VO from scratch requires the sibling directories `../Research2` (for base flight bags) and `../ROS` (for camera calibration and VO node pipelines).

---

## Reproduction Commands

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full verification instructions and cryptographic checksums.

```bash
# 1. Activate Python virtual environment
source .venv/bin/activate

# 2. Recompute all v2 tables and verify zero numerical divergence
python scripts/reproduce_v2_tables.py

# 3. Run test suites
pytest r3_v2/tests

# 4. Run full analysis pipeline
bash scripts/run_r3_analysis.sh
```

---

## Repository Structure

```
MOSAIC/
├── CITATION.cff                     # Citation metadata
├── LICENSE                          # Apache-2.0 License
├── README.md                        # Primary documentation and v2 report
├── REPRODUCIBILITY.md               # Exact commands and checksums
├── docs/
│   ├── V2_PROTOCOL.md               # Pre-registered strict evaluation protocol
│   ├── V2_DEVIATIONS.md             # Reconciled deviations and audit log
│   ├── V2_CHANGELOG.txt             # Git changelog of train_and_evaluate_v2.py
│   └── r3_final_research_report.md  # Archival Phase 1 report (superseded)
├── models/
│   ├── expanded_mlp.pt              # Authoritative frozen R2 model (1,057 params)
│   ├── expanded_scaler.joblib       # Authoritative frozen R2 scaler (15 feats)
│   └── frozen_r2_manifest.json      # Checksums and split specifications
├── r3_v2/
│   ├── data/
│   │   └── r3_v2_strict_dataset.csv # Full strict dataset (51,325 frames)
│   ├── models/                      # Checkpoints for seeds 42..46 (Models A & B)
│   ├── results/                     # All CSV tables, summary JSON, and release bundle
│   ├── src/                         # Feature extraction, models, and evaluators
│   └── tests/                       # Unit tests for strict labels and leak-free splits
└── scripts/
    ├── reproduce_v2_tables.py       # Standalone table reproduction script
    └── run_r3_analysis.sh           # Main verification runner
```
