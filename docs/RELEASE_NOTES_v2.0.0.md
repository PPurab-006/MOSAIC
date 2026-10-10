# Release v2.0.0 — Strict Prospective Re-Evaluation of VO Failure Prediction

> [!WARNING]
> **SUPERSEDED**: The original v1 experimental results in this repository used a label that included the current frame (`range(0, k+1)`). Forensic audit revealed that 83% of frozen-model true positives under visual degradation were frames where VO tracking had already failed at time $t$ ($F_t = 1$).
> The v1 headline numbers (AUROC 0.9371, recall 87.20%, precision 0.94, and the claim of unconditional "mechanism-general" transfer) are formally retracted.
> See `docs/V2_PROTOCOL.md`, `docs/V2_DEVIATIONS.md`, and `r3_v2/results/` for the strict re-evaluation.

---

### Key Findings of the V2 Strict Re-Evaluation

1. **Frozen Model Collapse Under Strict Labels**:
   Under strict prospective labels ($y_t = \max(F_{t+1}..F_{t+5})$, $F_t=1$ purged), the frozen R2 model drops to **0.6545** pooled visual AUROC, falling below chance on all 3 base flights in Transition (0.4561–0.4787) and Medium (0.1025–0.4022). The motion-only model fails to anticipate visual tracking collapse.

2. **Retraining & Regime Recognition**:
   Retraining on strict labels with 15 motion features achieves **0.8798 ± 0.0057** pooled visual AUROC (Model A). However, a model with access only to regime identity (Baseline D) reaches **0.9029 ± 0.0014**. Most of the pooled visual AUROC represents coarse regime recognition rather than fine-grained frame-level dynamics.

3. **Value of Image-Quality Features**:
   Adding 20 visual quality features (Model B) yields an honest **+0.0526** pooled gain over motion-only (Cluster 95% CI: [+0.0433, +0.0592], reaching **0.9345 ± 0.0014**). It adds a consistent **+0.0407** in Transition ([+0.0267, +0.0470], beating Model A on all 3 flights). In Nominal (+0.0106) and Low (+0.0087), image features add negligible discriminatory benefit.

4. **Medium Regime Assessment**:
   $B - A$ in Medium is **+0.0974** ([-0.0076, +0.1818]). Model B beats Model A on 2 of 3 flights (0.7452 vs 0.5634; 0.7016 vs 0.5932), and is essentially tied on the third (0.6996 vs 0.7072). This is suggestive. What is robust is Model B over the frozen model: Frozen is < 0.50 on all 3 Medium flights (0.1025, 0.4022, 0.3146) while Model B is $\ge 0.70$ on all 3 flights.

5. **Preserved Motion Generalization**:
   Motion domain performance is fully preserved without suffering: Model B achieves **0.7749 ± 0.0052** (0.7808 ensemble) vs **0.7665** for the frozen baseline.

6. **Cluster Realism & Limits**:
   Visual within-regime metrics are reported as **Per-Flight Ranges (n=3)** across the 3 physical base flights (`sweep_G_C_R1`, `sweep_G_B_R2`, `sweep_G_S_R2`). The evaluation is bounded by 3 physical base flights, one synthetic degradation type (uniform veil contrast attenuation), and an undefined High regime (single-class continuous failure).

---

### Master Comparison Table

| Model | Pooled Visual AUROC (Seeds 42–46) | Nominal ($n=3$) | Low ($n=3$) | Transition ($n=3$) | Medium ($n=3$) | High | Motion AUROC (Seeds 42–46) | Motion Cluster 95% CI ($n=14$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen R2 Baseline** | $0.6545 \pm 0.0000$ | $0.7661$ [$0.7420, 0.7886$] | $0.6935$ [$0.5814, 0.7676$] | $0.4711$ [$0.4561, 0.4787$] | $0.2724$ [$0.1025, 0.4022$] | Undefined | $0.7665 \pm 0.0000$ | $0.7665$ [$0.7169, 0.8054$] |
| **Model A (15 Motion)** | $0.8798 \pm 0.0057$ | $0.7356$ [$0.6919, 0.7917$] | $0.6711$ [$0.5756, 0.7332$] | $0.9279$ [$0.9217, 0.9389$] | $0.6169$ [$0.5634, 0.7072$] | Undefined | $0.7556 \pm 0.0053$ | $0.7605$ [$0.7151, 0.7964$] |
| **Model B (35 Features)**| **$0.9345 \pm 0.0014$** | $0.7473$ [$0.7205, 0.7786$] | $0.6816$ [$0.5956, 0.7270$] | **$0.9690$** [$0.9656, 0.9729$] | **$0.7159$** [$0.6996, 0.7452$] | Undefined | **$0.7749 \pm 0.0052$** | $0.7808$ [$0.7366, 0.8173$] |
| **Baseline C (MLP: 20 Image)** | $0.9085 \pm 0.0010$ | $0.6144$ [$0.5472, 0.6745$] | $0.5187$ [$0.4403, 0.5648$] | $0.9655$ [$0.9629, 0.9675$] | $0.4835$ [$0.3953, 0.5883$] | Undefined | $0.6513 \pm 0.0053$ | $0.6571$ [$0.5872, 0.7077$] |
| **Baseline C (LogReg: 20 Image)**| $0.8905 \pm 0.0000$ | $0.4856$ [$0.4423, 0.5155$] | $0.4766$ [$0.4546, 0.4976$] | $0.8907$ [$0.8651, 0.9217$] | $0.4109$ [$0.3655, 0.5012$] | Undefined | $0.5469 \pm 0.0000$ | $0.5469$ [$0.4940, 0.5895$] |
| **Baseline D (MLP: Regime+$\alpha$)** | $0.9029 \pm 0.0014$ | $0.5000$ [$0.5000, 0.5000$] | $0.5000$ [$0.5000, 0.5000$] | $0.9414$ [$0.9331, 0.9556$] | $0.5000$ [$0.5000, 0.5000$] | Undefined | $0.5000 \pm 0.0001$ | $0.5000$ [$0.5000, 0.5000$] |
| **Baseline D (LogReg: Regime+$\alpha$)**| $0.8826 \pm 0.0000$ | $0.5000$ [$0.5000, 0.5000$] | $0.5000$ [$0.5000, 0.5000$] | $0.9000$ [$0.8692, 0.9164$] | $0.5000$ [$0.5000, 0.5000$] | Undefined | $0.5000 \pm 0.0000$ | $0.5000$ [$0.5000, 0.5000$] |

---

### Reproducibility & Bundled Artifacts
- **Attached Archive**: `r3_v2_reevaluation_bundle.zip` contains all 14 re-evaluation assets (tables, JSON summary, dataset, feature extraction code, and protocol docs).
- **Reproduction**: Run `python scripts/reproduce_v2_tables.py` to verify all numbers from `r3_v2_strict_dataset.csv`.
