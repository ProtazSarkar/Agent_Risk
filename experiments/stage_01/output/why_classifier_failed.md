# Failure Analysis Report: Why the XGBoost Multi-Class Classifier Failed

**Stage**: `stage_01` Baseline Evaluation  
**Task**: SemEval 2027 AgentRisk - Agent Risk Severity Prediction  
**Dataset**: 568 Train Samples, 143 Dev Samples (Dense MiniLM Embeddings: 384 dimensions)  
**Date**: November 2026  

---

## 1. Executive Summary

In Stage 01, we evaluated a unified **XGBoost multi-class classifier** (`objective='multi:softprob'`) trained on 384-dimensional dense semantic embeddings (`all-MiniLM-L6-v2`) to predict agent risk severity levels (1 to 5).

The classifier achieved an overall accuracy of **41.26%** and a Quadratic Weighted Kappa (QWK) of only **0.1261**, completely failing to detect rare severity levels (0.0000 recall on Severity Level 2 and Level 5).

This failure was not caused by model implementation or software bugs; it stems from **three fundamental mathematical and structural mismatches** between the problem formulation and the algorithm.

---

## 2. Compounding Bottlenecks

### Bottleneck 1: High Dimensionality vs. Low Sample Size ($p \approx n$)
* **Data Dimensions**: $p = 384$ dense embedding dimensions (`emb_0` through `emb_383`).
* **Sample Size**: $n = 568$ training records.
* **The Problem**: In tree-based gradient boosting (XGBoost / LightGBM), decision trees split orthogonally along single coordinate axes. When $p$ is nearly as large as $n$ ($384 \approx 568$), each split partitions the remaining small subset of data into increasingly sparse subspaces. The trees easily overfit to noisy embedding dimensions or split on spurious correlations that do not generalize to the held-out development split.

### Bottleneck 2: Extreme Class Imbalance
* **Training Class Distribution**:
  * Level 1 (Benign): **265 samples (46.7%)** — Majority
  * Level 2 (Low Risk): **33 samples (5.8%)** — Minority
  * Level 3 (Moderate Risk): **127 samples (22.4%)**
  * Level 4 (High Risk): **136 samples (23.9%)**
  * Level 5 (Catastrophic Risk): **7 samples (1.2%)** — Extreme Minority
* **The Problem**: Standard multi-class classification minimizes cross-entropy loss ($\mathcal{L}_{\text{CE}}$). In a multi-class setting with 5 nominal targets, the gradient contribution from 7 samples in Class 5 is overwhelmingly drowned out by the 265 samples in Class 1 and 136 in Class 4. Even with sample weighting, the trees determine that the safest risk-minimizing policy is to ignore Level 2 and Level 5 entirely and dump predictions into majority classes (Levels 1, 3, and 4).

### Bottleneck 3: Ignoring Ordinality (Treating Ordinal Risk as Nominal Classes)
* **The Problem**: Risk severity is strictly **ordinal**:
  $$\text{Level 1} < \text{Level 2} < \text{Level 3} < \text{Level 4} < \text{Level 5}$$
* **Symmetric Penalty in Cross-Entropy**: Multi-class softmax treats all classes as mutually exclusive, unordered categorical bins.
  * If the true label is **Level 5**, predicting **Level 1** (a catastrophic false negative) receives the exact same penalty as predicting **Level 4** (a minor off-by-one boundary discrepancy).
* **Destruction of Quadratic Weighted Kappa (QWK)**: The official competition evaluation metric is QWK:
  $$\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(N - 1)^2}$$
  Because multi-class XGBoost is unaware of the quadratic distance matrix $w_{i,j}$, its off-class predictions randomly jump across the spectrum (e.g., predicting Level 4 for true Level 1, or Level 1 for true Level 4), inflicting maximum quadratic penalties on the QWK score.

---

## 3. Empirical Evidence (Stage 01 Confusion Matrix)

The stage 01 confusion matrix on the 143 dev samples clearly demonstrates this failure mode:

```
True \ Pred   Sev 1     Sev 2     Sev 3     Sev 4     Sev 5     Total True
--------------------------------------------------------------------------
Sev 1 (Safe)   40         0        10        16         0           66
Sev 2 (Low)     4         0         0         4         0            8  <-- 0% Recall
Sev 3 (Mod)    16         0         7         9         0           32
Sev 4 (High)   18         0         5        12         0           35
Sev 5 (Crit)    0         0         0         2         0            2  <-- 0% Recall
--------------------------------------------------------------------------
Total Pred     78         0        22        41         0          143
```

* **Zero Predictions for Rare Classes**: Severity 2 and Severity 5 were predicted **0 times**.
* **Wild Diagonal Offsets**: 16 Level 1 instances were predicted as Level 4, and 18 Level 4 instances were predicted as Level 1.
* **Resulting QWK**: **0.1261** (barely above chance).

---

## 4. The Solution: Shifting to a Regression Approach

Instead of forcing nominal 5-class classification, we formulate agent risk estimation as **continuous regression with threshold optimization**:

```
[Dense MiniLM Embeddings (384-d)]
               │
               ▼
   [Regularized Regressor]  (Ridge Regression / SVR with L2 norm penalty)
               │
               ▼
   Continuous Risk Score  $\hat{y} \in [1.0, 5.0]$
               │
               ▼
   [Nelder-Mead Threshold Optimizer]  (Custom decision cutoffs: $t_1, t_2, t_3, t_4$)
               │
               ▼
   Discrete Ordinal Verdict  $\hat{k} \in \{1, 2, 3, 4, 5\}$
```

### Why This Fixes All Three Bottlenecks:
1. **L2 Regularization vs. High Dimensionality**: Ridge regression ($\ell_2$-regularized linear model) and SVR handle $p \approx n$ smoothly by shrinking weights radially rather than greedy axis-aligned partitioning, preventing catastrophic tree overfitting.
2. **Distance-Aware Optimization**: Mean Squared Error (MSE) directly minimizes $(y - \hat{y})^2$, mathematically aligning the training objective with the Quadratic Weighted Kappa penalty. An off-by-one error (e.g., predicted 3.8 vs. true 4.0) incurs negligible loss, while large errors (predicted 1.0 vs. true 5.0) incur heavy penalties.
3. **Nelder-Mead Threshold Calibration**: Rather than arbitrary rounding (e.g., $1.5, 2.5, 3.5, 4.5$), Nelder-Mead optimization tunes the 4 decision boundaries on validation data specifically to maximize the non-differentiable QWK metric, restoring balanced sensitivity for minority risk levels.

---

## 5. Next Steps for Stage 02

* **Implementation File**: `experiments/stage_02/regid_regression_train.py`
* **Algorithm**: `RidgeCV` / `Ridge` + `scipy.optimize.minimize(..., method='Nelder-Mead')`
* **Target Metric**: Maximize Quadratic Weighted Kappa ($\text{QWK} > 0.20$) while maintaining calibrated ordinal boundaries.

