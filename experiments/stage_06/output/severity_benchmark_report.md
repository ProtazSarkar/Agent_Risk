# Stage 06 Benchmark Report: Risk Severity Prediction (Levels 1 to 5)

## 1. Executive Summary

This report evaluates the **Stage 06 Risk Level (1–5) Prediction Suite**.
It combines the **Multi-Domain Probability Ensemble** (extracted via the 11 Stage 05 Semantic Prototype Classifiers) with gradient-boosted trees, random forests, and ordinal regressors.

- **Champion Architecture**: `HistGradientBoosting_Regressor`
- **Held-Out Dev Evaluation (176 instances)**:
  - **Exact Accuracy (1–5)**: **59.66%**
  - **Within-1 (Adjacent) Accuracy**: **85.80%**
  - **Quadratic Weighted Kappa (QWK)**: **0.6895**
  - **Mean Absolute Error (MAE)**: **0.5966** levels
  - **Root Mean Squared Error (RMSE)**: **1.0417**
  - **Macro F1-Score**: **0.5134**
  - **Mean Prediction Confidence**: **62.08%**
  - **High-Confidence (>= 70%) Accuracy**: **89.80%** (49 samples)

---

## 2. Model Comparison Matrix (Held-Out Dev Set)

| Candidate Model | QWK | Exact Acc | Within-1 Acc | MAE | RMSE | Macro F1 | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HistGradientBoosting_Classifier`  | **0.6420** | **60.80%** | 85.23% | 0.6420 | 1.1750 | 0.5308 | 85.4% |
| `HistGradientBoosting_Regressor` **(Champion)** | **0.6895** | **59.66%** | 85.80% | 0.5966 | 1.0417 | 0.5134 | 62.1% |
| `Hybrid_XGB_Ensemble`  | **0.6700** | **64.20%** | 86.36% | 0.5795 | 1.1078 | 0.5672 | 67.4% |
| `RandomForest_Classifier`  | **0.5981** | **60.23%** | 82.39% | 0.6932 | 1.2568 | 0.5094 | 64.0% |
| `RandomForest_Regressor`  | **0.6990** | **59.66%** | 87.50% | 0.5852 | 1.0308 | 0.5252 | 60.9% |
| `Ridge_Ordinal_Regressor`  | **0.5895** | **50.00%** | 82.39% | 0.7557 | 1.1990 | 0.3335 | 59.7% |
| `XGBoost_Classifier`  | **0.6805** | **63.64%** | 86.36% | 0.5909 | 1.1231 | 0.5716 | 78.9% |
| `XGBoost_Ordinal_Regressor`  | **0.6855** | **60.80%** | 86.93% | 0.5795 | 1.0335 | 0.5192 | 61.7% |

---

## 3. 5x5 Confusion Matrix (Levels 1 to 5)

| True \ Predicted | Pred 1 | Pred 2 | Pred 3 | Pred 4 | Pred 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Severity 1** |  58 |  11 |   4 |   3 |   0 |
| **Severity 2** |   5 |   8 |   0 |   2 |   0 |
| **Severity 3** |   2 |   5 |  19 |   9 |   0 |
| **Severity 4** |   2 |   6 |  12 |  16 |   2 |
| **Severity 5** |   0 |   4 |   2 |   2 |   4 |


---

## 4. Per-Severity Performance Breakdown

| Severity Level | True Count | Pred Count | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Severity 1** | 76 | 67 | 0.8657 | 0.7632 | 0.8112 |
| **Severity 2** | 15 | 34 | 0.2353 | 0.5333 | 0.3265 |
| **Severity 3** | 35 | 37 | 0.5135 | 0.5429 | 0.5278 |
| **Severity 4** | 38 | 32 | 0.5000 | 0.4211 | 0.4571 |
| **Severity 5** | 12 | 6 | 0.6667 | 0.3333 | 0.4444 |

---

## 5. Confidence Analysis & Reliability
- **Average Confidence**: `62.08%` (standard deviation: `10.95%`)
- **High-Confidence Accuracy (>= 70%)**: `89.80%` across `49` samples.
- **Low-Confidence Accuracy (< 50%)**: `41.67%` across `24` samples.
- The model exhibits strong calibration: higher model confidence directly corresponds to higher empirical accuracy.

---

## 6. Sample Prediction Trace (20 Held-Out Instances)

| Example ID | True | Pred | Conf | Top Contributing Domain | Match Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `AGENTRISK-004` | **4** | **4** | 57.6% | Privacy / Confidentiality | `EXACT` |
| `AGENTRISK-048` | **4** | **3** | 54.4% | Physical / Health, Security / System Integrity | `NEAR (±1)` |
| `AGENTRISK-089` | **1** | **1** | 76.0% | No Material Harm | `EXACT` |
| `AGENTRISK-138` | **1** | **1** | 74.6% | No Material Harm | `EXACT` |
| `AGENTRISK-172` | **4** | **1** | 75.6% | Financial / Assets, Security / System Integrity | `MISS` |
| `AGENTRISK-216` | **1** | **1** | 79.4% | No Material Harm | `EXACT` |
| `AGENTRISK-236` | **1** | **3** | 61.0% | No Material Harm | `MISS` |
| `AGENTRISK-286` | **1** | **1** | 75.5% | No Material Harm | `EXACT` |
| `AGENTRISK-334` | **4** | **3** | 56.4% | Privacy / Confidentiality | `NEAR (±1)` |
| `AGENTRISK-360` | **4** | **3** | 52.1% | Privacy / Confidentiality | `NEAR (±1)` |
| `AGENTRISK-406` | **1** | **1** | 90.3% | No Material Harm | `EXACT` |
| `AGENTRISK-452` | **4** | **4** | 48.4% | Privacy / Confidentiality | `EXACT` |
| `AGENTRISK-490` | **1** | **1** | 74.5% | No Material Harm | `EXACT` |
| `AGENTRISK-521` | **3** | **3** | 60.4% | Privacy / Confidentiality, Financial / Assets | `EXACT` |
| `AGENTRISK-553` | **3** | **1** | 78.1% | Privacy / Confidentiality | `MISS` |
| `AGENTRISK-593` | **1** | **1** | 67.3% | No Material Harm | `EXACT` |
| `AGENTRISK-621` | **1** | **1** | 75.1% | No Material Harm | `EXACT` |
| `AGENTRISK-696` | **2** | **2** | 52.8% | Privacy / Confidentiality | `EXACT` |
| `SYN-D02-DV-002` | **2** | **2** | 60.9% | Privacy / Confidentiality | `EXACT` |
| `SYN-D05-DV-001` | **1** | **1** | 64.4% | Operational / Service | `EXACT` |
