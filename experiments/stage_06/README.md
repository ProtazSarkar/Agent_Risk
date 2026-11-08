# Stage 06: Multi-Domain Probability Ensemble & Tree/Regression Severity Suite (Levels 1 to 5)

## 🎯 Architecture Vision & Breakthrough

Stage 06 merges the **Multi-Domain Probability Ensemble** with the **Risk Level (1 to 5) Severity Predictor** into a unified, high-performance pipeline:

1. **Feature Preparation (`prepare_dataset.py`)**:
   - Ingests non-truncated interaction summaries and feeds them through all **11 trained Stage 05 Semantic Prototype Classifiers** (`domain_01` to `domain_11`).
   - Extracts a rich 37-dimensional tabular feature representation:
     - **11 Calibrated Domain Risk Probabilities** ($P_{\text{no\_harm}}, P_{\text{privacy}}, \dots, P_{\text{info\_quality}}$)
     - **Compound Multi-Domain Harm Aggregates**: `max_harm_prob` ($r = +0.7560$), `sum_harm_prob` ($r = +0.6617$), `harm_to_safe_ratio` ($r = +0.6715$), `domain_entropy`
     - **Multi-Domain Threshold Counters**: `num_domains_ge_05`, `num_domains_ge_02`
     - **Domain Rank Dynamics**: `top1_harm_prob`, `top2_harm_prob`, `top_harm_margin`
     - **High-Consequence Hazard Flags**: `is_physical_high`, `is_security_high`, `is_financial_high`
     - **Latent Semantic Projections**: 8 PCA components fitted on 384-d dense embeddings
     - **Text & Refusal Keywords**: `refusal_keyword_count` ($r = -0.5427$), `critical_keyword_count` ($r = +0.3329$), `text_word_count`
   - Exports `train_severity_features.csv` (639 rows) and `dev_severity_features.csv` (176 rows).

2. **Model Training & Selection (`models.py`, `train.py`)**:
   - Benchmarks tree-based classifiers (`XGBoost_Classifier`, `HistGradientBoosting_Classifier`, `RandomForest_Classifier`), ordinal regressors (`XGBoost_Ordinal_Regressor`, `HistGradientBoosting_Regressor`, `RandomForest_Regressor`, `Ridge_Ordinal_Regressor`), and a hybrid ensemble (`Hybrid_XGB_Ensemble`).
   - Evaluates via **5-Fold Stratified Cross-Validation** on the training split.
   - Fits the champion model (`HistGradientBoosting_Regressor` with Nelder-Mead ordinal threshold optimization) and persists checkpoints to `models/`.

3. **Evaluation Gateway (`main.py`)**:
   - Evaluates on the held-out development set (both 176 full dev samples and 143 raw SemEval samples).
   - Generates exact accuracy, adjacent within-1 accuracy, QWK, MAE, RMSE, Macro F1, 5x5 confusion matrix, prediction confidence, and a 20-sample prediction trace.

---

## 📁 Directory Structure

```text
experiments/stage_06/
├── README.md                               # Stage 06 documentation
├── prepare_dataset.py                      # Multi-domain probability feature extraction pipeline
├── models.py                               # Tree-based & regression model wrappers with threshold tuning
├── train.py                                # 5-Fold Stratified Cross-Validation & training suite
├── main.py                                 # Evaluation gateway, confidence analysis & prediction trace
├── data/
│   ├── train_severity_features.csv         # 639 rows x 41 columns
│   └── dev_severity_features.csv           # 176 rows x 41 columns
├── models/
│   ├── best_severity_model.joblib          # Persisted champion model
│   └── *.joblib                            # All candidate checkpoints
└── output/
    ├── feature_summary.md                  # Feature statistics & correlation breakdown
    ├── training_cv_results.json            # 5-fold cross-validation results
    ├── severity_benchmark_results.json     # Complete test evaluation metrics
    └── severity_benchmark_report.md        # Comprehensive benchmark markdown report
```

---

## 📊 Benchmark Results (Held-Out Dev Set: 176 Instances)

| Candidate Model | QWK | Exact Acc (1-5) | Within-1 Acc | MAE | RMSE | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RandomForest_Regressor`** | **0.6990** | 59.66% | **87.50%** | 0.5852 | 1.0350 | 61.2% |
| **`HistGradientBoosting_Regressor` (Champ)** | **0.6895** | 59.66% | **85.80%** | 0.5966 | 1.0417 | 62.1% |
| **`XGBoost_Ordinal_Regressor`** | **0.6855** | 60.80% | **86.93%** | 0.5795 | 1.0347 | 62.4% |
| **`XGBoost_Classifier`** | **0.6805** | **63.64%** | **86.36%** | 0.5909 | 1.0375 | **79.1%** |
| **`Hybrid_XGB_Ensemble`** | **0.6700** | **64.20%** | **86.36%** | **0.5795** | **1.0287** | 67.4% |
| **`HistGradientBoosting_Classifier`** | 0.6420 | 60.80% | 85.23% | 0.6420 | 1.0963 | 85.2% |
| **`RandomForest_Classifier`** | 0.5981 | 60.23% | 82.39% | 0.6932 | 1.1345 | 64.3% |
| **`Ridge_Ordinal_Regressor`** | 0.5895 | 50.00% | 82.39% | 0.7557 | 1.1648 | 60.3% |

### Key Benchmark Metrics (SemEval Native Held-Out Split: 143 Instances):
- **Exact Accuracy (1 to 5)**: **62.24%** (Up from 6.29% in Stage 01 and 40.56% in Stage 03)
- **Within-1 (Adjacent) Accuracy**: **86.71%**
- **Quadratic Weighted Kappa (QWK)**: **0.6805** (Over 3X higher than Stage 03's 0.2239)
- **Mean Absolute Error (MAE)**: **0.5966** severity levels
- **High-Confidence Predictions ($\ge 70\%$)**: **89.80% exact accuracy**

---

## 5x5 Confusion Matrix (Held-Out Dev Set)

```text
       Pred 1   Pred 2   Pred 3   Pred 4   Pred 5
-------------------------------------------------
True 1 |   58       11        4        3        0
True 2 |    5        8        0        2        0
True 3 |    2        5       19        9        0
True 4 |    2        6       12       16        2
True 5 |    0        4        2        2        4
```

---

## 🚀 How to Run Stage 06

```powershell
# 1. Extract 11-domain probabilities and prepare the 37-feature tabular dataset:
python experiments/stage_06/prepare_dataset.py

# 2. Train and select champion model via 5-fold cross-validation:
python experiments/stage_06/train.py

# 3. Run the evaluation gateway and view prediction trace & confidence analysis:
python experiments/stage_06/main.py
```
