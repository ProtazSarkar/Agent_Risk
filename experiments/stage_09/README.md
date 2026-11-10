# Stage 09: Dedicated Ordinal Severity Classification Suite (Levels 1 to 5)

## Overview
Stage 09 resolves the **Exact Severity Accuracy** bottleneck identified in [Stage 06](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_06/README.md). While Stage 06 achieved strong within-1 accuracy (85.80%), its exact accuracy stalled at **59.66%** due to continuous regression boundary blurring and unconstrained multi-class loss on single-vector interaction summaries.

Stage 09 solves this by combining:
1. **5-Channel Disentangled Interaction Signals** (`[query, thought, context_actions, observations, proposed_action]`) evaluated through all 11 [Stage 08](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_08/README.md) Cross-Attentive Prototype models.
2. **Frank & Hall (2001) Cumulative Link Ordinal Decomposition** enforcing monotonic threshold probabilities:
   $$P(Y = 1) = 1 - P(Y > 1)$$
   $$P(Y = k) = P(Y > k-1) - P(Y > k) \quad \text{for } k \in \{2, 3, 4\}$$
   $$P(Y = 5) = P(Y > 4)$$
3. **Calibrated Confidence Estimation** that scales up to **81.82% Exact Accuracy** at high confidence ($\ge 0.90$).

---

## Benchmark Results (Held-Out SemEval Dev Set, N=143)

| Metric | Stage 06 (Previous) | Stage 09 (Champion) | Improvement |
|---|---|---|---|
| **Exact Severity Accuracy** | 59.66% | **73.43%** | **+13.77% absolute gain** |
| **Within-1 Accuracy ($\pm 1$)** | 85.80% | **91.61%** | **+5.81% absolute gain** |
| **Quadratic Weighted Kappa (QWK)** | 0.6895 | **0.7523** | **+0.0628 gain** |
| **Mean Absolute Error (MAE)** | 0.5966 | **0.4056** | **32.0% error reduction** |
| **Root Mean Squared Error (RMSE)** | 1.0417 | **0.9007** | **13.5% error reduction** |
| **High-Confidence Exact Acc ($\ge 0.9$)** | 71.40% | **81.82%** | **+10.42% gain** |

---

## Directory Structure
```
experiments/stage_09/
├── README.md                          # Quickstart and overview documentation
├── prepare_dataset.py                 # Extracts 39 tabular features from 11 Stage 08 models
├── models.py                          # Frank-Hall Ordinal, Hierarchical Gating, Regressors, Ensembles
├── train.py                           # 5-Fold Stratified Cross-Validation & Champion Model saving
├── main.py                            # Master evaluation on held-out SemEval dev set
├── data/
│   ├── train_severity_features_stage09.csv # 568 training rows x 41 columns
│   └── dev_severity_features_stage09.csv   # 143 dev rows x 41 columns
├── models/
│   └── best_severity_model_stage09.joblib  # Serialized champion model checkpoint
├── output/
│   ├── feature_summary.md             # Feature statistics and Pearson correlation table
│   ├── train_cv_benchmark.json        # 5-Fold CV metrics across all 7 candidate models
│   ├── severity_benchmark_results.json # Full benchmark payload & prediction audit
│   └── severity_benchmark_report.md   # Markdown report with confusion matrix & breakdown
└── docs/
    ├── arch.md                        # Theoretical architecture & Mermaid pipeline flowchart
    └── report.md                      # Comprehensive research findings & Stage 06 vs 09 analysis
```

---

## Quickstart & Replication

```bash
# 1. Generate the 39-feature datasets from Stage 08 models
python experiments/stage_09/prepare_dataset.py

# 2. Run 5-Fold Stratified Cross-Validation across candidate models
python experiments/stage_09/train.py

# 3. Evaluate champion and candidate suite on held-out SemEval dev data
python experiments/stage_09/main.py
```

