# `experiments/` - Research & Empirical Workshops

## 🎯 Purpose
The `experiments/` directory holds empirical research pipelines, tabular feature formulations, and progressive modeling stages for SemEval 2027 AgentRisk.

---

## 📁 Stages Overview

- **[`stage_01/`](stage_01/) - Baseline Raw Dense Embeddings (384-d)**:
  - Embeds interactions via `all-MiniLM-L6-v2` into 384 dense features.
  - Documents why high dimensionality vs low sample size ($p \approx n$) and unweighted multi-class classification failed on ordinal risk targets.
  - Benchmarks XGBoost, Ridge, and SVR regressors with Nelder-Mead threshold optimization.

- **[`stage_02/`](stage_02/) - Pure Domain Label Similarities (5-d)**:
  - Completely drops the 384 raw embeddings to test if 5 high-level domain prototype similarities ($l_1 \dots l_5$) are sufficient.
  - Identifies the *Feature Overlap* and *Dynamic Range Compression* bottlenecks where uncalibrated models collapsed under fixed user ranges.

- **[`stage_03/`](stage_03/) - Hybrid 394-Dimensional Architecture (Embeddings + Domain Projections)**:
  - Resolves both Stage 01 and Stage 02 limitations by combining 384 dense embeddings + 5 domain similarities + 5 attention weights ($384 + 5 + 5 = 394$ features).
  - Uses enhanced multi-perspective domain prototype descriptions in `label.py`.
  - Implements Dual-Branch Deep Neural Regressors with Dynamic Range Calibration ($val_{\text{cal}} = \alpha \cdot val + \beta$) and Deep Classifiers.
  - Lifts exact accuracy from **6.29%** to **40.56%** and QWK to **0.2239**.

- **[`stage_04/`](stage_04/) - Domain-Expert Binary Classifiers & Multi-Prototype Anchors**:
  - Investigates single-domain classification ('No Material Harm').
  - Discovered the 100% action truncation defect in Stage 01/04 embeddings where 256-token cutoffs dropped proposed actions.
  - Implemented multi-prototype anchors for safe refusals and temperature scaling ($\gamma = 10.0$).

- **[`stage_05/`](stage_05/) - Context Distillation & 11-Domain Semantic Classifier Suite**:
  - Solves the 100% action truncation defect by distilling interactions into dense 60–90 word summaries (supporting Groq API & deterministic structural extraction).
  - Encodes summaries with `all-MiniLM-L6-v2` with **0.0% truncation** (100% context retained).
  - Deploys 11 isolated domain prototype classifiers achieving **91.25% Macro Accuracy** and **0.8285 Macro ROC-AUC**.

- **[`stage_06/`](stage_06/) - Multi-Domain Probability Ensemble & Feature Preparation Pipeline**:
  - Bridges Stage 05 domain models with Stage 07 severity models.
  - Extracts 37 engineered tabular features: 11 domain probabilities, compound harm statistics, domain entropy, high-consequence flags, PCA latents, and text keyword indicators.
  - Generates `train_severity_features.csv` and `dev_severity_features.csv`.

- **[`stage_07/`](stage_07/) - Tree-Based & Regression Severity Suite (Levels 1 to 5)**:
  - Final ordinal risk severity predictor evaluating XGBoost, HistGradientBoosting, Random Forest, Ordinal Regressors, and Hybrid Ensembles.
  - Generates predictive confidence scores ($\max_k P(y=k)$) for every prediction.
  - Achieves **62.24% Exact Accuracy**, **86.71% Within-1 Accuracy**, **0.6805 QWK**, and **89.80% High-Confidence Accuracy** on held-out SemEval dev data.

---

## 🚀 Running Experiments

```powershell
# Stage 01 Benchmark
python experiments/stage_01/benchmark.py

# Stage 02 Evaluation Gateway & Deep Learning Suite
python experiments/stage_02/nuron_train.py

# Stage 03 Hybrid Benchmark
python experiments/stage_03/main.py

# Stage 04 Domain Classifier
python experiments/stage_04/domain_01/main.py

# Stage 05 Master 11-Domain Benchmark Suite
python experiments/stage_05/run_all_domains.py

# Stage 06 Multi-Domain Feature Preparation Pipeline
python experiments/stage_06/prepare_dataset.py

# Stage 07 Severity Training & Evaluation
python experiments/stage_07/train.py
python experiments/stage_07/main.py
```
