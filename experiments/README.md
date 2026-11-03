# `experiments/` - Research & Empirical Workshops

## 🎯 Purpose
The `experiments/` directory holds research workshops, reproducibility pipelines, and model evaluation routines. Per project design, the active empirical focus is maintained in **`stage_01/`**.

---

## 📁 Directory Content
- **[`stage_01/`](stage_01/)**: End-to-end baseline pipeline implementing:
  - Step 1: Automatic DAG decomposition from prompt suites.
  - Step 2: Tabular psycholinguistic feature extraction and XGBoost shield model training.
  - Step 3: Multi-metric evaluation (F1, Precision, Recall, ROC-AUC, FPR) and report export.

---

## 🚀 Running Experiments
```bash
# Generate task DAGs
python experiments/stage_01/01_generate_dags.py --mock

# Train the risk shield classifier
python experiments/stage_01/02_train_xgboost.py

# Evaluate baseline accuracy and export report
python experiments/stage_01/03_evaluate_baseline.py
```
