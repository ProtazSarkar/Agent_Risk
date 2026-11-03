# `experiments/stage_01/` - Baseline DAG Generation & XGBoost Shield

## 🎯 Purpose
Stage 01 provides the core baseline research workshop. It connects the LLM Task Planner directly to the Feature Extractor and XGBoost Shield, establishing empirical benchmarks for runtime safety prediction.

---

## 📁 Directory Content
- **`config.yaml`**: Stage configuration parameters (model name, temperatures, paths, classification thresholds).
- **`01_generate_dags.py`**: Reads prompts from `data/sample/sample_prompts.json` or SemEval datasets and outputs structured DAGs to `data/processed/stage_01_dags.json`.
- **`02_train_xgboost.py`**: Extracts 24 tabular features across all tasks and trains the `XGBoostShield`, saving serialized model artifacts to `models/xgboost/risk_classifier.json`.
- **`03_evaluate_baseline.py`**: Evaluates the trained shield on test/validation DAGs, reporting classification metrics and saving results to `docs/reports/stage_01_baseline_report.json`.

---

## 🔬 Execution Instructions
```bash
# Step 1: Decompose prompts into DAGs
python experiments/stage_01/01_generate_dags.py --mock

# Step 2: Extract features & train model
python experiments/stage_01/02_train_xgboost.py

# Step 3: Evaluate & generate report
python experiments/stage_01/03_evaluate_baseline.py
```
