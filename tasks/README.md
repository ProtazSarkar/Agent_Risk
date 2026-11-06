# `tasks/` - SemEval 2027 Shared Task Submissions & Benchmarks

## 🎯 Purpose
The `tasks/` directory stores **official benchmark submissions**, **evaluation outputs**, and **final predictions** for the **SemEval 2027 Shared Task on Risk Understanding in Agentic Systems**.

This folder isolates competition and shared-task submission artifacts from internal training pipelines, providing reproducible validation benchmarks and official score reports.

---

## 📁 Directory Content
- **`semeval_2027/`**: Final model inference predictions and official evaluation submission files:
  - **Subtask 1 (Risk Level Estimation)**: Predicted ordinal risk severities (`0` to `4`) for evaluation test splits.
  - **Subtask 2 (Risk Characterization)**: Multi-label classification predictions for *Harm Domains* and *Risk Drivers*.
  - **Subtask 3 (Policy-Guided Intervention)**: Prescribed intervention outputs (`allow`, `monitor_and_log`, `request_human_confirmation`, `modify_tool_parameters`, `block_action_and_halt`).
- **`leaderboard_metrics.json`**: Official or local benchmark performance records (Macro-F1, Quadratic Weighted Kappa, AUC-ROC).

---

## 📊 Generating Task Submissions
To run final model inference and generate formatted submission JSON files for the SemEval evaluation test split:

```bash
# Ingest the test split
python scripts/fetch_semeval_data.py --dataset AgentRisk/agentrisk-data-2027 --split test --output-dags data/processed/semeval_test_dags.json

# Run evaluation and output final predictions to tasks/
python experiments/stage_01/03_evaluate_baseline.py --input data/processed/semeval_test_dags.json --report tasks/semeval_2027/final_predictions.json
```

