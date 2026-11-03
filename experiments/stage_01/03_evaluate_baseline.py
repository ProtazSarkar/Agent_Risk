"""Stage 01 - Step 3: Evaluate Baseline XGBoost Shield on Test Set."""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from agent_risk.config import settings
from agent_risk.planner.schema import PlanDAG
from agent_risk.risk_engine.feature_extractor import FeatureExtractor
from agent_risk.risk_engine.xgboost_shield import XGBoostShield
from agent_risk.utils.evaluator import evaluate_risk_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(dags_file: Path, model_file: Path, report_output: Path):
    logger.info(f"Loading evaluation DAGs from: {dags_file}")
    with open(dags_file, "r", encoding="utf-8") as f:
        dags_data = json.load(f)

    extractor = FeatureExtractor()
    shield = XGBoostShield(model_path=model_file)
    shield.load_model()

    all_rows = []
    for item in dags_data:
        plan = PlanDAG.model_validate(item)
        df_plan = extractor.extract_dag_dataframe(plan)
        gt_risk = float(item.get("ground_truth_risk", 0.0))
        df_plan["label"] = 1 if gt_risk > 0.4 else 0
        all_rows.append(df_plan)

    dataset = pd.concat(all_rows, ignore_index=True)
    X = dataset[extractor.feature_names]
    y_true = dataset["label"].values

    probs = shield.predict_proba(X)
    preds = (probs >= shield.block_threshold).astype(int)

    metrics = evaluate_risk_predictions(y_true, preds, probs)
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved evaluation report to: {report_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline XGBoost Shield.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.processed_data_dir / "stage_01_dags.json",
        help="Path to evaluation DAGs JSON",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.xgboost_model_path,
        help="Path to trained XGBoost model",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=settings.root_dir / "docs" / "reports" / "stage_01_baseline_report.json",
        help="Path to save evaluation report JSON",
    )
    args = parser.parse_args()

    run(args.input, args.model, args.report)

