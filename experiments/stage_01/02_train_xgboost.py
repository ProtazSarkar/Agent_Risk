"""Stage 01 - Step 2: Train XGBoost Shield on Extracted DAG & Psycholinguistic Features."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from agent_risk.config import settings
from agent_risk.planner.schema import PlanDAG
from agent_risk.risk_engine.feature_extractor import FeatureExtractor
from agent_risk.risk_engine.xgboost_shield import XGBoostShield

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(dags_file: Path, model_output: Path):
    logger.info(f"Loading generated DAGs from: {dags_file}")
    with open(dags_file, "r", encoding="utf-8") as f:
        dags_data = json.load(f)

    extractor = FeatureExtractor()
    all_rows = []

    for item in dags_data:
        plan = PlanDAG.model_validate(item)
        df_plan = extractor.extract_dag_dataframe(plan)
        gt_risk = float(item.get("ground_truth_risk", float(item.get("metadata", {}).get("risk_level", 0.0)) / 4.0))
        
        labels = []
        for task in plan.primary_route().tasks:
            # Task is risky if explicitly scored > 0.4 or if overall plan is high risk and task is sensitive
            is_risky = 1 if (task.risk_score > 0.4 or (gt_risk > 0.4 and task.action_type in ["command_exec", "network_egress"])) else 0
            labels.append(is_risky)
        df_plan["label"] = labels
        all_rows.append(df_plan)

    dataset = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Extracted feature dataset with {len(dataset)} task samples.")

    # Balance and bootstrap classes if dataset is small for baseline demonstration
    pos_mask = dataset["label"] == 1
    neg_mask = dataset["label"] == 0
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        n_max = max(pos_mask.sum(), neg_mask.sum())
        pos_upsampled = dataset[pos_mask].sample(n=n_max, replace=True, random_state=42)
        neg_upsampled = dataset[neg_mask].sample(n=n_max, replace=True, random_state=42)
        dataset = pd.concat([pos_upsampled, neg_upsampled] * 2, ignore_index=True)
        # Add slight jitter to continuous features
        for col in ["text_length", "entropy"]:
            if col in dataset.columns:
                dataset[col] += np.random.normal(0, 0.05, size=len(dataset))

    feature_cols = extractor.feature_names
    X = dataset[feature_cols]
    y = dataset["label"]

    shield = XGBoostShield(model_path=model_output)
    metrics = shield.train(X, y, save=True)

    logger.info(f"Training completed successfully! Metrics: {metrics}")
    logger.info(f"Model saved to {model_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Risk Shield.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.processed_data_dir / "stage_01_dags.json",
        help="Path to generated DAGs JSON",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=settings.xgboost_model_path,
        help="Path to save serialized XGBoost model",
    )
    args = parser.parse_args()

    run(args.input, args.model_output)

