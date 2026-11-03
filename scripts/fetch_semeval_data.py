"""CLI script to download, cache, and convert 'AgentRisk/agentrisk-data-2027' into DAGs."""

import argparse
import json
import logging
from pathlib import Path

from agent_risk.config import settings
from agent_risk.data.loader import AgentRiskDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch and ingest AgentRisk/agentrisk-data-2027.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="AgentRisk/agentrisk-data-2027",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train, dev, test)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode using cached or synthetic benchmark instances",
    )
    parser.add_argument(
        "--output-dags",
        type=Path,
        default=settings.processed_data_dir / "semeval_2027_dags.json",
        help="Destination path for converted DAGs JSON",
    )
    args = parser.parse_args()

    logger.info(f"Loading '{args.dataset}' [split={args.split}]...")
    instances = AgentRiskDatasetLoader.load(
        dataset_name=args.dataset,
        split=args.split,
        force_offline=args.offline,
    )
    logger.info(f"Loaded {len(instances)} instances.")

    # Convert interaction traces to PlanDAG format
    logger.info(f"Converting instances to structured PlanDAG format...")
    dags = [inst.to_plan_dag().model_dump() for inst in instances]

    args.output_dags.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_dags, "w", encoding="utf-8") as f:
        json.dump(dags, f, indent=2)

    logger.info(f"Successfully converted and saved {len(dags)} DAGs to {args.output_dags}")


if __name__ == "__main__":
    main()

