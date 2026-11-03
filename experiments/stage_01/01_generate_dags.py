"""Stage 01 - Step 1: Generate Task DAGs from Input Prompts using API Planner."""

import argparse
import json
import logging
from pathlib import Path

from agent_risk.config import settings
from agent_risk.planner.api_planner import APIPlanner
from agent_risk.utils.metrics import profile_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(prompts_file: Path, output_file: Path, mock: bool = False):
    logger.info(f"Loading input prompts from: {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    planner = APIPlanner()
    if mock:
        planner._client = None

    generated_plans = []
    logger.info(f"Generating plans for {len(prompts_data)} prompts...")

    with profile_block("DAG_Generation") as profiler:
        for item in prompts_data:
            prompt_text = item.get("prompt", "")
            logger.info(f"Processing prompt [{item.get('id')}]: {prompt_text[:50]}...")
            plan = planner.generate_plan(prompt_text)
            plan_dict = plan.model_dump()
            # Annotate with ground truth if present
            plan_dict["ground_truth_risk"] = item.get("ground_truth_risk", 0.0)
            plan_dict["ground_truth_necessity"] = item.get("ground_truth_necessity", 1.0)
            generated_plans.append(plan_dict)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_plans, f, indent=2)

    logger.info(
        f"Completed {len(generated_plans)} plans in {profiler.elapsed_time_sec:.2f}s "
        f"-> Saved to {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task DAGs via API Planner.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.data_dir / "sample" / "sample_prompts.json",
        help="Path to input prompts JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.processed_data_dir / "stage_01_dags.json",
        help="Path to output DAGs JSON",
    )
    parser.add_argument("--mock", action="store_true", help="Force mock offline planner")
    args = parser.parse_args()

    run(args.input, args.output, args.mock)

