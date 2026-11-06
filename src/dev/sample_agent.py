"""Sample experimental AI Agent prototype for risk analysis.

Provides a base agent structure with planning, step interception, and safety verification.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent_risk.planner.api_planner import APIPlanner
from agent_risk.planner.schema import PlanDAG, Route, TaskNode
from agent_risk.risk_engine.feature_extractor import FeatureExtractor
from agent_risk.risk_engine.xgboost_shield import RiskVerdict, XGBoostShield


class BaseDevAgent(ABC):
    """Abstract base class for experimental AI agents developed in src/dev."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.context: Dict[str, Any] = {}

    @abstractmethod
    def act(self, input_query: str) -> Dict[str, Any]:
        """Execute agent workflow on user input query."""
        pass


class RiskAnalysisAgent(BaseDevAgent):
    """Experimental agent that decomposes user goals and validates each action via the Risk Shield."""

    def __init__(
        self,
        name: str = "RiskInspectorAgent",
        role: str = "Autonomous Task Risk Auditor",
        warning_threshold: float = 0.35,
        block_threshold: float = 0.70,
    ):
        super().__init__(name=name, role=role)
        self.planner = APIPlanner(api_key=None)  # Auto-falls back if no API key
        self.extractor = FeatureExtractor()
        self.shield = XGBoostShield(warning_threshold=warning_threshold, block_threshold=block_threshold)
        self.shield.load_model()

    def inspect_task(self, task: TaskNode) -> Dict[str, Any]:
        """Evaluate a single task node for potential hazards."""
        feat = self.extractor.extract_task_features(task)
        import pandas as pd
        df = pd.DataFrame([feat])
        prob = float(self.shield.predict_proba(df)[0])
        verdict, _ = self.shield.evaluate_verdict(prob)
        return {
            "task_id": task.id,
            "task_name": task.name,
            "risk_score": prob,
            "verdict": verdict.value,
            "safe_to_execute": verdict != RiskVerdict.BLOCKED,
        }

    def act(self, input_query: str) -> Dict[str, Any]:
        """Plan tasks and assess overall execution route safety."""
        plan = self.planner.generate_plan(input_query)
        route = plan.primary_route()

        evaluations = [self.inspect_task(t) for t in route.tasks]
        has_blocked = any(not e["safe_to_execute"] for e in evaluations)

        return {
            "agent": self.name,
            "query": input_query,
            "plan_id": plan.plan_id,
            "tasks_count": len(route.tasks),
            "evaluations": evaluations,
            "final_permission": "DENIED" if has_blocked else "GRANTED",
        }

