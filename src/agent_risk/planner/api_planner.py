"""API Planner: LLM interface (Groq / OpenAI-compatible) for DAG plan decomposition."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agent_risk.config import settings
from agent_risk.planner.schema import PlanDAG, Route, TaskNode

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are an Agentic Task Decomposition Planner.
Given a user prompt/goal, break it down into a directed acyclic graph (DAG) of explicit atomic steps.
You MUST output strictly valid JSON conforming to the following structure:
{
  "plan_id": "plan-<uuid>",
  "goal": "<user prompt>",
  "routes": [
    {
      "route_id": "primary-route",
      "rationale": "<reasoning for this step decomposition>",
      "tasks": [
        {
          "id": "task-1",
          "name": "<action_name>",
          "description": "<detailed action description>",
          "action_type": "<file_read|file_write|command_exec|network_egress|network_ingress|computation|human_interaction|other>",
          "inputs": {},
          "expected_output": "<output specification>",
          "dependencies": [],
          "is_critical": true
        }
      ]
    }
  ]
}
Ensure all dependencies refer only to previously declared task IDs (no cycles).
Do not wrap your answer in markdown formatting or commentary; output raw JSON only.
"""


class BasePlanner(ABC):
    """Abstract Base Class for Task Decomposition Planners."""

    @abstractmethod
    def generate_plan(self, prompt: str, **kwargs: Any) -> PlanDAG:
        """Decompose a prompt into a PlanDAG."""
        pass


class APIPlanner(BasePlanner):
    """LLM API interface (Groq, OpenAI, or compatible endpoints)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "groq",
    ):
        self.api_key = api_key or settings.groq_api_key or settings.openai_api_key
        self.model = model or settings.default_llm_model
        self.provider = provider.lower()
        self._client = self._init_client()

    def _init_client(self) -> Any:
        if not self.api_key:
            logger.warning(
                "No API key provided for APIPlanner. Will fallback to deterministic mock planner."
            )
            return None

        if self.provider == "groq":
            try:
                from groq import Groq
                return Groq(api_key=self.api_key)
            except ImportError:
                logger.warning("groq library not installed. Falling back to mock planner.")
                return None
        return None

    def _mock_plan(self, prompt: str) -> PlanDAG:
        """Fallback deterministic plan generation for testing and offline development."""
        words = prompt.strip().split()
        first_few = "_".join(words[:3]).lower() if words else "task"

        t1 = TaskNode(
            id="task-1",
            name=f"prepare_{first_few}",
            description=f"Validate inputs and prerequisites for: {prompt[:80]}",
            action_type="computation",
            inputs={"prompt": prompt},
            expected_output="Validated configuration",
            dependencies=[],
            is_critical=True,
        )
        t2 = TaskNode(
            id="task-2",
            name=f"execute_{first_few}",
            description=f"Execute core logic for objective: {prompt[:80]}",
            action_type="command_exec" if any(w in prompt.lower() for w in ["run", "execute", "firewall", "script"]) else "file_read",
            inputs={"context": "task-1.output"},
            expected_output="Execution results",
            dependencies=["task-1"],
            is_critical=True,
        )
        t3 = TaskNode(
            id="task-3",
            name="finalize_and_report",
            description="Collect output metrics and format response to user",
            action_type="network_egress" if any(w in prompt.lower() for w in ["email", "upload", "send"]) else "computation",
            inputs={"results": "task-2.output"},
            expected_output="Final user report",
            dependencies=["task-2"],
            is_critical=False,
        )

        route = Route(
            route_id="primary-route",
            rationale="Standard 3-stage validation-execution-reporting pipeline",
            tasks=[t1, t2, t3],
        )

        return PlanDAG(
            plan_id=f"plan-mock-{hash(prompt) % 10000:04d}",
            goal=prompt,
            routes=[route],
            metadata={"planner": "mock_fallback"},
        )

    def generate_plan(self, prompt: str, **kwargs: Any) -> PlanDAG:
        """Decompose prompt into a PlanDAG using the API or fallback mock."""
        if self._client is None:
            return self._mock_plan(prompt)

        try:
            if self.provider == "groq":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=kwargs.get("temperature", 0.2),
                )
                content = response.choices[0].message.content
                data = json.loads(content)
                return PlanDAG.model_validate(data)

        except Exception as e:
            logger.error(f"Error calling LLM API planner: {e}. Falling back to mock plan.")
            return self._mock_plan(prompt)

        return self._mock_plan(prompt)

