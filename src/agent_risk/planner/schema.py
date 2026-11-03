"""Pydantic schemas and graph representations for agentic task decomposition."""

from enum import Enum
from typing import Any, Dict, List, Optional
import networkx as nx
from pydantic import BaseModel, Field, model_validator


class TaskActionType(str, Enum):
    """Categorization of agentic action types."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    COMMAND_EXEC = "command_exec"
    NETWORK_EGRESS = "network_egress"
    NETWORK_INGRESS = "network_ingress"
    DATABASE_QUERY = "database_query"
    COMPUTATION = "computation"
    HUMAN_INTERACTION = "human_interaction"
    OTHER = "other"


class TaskNode(BaseModel):
    """An atomic step/task within an execution DAG."""
    id: str = Field(description="Unique task identifier, e.g. task-1")
    name: str = Field(description="Short semantic task name")
    description: str = Field(description="Detailed description of what the action performs")
    action_type: str = Field(default=TaskActionType.COMPUTATION.value, description="Action category")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Parameters or dependencies data")
    expected_output: str = Field(default="", description="Description of the expected artifact or result")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must execute before this")
    is_critical: bool = Field(default=False, description="Whether this task is strictly required for goal completion")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Predicted risk score [0, 1]")
    necessity_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Task necessity score [0, 1]")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extension metadata")


class Route(BaseModel):
    """An alternate execution pathway decomposing a goal into a DAG of tasks."""
    route_id: str = Field(description="Unique route identifier, e.g. primary-route or fallback-route")
    rationale: str = Field(description="Why this specific sequence/route was planned")
    tasks: List[TaskNode] = Field(default_factory=list, description="Ordered or dependent sequence of tasks")

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Find a task node by id."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def to_networkx(self) -> nx.DiGraph:
        """Convert this route's tasks and dependencies into a directed acyclic graph."""
        graph = nx.DiGraph(route_id=self.route_id, rationale=self.rationale)
        for task in self.tasks:
            graph.add_node(
                task.id,
                name=task.name,
                description=task.description,
                action_type=task.action_type,
                risk_score=task.risk_score,
                necessity_score=task.necessity_score,
                is_critical=task.is_critical,
            )

        for task in self.tasks:
            for dep_id in task.dependencies:
                if dep_id in graph:
                    graph.add_edge(dep_id, task.id)

        return graph

    def is_acyclic(self) -> bool:
        """Check if this route forms a valid Directed Acyclic Graph."""
        return nx.is_directed_acyclic_graph(self.to_networkx())


class PlanDAG(BaseModel):
    """The complete multi-route task graph for an agentic goal."""
    plan_id: str = Field(description="Unique plan identifier")
    goal: str = Field(description="The primary user prompt or objective being fulfilled")
    routes: List[Route] = Field(default_factory=list, description="Candidate execution routes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or generator details")

    @model_validator(mode="after")
    def validate_routes_exist(self) -> "PlanDAG":
        if not self.routes:
            raise ValueError("PlanDAG must contain at least one route.")
        return self

    def primary_route(self) -> Route:
        """Convenience method to retrieve the primary or first route."""
        return self.routes[0]

    def to_networkx(self, route_id: Optional[str] = None) -> nx.DiGraph:
        """Extract a NetworkX DiGraph for a specific route or the primary route."""
        if route_id is not None:
            for r in self.routes:
                if r.route_id == route_id:
                    return r.to_networkx()
            raise KeyError(f"Route '{route_id}' not found in plan.")
        return self.primary_route().to_networkx()

