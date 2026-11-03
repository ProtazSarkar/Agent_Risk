"""Necessity and Redundancy Scorer for Agent Task DAGs.

Identifies superfluous, disconnected, or unneeded tasks that introduce unnecessary
attack surface or latency into agent execution trajectories.
"""

import logging
from typing import Dict, List, Set, Tuple
import networkx as nx

from agent_risk.config import settings
from agent_risk.planner.schema import Route, TaskNode

logger = logging.getLogger(__name__)


class NecessityFilter:
    """Evaluates task indispensability and redundancy within an execution route DAG."""

    def __init__(self, necessity_threshold: float = None):
        self.necessity_threshold = necessity_threshold or settings.necessity_threshold

    def score_route(self, route: Route) -> Dict[str, float]:
        """Compute necessity scores for all tasks in a route.

        Returns a dictionary mapping task_id -> necessity_score in [0.0, 1.0].
        """
        graph = route.to_networkx()
        scores: Dict[str, float] = {}

        if graph.number_of_nodes() == 0:
            return scores

        # Identify terminal (leaf) nodes with out-degree 0
        all_leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
        # Critical goal leaves take precedence for determining purposeful execution paths
        critical_leaves = [
            n for n in all_leaves if graph.nodes[n].get("is_critical", False)
        ]
        target_leaves = critical_leaves if critical_leaves else all_leaves

        for task in route.tasks:
            # 1. Base necessity from criticality flag
            base_score = 0.5 if not task.is_critical else 0.8

            # 2. Reachability to target goal terminals
            reaches_leaf = False
            for leaf in target_leaves:
                if task.id == leaf or nx.has_path(graph, task.id, leaf):
                    reaches_leaf = True
                    break

            if not reaches_leaf:
                # Disconnected branch / dead branch not contributing to goal
                scores[task.id] = 0.05
                task.necessity_score = 0.05
                continue

            # 3. Centrality and dependency load
            # Tasks that many downstream nodes depend on are strictly more necessary
            descendants = nx.descendants(graph, task.id)
            descendant_ratio = len(descendants) / max(1, graph.number_of_nodes() - 1)

            final_score = min(1.0, base_score + 0.3 * descendant_ratio)
            scores[task.id] = round(final_score, 4)
            task.necessity_score = scores[task.id]

        return scores

    def filter_unnecessary_tasks(
        self,
        route: Route,
    ) -> Tuple[List[TaskNode], List[TaskNode]]:
        """Separate tasks into necessary tasks and pruned/flagged tasks.

        Returns:
            (necessary_tasks, pruned_tasks)
        """
        scores = self.score_route(route)
        necessary: List[TaskNode] = []
        pruned: List[TaskNode] = []

        for task in route.tasks:
            if scores.get(task.id, 0.0) >= self.necessity_threshold:
                necessary.append(task)
            else:
                pruned.append(task)

        return necessary, pruned
