"""Feature extraction module for risk shielding.

Converts task nodes, DAG topology, and psycholinguistic text markers into
tabular numerical features for XGBoost risk classification.
"""

import re
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import networkx as nx

from agent_risk.planner.schema import PlanDAG, Route, TaskNode


# Psycholinguistic keyword banks
URGENCY_KEYWORDS = [
    "immediately", "urgent", "critical", "now", "emergency", "instant",
    "override", "force", "bypass", "quick", "asap", "fast"
]

DECEPTION_KEYWORDS = [
    "pretend", "simulate", "hypothetical", "ignore", "disregard",
    "secret", "hidden", "stealth", "covert", "evade", "obfuscate"
]

PRIVILEGE_KEYWORDS = [
    "sudo", "admin", "root", "chmod", "firewall", "registry",
    "privilege", "elevated", "system32", "auth", "token", "credential", "password"
]

EXFILTRATION_KEYWORDS = [
    "curl", "wget", "pastebin", "webhook", "dns", "base64",
    "exfiltrate", "dump", "leak", "tunnel", "backdoor", "steal"
]


class FeatureExtractor:
    """Extracts tabular psycholinguistic and topological features from tasks and DAGs."""

    def __init__(self):
        self.feature_names = [
            # Psycholinguistic & lexical features
            "text_length",
            "word_count",
            "urgency_count",
            "deception_count",
            "privilege_count",
            "exfiltration_count",
            "imperative_ratio",
            "entropy",
            # Action type one-hot indicators
            "is_file_read",
            "is_file_write",
            "is_command_exec",
            "is_network_egress",
            "is_network_ingress",
            "is_database_query",
            "is_computation",
            # Graph topological features
            "dag_node_count",
            "dag_edge_count",
            "in_degree",
            "out_degree",
            "is_root_node",
            "is_leaf_node",
            "is_critical_flag",
        ]

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of the text character distribution."""
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        return -float(sum(p * np.log2(p) for p in prob if p > 0))

    def extract_task_features(
        self,
        task: TaskNode,
        dag_graph: Union[nx.DiGraph, None] = None,
    ) -> Dict[str, float]:
        """Extract a single feature vector dictionary for a specific task."""
        text = f"{task.name} {task.description} {json_dumps_safe(task.inputs)} {task.expected_output}".lower()
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)

        urgency_hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text)
        deception_hits = sum(1 for kw in DECEPTION_KEYWORDS if kw in text)
        privilege_hits = sum(1 for kw in PRIVILEGE_KEYWORDS if kw in text)
        exfiltration_hits = sum(1 for kw in EXFILTRATION_KEYWORDS if kw in text)

        # Basic heuristic for imperative verbs (command-style tasks)
        imperatives = sum(1 for w in words if w.endswith(("ed", "ing", "e", "run", "do", "get", "send")))
        imperative_ratio = (imperatives / word_count) if word_count > 0 else 0.0

        # Action type flags
        act = task.action_type.lower()
        is_read = float("read" in act)
        is_write = float("write" in act)
        is_exec = float("exec" in act or "command" in act)
        is_egress = float("egress" in act or "network" in act and "send" in text)
        is_ingress = float("ingress" in act)
        is_db = float("data" in act or "db" in act or "query" in act)
        is_comp = float("comp" in act)

        # Topological attributes
        if dag_graph is not None and task.id in dag_graph:
            node_count = float(dag_graph.number_of_nodes())
            edge_count = float(dag_graph.number_of_edges())
            in_deg = float(dag_graph.in_degree(task.id))
            out_deg = float(dag_graph.out_degree(task.id))
            is_root = 1.0 if in_deg == 0 else 0.0
            is_leaf = 1.0 if out_deg == 0 else 0.0
        else:
            node_count = 1.0
            edge_count = float(len(task.dependencies))
            in_deg = float(len(task.dependencies))
            out_deg = 0.0
            is_root = 1.0 if in_deg == 0 else 0.0
            is_leaf = 1.0

        features: Dict[str, float] = {
            "text_length": float(len(text)),
            "word_count": float(word_count),
            "urgency_count": float(urgency_hits),
            "deception_count": float(deception_hits),
            "privilege_count": float(privilege_hits),
            "exfiltration_count": float(exfiltration_hits),
            "imperative_ratio": float(imperative_ratio),
            "entropy": self._calculate_entropy(text),
            "is_file_read": is_read,
            "is_file_write": is_write,
            "is_command_exec": is_exec,
            "is_network_egress": is_egress,
            "is_network_ingress": is_ingress,
            "is_database_query": is_db,
            "is_computation": is_comp,
            "dag_node_count": node_count,
            "dag_edge_count": edge_count,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "is_root_node": is_root,
            "is_leaf_node": is_leaf,
            "is_critical_flag": 1.0 if task.is_critical else 0.0,
        }

        return features

    def extract_route_dataframe(self, route: Route) -> pd.DataFrame:
        """Extract a pandas DataFrame where each row corresponds to a task in the route."""
        graph = route.to_networkx()
        rows = []
        for task in route.tasks:
            feat = self.extract_task_features(task, dag_graph=graph)
            feat["task_id"] = task.id
            feat["route_id"] = route.route_id
            rows.append(feat)

        df = pd.DataFrame(rows)
        # Ensure standard feature ordering
        feature_cols = [c for c in self.feature_names if c in df.columns]
        meta_cols = [c for c in df.columns if c not in self.feature_names]
        return df[meta_cols + feature_cols]

    def extract_dag_dataframe(self, plan: PlanDAG) -> pd.DataFrame:
        """Extract tabular features across all routes in a PlanDAG."""
        dfs = [self.extract_route_dataframe(r) for r in plan.routes]
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        combined["plan_id"] = plan.plan_id
        return combined


def json_dumps_safe(val: Any) -> str:
    """Safe serializer helper for metadata/inputs dicts."""
    try:
        import json
        return json.dumps(val)
    except Exception:
        return str(val)

