"""Agentic Risk Analysis Framework.

A production-grade, modular toolkit for decomposing agent goals into multi-route task DAGs,
extracting psycholinguistic and topological features, applying ML risk shielding (XGBoost),
and applying game-theoretic decision policies.
"""

from agent_risk.config import settings
from agent_risk.data.loader import AgentRiskDatasetLoader, AgentRiskInstance

__version__ = "0.1.0"
__all__ = ["settings", "AgentRiskDatasetLoader", "AgentRiskInstance", "__version__"]

