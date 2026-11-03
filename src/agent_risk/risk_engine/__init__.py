"""Risk Engine module: feature extraction, XGBoost classifier shield, and necessity scoring."""

from agent_risk.risk_engine.feature_extractor import FeatureExtractor
from agent_risk.risk_engine.xgboost_shield import RiskVerdict, XGBoostShield
from agent_risk.risk_engine.necessity_filter import NecessityFilter

__all__ = [
    "FeatureExtractor",
    "XGBoostShield",
    "RiskVerdict",
    "NecessityFilter",
]

