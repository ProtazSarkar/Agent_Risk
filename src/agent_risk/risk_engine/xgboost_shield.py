"""XGBoost Shield: High-throughput tabular classifier for risk scoring and mitigation."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from agent_risk.config import settings
from agent_risk.risk_engine.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class RiskVerdict(str, Enum):
    """Categorical risk decisions."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


class XGBoostShield:
    """Unified XGBoost classifier acting as an active security shield for agent actions."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        warning_threshold: Optional[float] = None,
        block_threshold: Optional[float] = None,
    ):
        self.model_path = Path(model_path or settings.xgboost_model_path)
        self.warning_threshold = warning_threshold or settings.risk_threshold_warning
        self.block_threshold = block_threshold or settings.risk_threshold_block
        self.feature_extractor = FeatureExtractor()
        self.model = None

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        params: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, float]:
        """Train or fit XGBoost classifier on tabular risk features."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        try:
            import xgboost as xgb
            pos_weight = float((len(y) - sum(y)) / max(1, sum(y))) if sum(y) > 0 else 1.0
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": pos_weight,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": settings.random_seed,
            }
            if params:
                default_params.update(params)
            self.model = xgb.XGBClassifier(**default_params)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            logger.warning("XGBoost library not found. Training fallback HistGradientBoostingClassifier.")
            min_leaf = max(1, min(20, len(X) // 8))
            self.model = HistGradientBoostingClassifier(
                class_weight="balanced",
                min_samples_leaf=min_leaf,
                random_state=settings.random_seed,
                max_iter=50,
            )

        self.model.fit(X, y)

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        metrics = {
            "train_accuracy": float(accuracy_score(y, preds)),
            "train_f1": float(f1_score(y, preds, zero_division=0)),
        }
        try:
            metrics["train_roc_auc"] = float(roc_auc_score(y, probs))
        except Exception:
            pass

        if save:
            self.save_model()

        return metrics

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict continuous probability of high risk for given feature vectors."""
        if self.model is None:
            self.load_model()

        # If X is a DataFrame, ensure column alignment
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_extractor.feature_names if c in X.columns]
            X = X[cols]

        return self.model.predict_proba(X)[:, 1]

    def evaluate_verdict(self, risk_probability: float) -> Tuple[RiskVerdict, float]:
        """Map risk probability to categorical action shield verdict."""
        if risk_probability >= self.block_threshold:
            return RiskVerdict.BLOCKED, risk_probability
        elif risk_probability >= self.warning_threshold:
            return RiskVerdict.WARNING, risk_probability
        return RiskVerdict.SAFE, risk_probability

    def save_model(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Serialize model to disk."""
        target_path = Path(path or self.model_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            if hasattr(self.model, "save_model"):
                self.model.save_model(str(target_path))
            else:
                import pickle
                with open(target_path, "wb") as f:
                    pickle.dump(self.model, f)
            logger.info(f"Saved shield model to {target_path}")
        return target_path

    def load_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load serialized model from disk or initialize heuristic baseline."""
        target_path = Path(path or self.model_path)
        if target_path.exists():
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(target_path))
                logger.info(f"Loaded XGBoost model from {target_path}")
                return
            except Exception:
                try:
                    import pickle
                    with open(target_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded serialized fallback model from {target_path}")
                    return
                except Exception as e:
                    logger.warning(f"Could not load model file {target_path}: {e}")

        logger.info("Initializing heuristic baseline classifier.")
        self._initialize_heuristic_baseline()

    def _initialize_heuristic_baseline(self) -> None:
        """Train a lightweight baseline model on synthetic boundary cases for bootstrap."""
        X_syn = np.array([
            # benign reads & computations
            [10, 2, 0, 0, 0, 0, 0.1, 1.2, 1, 0, 0, 0, 0, 0, 0, 3, 2, 0, 1, 1, 0, 1],
            [20, 4, 0, 0, 0, 0, 0.2, 2.0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 1, 1, 0, 0, 0],
            # malicious privilege escalations
            [80, 15, 3, 2, 3, 2, 0.8, 4.5, 0, 1, 1, 1, 0, 0, 0, 4, 3, 1, 1, 0, 1, 1],
            [90, 18, 4, 1, 4, 3, 0.9, 4.8, 0, 0, 1, 1, 0, 0, 0, 2, 1, 0, 1, 1, 0, 1],
        ], dtype=float)
        y_syn = np.array([0, 0, 1, 1])

        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
            self.model.fit(X_syn, y_syn)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(random_state=42, max_iter=20)
            self.model.fit(X_syn, y_syn)
