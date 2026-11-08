"""
Stage 06: Tree-Based & Regression Severity Model Suite (Levels 1 to 5).

Implements competitive models for ordinal risk severity prediction (levels 1 to 5):
  1. XGBoostClassifier & XGBoostRegressor (Tree-based gradient boosting)
  2. HistGradientBoostingClassifier & HistGradientBoostingRegressor
  3. RandomForestClassifier & RandomForestRegressor
  4. Ordinal Ridge Regressor with Nelder-Mead threshold optimization
  5. Calibrated Multi-Class Softmax Classifier (for explicit confidence estimation)
  6. Hybrid Ensemble Model (best tree classifier + calibrated regressor)

All models are standardized to accept y in {1, 2, 3, 4, 5}, return predictions in {1, 2, 3, 4, 5},
and provide predict_proba(X) with shape (N, 5) representing probabilities for severities 1..5.
"""

from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge
import xgboost as xgb


class StandardizedClassifierWrapper(BaseEstimator):
    """Wraps a 0-indexed multi-class classifier to work with 1-indexed severity levels {1..5}."""

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_0idx = (y - 1).astype(int)
        self.base_classifier.fit(X, y_0idx)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds_0idx = self.base_classifier.predict(X)
        return (preds_0idx + 1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.base_classifier.predict_proba(X)
        if probs.shape[1] < 5:
            full_probs = np.zeros((len(X), 5))
            for i, c in enumerate(self.base_classifier.classes_):
                full_probs[:, c] = probs[:, i]
            return full_probs
        return probs


class ThresholdOptimizer:
    """Optimizes ordinal decision boundaries to map continuous regression outputs to discrete levels {1, 2, 3, 4, 5}."""

    def __init__(self, initial_cutoffs: Optional[List[float]] = None):
        self.cutoffs = initial_cutoffs or [1.5, 2.5, 3.5, 4.5]

    def fit(self, y_cont: np.ndarray, y_true: np.ndarray):
        def loss(thresholds):
            th = sorted(thresholds)
            preds = np.digitize(y_cont, th) + 1
            preds = np.clip(preds, 1, 5)
            from sklearn.metrics import cohen_kappa_score
            qwk = cohen_kappa_score(y_true, preds, weights="quadratic")
            return -qwk

        res = minimize(loss, self.cutoffs, method="Nelder-Mead", options={"maxiter": 300})
        self.cutoffs = sorted(res.x.tolist())
        return self

    def predict(self, y_cont: np.ndarray) -> np.ndarray:
        preds = np.digitize(y_cont, self.cutoffs) + 1
        return np.clip(preds, 1, 5).astype(int)


class OrdinalRegressorWrapper(BaseEstimator):
    """Wraps a continuous regressor with optimal ordinal threshold cutoffs and soft confidence."""

    def __init__(self, base_regressor):
        self.base_regressor = base_regressor
        self.optimizer = ThresholdOptimizer()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_regressor.fit(X, y.astype(float))
        y_cont = self.base_regressor.predict(X)
        self.optimizer.fit(y_cont, y)
        return self

    def predict_continuous(self, X: np.ndarray) -> np.ndarray:
        return self.base_regressor.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_cont = self.predict_continuous(X)
        return self.optimizer.predict(y_cont)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes pseudo-probabilities via Gaussian kernel around classes 1..5."""
        y_cont = self.predict_continuous(X)
        classes = np.array([1, 2, 3, 4, 5])
        dists = (y_cont[:, None] - classes[None, :]) ** 2
        exp_dists = np.exp(-dists / (2 * (0.65 ** 2)))
        probs = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
        return probs


class HybridSeverityEnsemble(BaseEstimator):
    """
    Ensemble combining:
      - Tree-based multi-class classifier (captures non-linear domain interactions)
      - Ordinal threshold-tuned regressor (enforces monotonic severity hierarchy)
    """

    def __init__(self, classifier: StandardizedClassifierWrapper, regressor: OrdinalRegressorWrapper, weight_cls: float = 0.55):
        self.classifier = classifier
        self.regressor = regressor
        self.weight_cls = weight_cls

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classifier.fit(X, y)
        self.regressor.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_cls = self.classifier.predict_proba(X)
        p_reg = self.regressor.predict_proba(X)
        return self.weight_cls * p_cls + (1.0 - self.weight_cls) * p_reg

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) + 1


def build_candidate_models() -> Dict[str, Any]:
    """Builds a rich dictionary of tree-based and regression models."""
    models = {}

    # 1. XGBoost Multi-Class Classifier
    xgb_cls = xgb.XGBClassifier(
        n_estimators=130,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=5,
        random_state=42,
        eval_metric="mlogloss"
    )
    models["XGBoost_Classifier"] = StandardizedClassifierWrapper(xgb_cls)

    # 2. HistGradientBoosting Classifier
    hgb_cls = HistGradientBoostingClassifier(
        max_iter=140,
        max_depth=5,
        learning_rate=0.07,
        l2_regularization=0.5,
        random_state=42
    )
    models["HistGradientBoosting_Classifier"] = StandardizedClassifierWrapper(hgb_cls)

    # 3. Random Forest Classifier
    rf_cls = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=42
    )
    models["RandomForest_Classifier"] = StandardizedClassifierWrapper(rf_cls)

    # 4. XGBoost Ordinal Regressor
    xgb_reg = xgb.XGBRegressor(
        n_estimators=110,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        random_state=42
    )
    models["XGBoost_Ordinal_Regressor"] = OrdinalRegressorWrapper(xgb_reg)

    # 5. HistGradientBoosting Ordinal Regressor
    hgb_reg = HistGradientBoostingRegressor(
        max_iter=130,
        max_depth=4,
        learning_rate=0.07,
        random_state=42
    )
    models["HistGradientBoosting_Regressor"] = OrdinalRegressorWrapper(hgb_reg)

    # 6. Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    models["RandomForest_Regressor"] = OrdinalRegressorWrapper(rf_reg)

    # 7. Ridge Ordinal Regressor
    ridge_reg = Ridge(alpha=10.0)
    models["Ridge_Ordinal_Regressor"] = OrdinalRegressorWrapper(ridge_reg)

    # 8. Hybrid Ensemble (XGBoost Classifier + XGBoost Regressor)
    models["Hybrid_XGB_Ensemble"] = HybridSeverityEnsemble(
        classifier=StandardizedClassifierWrapper(xgb.XGBClassifier(
            n_estimators=110, max_depth=4, learning_rate=0.07,
            objective="multi:softprob", num_class=5, random_state=42
        )),
        regressor=OrdinalRegressorWrapper(xgb.XGBRegressor(
            n_estimators=110, max_depth=4, learning_rate=0.07, random_state=42
        )),
        weight_cls=0.55
    )

    return models

