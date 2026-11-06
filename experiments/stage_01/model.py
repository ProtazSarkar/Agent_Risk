"""
XGBoost Risk Severity Classifier for Agentic Systems.

Classifies agent interaction trajectories and proposed actions into
ordinal risk severity levels (1 to 5) based on dense semantic embeddings.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)
from sklearn.utils.class_weight import compute_sample_weight


class AgentRiskXGBoostClassifier:
    """
    Unified XGBoost multi-class / ordinal classifier for agent risk severity estimation.

    Target Severity Levels:
      1: Benign / Safe (Read-only queries, passive inspection)
      2: Low Risk (Minor configuration, non-sensitive preferences)
      3: Moderate Risk (External resource calls, batch operations)
      4: High Risk (Modifying permissions, sending credentials, data export)
      5: Critical / Catastrophic Risk (Destructive execution, sensitive breach)
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        random_state: int = 42,
        num_classes: int = 5,
        **kwargs: Any
    ):
        self.num_classes = num_classes
        self.params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "random_state": random_state,
            **kwargs
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, list],
        sample_weight: Optional[np.ndarray] = None,
        balance_classes: bool = True
    ) -> "AgentRiskXGBoostClassifier":
        """
        Fits the XGBoost classifier on feature matrix X and targets y.

        Args:
            X: Embedding feature matrix (N, D), typically 384 dimensions.
            y: Target risk severity levels (1 to 5). Automatically mapped to 0-indexed classes.
            sample_weight: Optional sample weights.
            balance_classes: If True and sample_weight is None, computes balanced class weights.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)

        # Handle 1-based indexing [1..5] -> [0..4]
        if y_arr.min() >= 1 and y_arr.max() <= self.num_classes:
            y_mapped = y_arr - 1
        else:
            y_mapped = y_arr

        if sample_weight is None and balance_classes:
            sample_weight = compute_sample_weight("balanced", y_mapped)

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_arr, y_mapped, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns class probabilities (N, 5) for severity levels 1 through 5."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X_arr)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns predicted risk severity levels mapped back to [1, 2, 3, 4, 5]."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")
        X_arr = np.asarray(X, dtype=np.float32)
        preds_zero_indexed = self.model.predict(X_arr)
        return preds_zero_indexed + 1

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, list]
    ) -> Dict[str, Any]:
        """
        Evaluates predictions on X against ground truth y.

        Returns:
            Dictionary containing accuracy, macro_f1, weighted_f1, qwk, report, and confusion_matrix.
        """
        y_true = np.asarray(y, dtype=int)
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

        report_str = classification_report(
            y_true,
            y_pred,
            labels=list(range(1, self.num_classes + 1)),
            target_names=[f"Severity {i}" for i in range(1, self.num_classes + 1)],
            digits=4,
            zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(1, self.num_classes + 1))).tolist()

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "qwk": qwk,
            "classification_report": report_str,
            "confusion_matrix": cm,
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist()
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model weights in XGBoost JSON format."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        print(f" Saved XGBoost model artifact to: {path}")

    def load(self, filepath: Union[str, Path]) -> "AgentRiskXGBoostClassifier":
        """Loads model weights from an XGBoost JSON artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(str(path))
        self.is_fitted = True
        print(f" Loaded XGBoost model artifact from: {path}")
        return self

