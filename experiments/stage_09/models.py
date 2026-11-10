"""
Stage 09: Dedicated Ordinal Severity Models Suite (Levels 1 to 5).

Addresses the Exact Accuracy limitation of Stage 06 by implementing:
  1. StandardizedClassifierWrapper:
     Wraps 0-indexed multi-class classifiers for 1-indexed severity levels {1..5}.
  2. FrankHallOrdinalClassifier:
     Frank & Hall (2001) Cumulative Link Ordinal Decomposition:
     Decomposes 5-level target into 4 monotonic binary threshold classifiers:
       P(Y > 1), P(Y > 2), P(Y > 3), P(Y > 4).
     Enforces strict monotonic probability constraints and computes exact
     class probabilities:
       P(Y = 1) = 1 - P(Y > 1)
       P(Y = k) = P(Y > k-1) - P(Y > k) for k in {2, 3, 4}
       P(Y = 5) = P(Y > 4)
  3. HierarchicalGatedClassifier:
     3-Tier Coarse-to-Fine Gated Decision Architecture:
       Tier 1: Safe (Level 1) vs Harmful (Levels 2..5)
       Tier 2: Low/Moderate (Levels 2..3) vs High/Critical (Levels 4..5)
       Tier 3: Fine distinction between adjacent pairs (2 vs 3, 4 vs 5)
  4. OrdinalThresholdRegressor:
     Continuous regression with Nelder-Mead boundary threshold optimization.
  5. ConsolidatedSeverityEnsemble:
     Calibrated consensus stacking of the top tree and ordinal models.
"""

from typing import Dict, Any, Tuple, Optional, Union, List, Callable
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier
)
import xgboost as xgb


class StandardizedClassifierWrapper(BaseEstimator):
    """Wraps a 0-indexed or 1-indexed classifier to consistently handle severities {1..5}."""

    def __init__(self, base_classifier: Any):
        self.base_classifier = base_classifier

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Normalize target to 0..4 for classifiers expecting 0-indexed classes
        if isinstance(self.base_classifier, xgb.XGBClassifier):
            self.base_classifier.fit(X, (y - 1).astype(int))
        else:
            self.base_classifier.fit(X, y.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.base_classifier.predict(X)
        if isinstance(self.base_classifier, xgb.XGBClassifier):
            return (preds + 1).astype(int)
        return preds.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.base_classifier.predict_proba(X)
        if probs.shape[1] == 5:
            return probs
        # If any classes are missing in training batch, pad to 5
        full_probs = np.zeros((len(X), 5))
        classes = self.base_classifier.classes_
        for idx, c in enumerate(classes):
            c_idx = c if not isinstance(self.base_classifier, xgb.XGBClassifier) else c
            if 0 <= c_idx < 5:
                full_probs[:, c_idx] = probs[:, idx]
            elif 1 <= c_idx <= 5:
                full_probs[:, c_idx - 1] = probs[:, idx]
        return full_probs


class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Frank & Hall (2001) Cumulative Link Ordinal Decomposition.

    Decomposes the 5-class ordinal target into K-1 = 4 binary classification problems:
      C_1: P(Y > 1 | X)
      C_2: P(Y > 2 | X)
      C_3: P(Y > 3 | X)
      C_4: P(Y > 4 | X)

    Guarantees monotonic probability ordering:
      1 >= P(Y > 1) >= P(Y > 2) >= P(Y > 3) >= P(Y > 4) >= 0

    Calculates exact discrete probabilities:
      P(Y = 1) = 1 - P(Y > 1)
      P(Y = 2) = P(Y > 1) - P(Y > 2)
      P(Y = 3) = P(Y > 2) - P(Y > 3)
      P(Y = 4) = P(Y > 3) - P(Y > 4)
      P(Y = 5) = P(Y > 4)
    """

    def __init__(self, estimator_builder: Optional[Callable[[int], Any]] = None):
        self.estimator_builder = estimator_builder or (
            lambda k: HistGradientBoostingClassifier(random_state=42, min_samples_leaf=15, max_iter=100)
        )
        self.classifiers: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classifiers = []
        for k in range(1, 5):  # Cutoffs: 1, 2, 3, 4
            y_binary = (y > k).astype(int)
            clf = self.estimator_builder(k)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
        return self

    def predict_cumulative_probs(self, X: np.ndarray) -> np.ndarray:
        """Computes monotonic P(Y > k) for k in {1, 2, 3, 4}."""
        cum_probs = []
        for clf in self.classifiers:
            p_gt = clf.predict_proba(X)[:, 1]  # Prob that Y > k
            cum_probs.append(p_gt)
        cum_probs = np.column_stack(cum_probs)  # (N, 4)

        # Monotonicity projection: P(Y > k) <= P(Y > k-1)
        for i in range(1, 4):
            cum_probs[:, i] = np.minimum(cum_probs[:, i], cum_probs[:, i - 1])
        cum_probs = np.clip(cum_probs, 0.0, 1.0)
        return cum_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cum_probs = self.predict_cumulative_probs(X)
        N = len(X)
        probs = np.zeros((N, 5), dtype=np.float64)

        probs[:, 0] = 1.0 - cum_probs[:, 0]
        probs[:, 1] = cum_probs[:, 0] - cum_probs[:, 1]
        probs[:, 2] = cum_probs[:, 1] - cum_probs[:, 2]
        probs[:, 3] = cum_probs[:, 2] - cum_probs[:, 3]
        probs[:, 4] = cum_probs[:, 3]

        probs = np.maximum(probs, 0.0)
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) + 1


class HierarchicalGatedClassifier(BaseEstimator, ClassifierMixin):
    """
    3-Tier Coarse-to-Fine Gated Decision Network:
      - Tier 1 Gate: Safe (1) vs Harm (2..5)
      - Tier 2 Gate: Low/Moderate Harm (2..3) vs High/Critical Harm (4..5)
      - Tier 3a Gate: Level 2 vs Level 3
      - Tier 3b Gate: Level 4 vs Level 5
    """

    def __init__(self):
        self.gate1 = HistGradientBoostingClassifier(random_state=42, min_samples_leaf=15)
        self.gate2 = HistGradientBoostingClassifier(random_state=42, min_samples_leaf=12)
        self.gate3a = HistGradientBoostingClassifier(random_state=42, min_samples_leaf=5)
        self.gate3b = HistGradientBoostingClassifier(random_state=42, min_samples_leaf=2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Tier 1: Safe (1) vs Harmful (2..5)
        y_g1 = (y > 1).astype(int)
        self.gate1.fit(X, y_g1)

        # Tier 2: Harmful instances (y > 1) -> Low/Mid (2,3) vs High/Critical (4,5)
        harm_mask = y > 1
        y_g2 = (y[harm_mask] >= 4).astype(int)
        self.gate2.fit(X[harm_mask], y_g2)

        # Tier 3a: Low/Mid harm (2 vs 3)
        mask_23 = (y == 2) | (y == 3)
        if np.sum(mask_23) > 0:
            y_g3a = (y[mask_23] == 3).astype(int)
            self.gate3a.fit(X[mask_23], y_g3a)

        # Tier 3b: High/Critical harm (4 vs 5)
        mask_45 = (y == 4) | (y == 5)
        if np.sum(mask_45) > 0:
            y_g3b = (y[mask_45] == 5).astype(int)
            self.gate3b.fit(X[mask_45], y_g3b)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_harm = self.gate1.predict_proba(X)[:, 1]  # P(Y > 1)
        p_safe = 1.0 - p_harm

        p_high = self.gate2.predict_proba(X)[:, 1]  # P(Y in {4,5} | Y > 1)
        p_low = 1.0 - p_high                       # P(Y in {2,3} | Y > 1)

        p_3_given_low = self.gate3a.predict_proba(X)[:, 1]  # P(Y = 3 | Y in {2,3})
        p_2_given_low = 1.0 - p_3_given_low

        p_5_given_high = self.gate3b.predict_proba(X)[:, 1]  # P(Y = 5 | Y in {4,5})
        p_4_given_high = 1.0 - p_5_given_high

        probs = np.zeros((len(X), 5), dtype=np.float64)
        probs[:, 0] = p_safe
        probs[:, 1] = p_harm * p_low * p_2_given_low
        probs[:, 2] = p_harm * p_low * p_3_given_low
        probs[:, 3] = p_harm * p_high * p_4_given_high
        probs[:, 4] = p_harm * p_high * p_5_given_high

        probs = np.maximum(probs, 0.0)
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) + 1


class ThresholdOptimizer:
    """Optimizes ordinal cutoffs to map continuous predictions to discrete levels {1..5}."""

    def __init__(self, initial_cutoffs: Optional[List[float]] = None):
        self.cutoffs = initial_cutoffs or [1.5, 2.5, 3.5, 4.5]

    def fit(self, y_cont: np.ndarray, y_true: np.ndarray):
        from sklearn.metrics import accuracy_score

        def loss(th):
            th_sorted = sorted(th)
            preds = np.digitize(y_cont, th_sorted) + 1
            preds = np.clip(preds, 1, 5)
            # Maximize exact accuracy
            acc = accuracy_score(y_true, preds)
            return -acc

        res = minimize(loss, self.cutoffs, method="Nelder-Mead", options={"maxiter": 300})
        self.cutoffs = sorted(res.x.tolist())
        return self

    def predict(self, y_cont: np.ndarray) -> np.ndarray:
        preds = np.digitize(y_cont, self.cutoffs) + 1
        return np.clip(preds, 1, 5).astype(int)


class OrdinalThresholdRegressor(BaseEstimator):
    """Continuous regressor with Nelder-Mead threshold cutoffs."""

    def __init__(self, base_regressor: Any = None):
        self.base_regressor = base_regressor or HistGradientBoostingRegressor(random_state=42)
        self.optimizer = ThresholdOptimizer()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_regressor.fit(X, y.astype(float))
        y_cont = self.base_regressor.predict(X)
        self.optimizer.fit(y_cont, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_cont = self.base_regressor.predict(X)
        return self.optimizer.predict(y_cont)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_cont = self.base_regressor.predict(X)
        classes = np.array([1, 2, 3, 4, 5])
        dists = (y_cont[:, None] - classes[None, :]) ** 2
        exp_dists = np.exp(-dists / (2 * (0.65 ** 2)))
        probs = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
        return probs


class ConsolidatedSeverityEnsemble(BaseEstimator, ClassifierMixin):
    """
    Consolidated Multi-Architecture Severity Stacker:
      Unifies:
        - Primary (65%): HistGradientBoosting Multi-Class Classifier
        - Secondary (20%): Frank & Hall Ordinal Cumulative Link Classifier
        - Tertiary (15%): Multi-Class XGBoost Classifier
    """

    def __init__(
        self,
        clf_hgb: Optional[HistGradientBoostingClassifier] = None,
        clf_fh: Optional[FrankHallOrdinalClassifier] = None,
        clf_xgb: Optional[xgb.XGBClassifier] = None,
        weights: Optional[List[float]] = None
    ):
        self.clf_hgb = clf_hgb or HistGradientBoostingClassifier(random_state=42)
        self.clf_fh = clf_fh or FrankHallOrdinalClassifier()
        self.clf_xgb = clf_xgb or xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        self.weights = weights or [0.70, 0.20, 0.10]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf_hgb.fit(X, y)
        self.clf_fh.fit(X, y)
        self.clf_xgb.fit(X, (y - 1).astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_hgb = self.clf_hgb.predict_proba(X)
        p_fh = self.clf_fh.predict_proba(X)
        p_xgb = self.clf_xgb.predict_proba(X)

        w1, w2, w3 = self.weights
        total_w = w1 + w2 + w3
        probs = (w1 * p_hgb + w2 * p_fh + w3 * p_xgb) / total_w
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) + 1

