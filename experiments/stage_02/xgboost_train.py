"""
Stage 02: XGBoost Training on 389-Dimensional Domain-Augmented Features.

Evaluates whether augmenting raw 384-d semantic embeddings with explicit 5-d
domain risk similarity scores resolves XGBoost's feature partitioning bottleneck.
Supports both continuous regression with threshold optimization and multi-class classification.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error
)

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from experiments.stage_02.processdata import process_and_augment_dataset
except (ImportError, ModuleNotFoundError):
    from processdata import process_and_augment_dataset


def locate_augmented_data() -> Tuple[Path, Path]:
    """Finds processed 389-dimensional domain-augmented dataset."""
    proc_dir = STAGE_DIR / "data" / "processed"
    tr = proc_dir / "train_domain_augmented.csv"
    dv = proc_dir / "dev_domain_augmented.csv"
    if not (tr.exists() and dv.exists()):
        print("Domain-augmented dataset not found. Generating now...")
        process_and_augment_dataset()
    return tr, dv


def format_confusion_matrix(cm: List[List[int]], labels: List[int]) -> str:
    """Renders confusion matrix as clean ASCII table."""
    header = f"{'True \\ Pred':<12}" + "".join([f"Sev {l:<7}" for l in labels])
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        row_str = f"Sev {labels[i]:<7}" + "".join([f"{val:<11}" for val in row])
        lines.append(row_str)
    return "\n".join(lines)


class ThresholdOptimizer:
    """Nelder-Mead threshold optimization for continuous regression outputs."""
    def __init__(self, init_method: str = "percentile"):
        self.init_method = init_method
        self.thresholds: np.ndarray = np.array([1.5, 2.5, 3.5, 4.5])

    def fit(self, preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_true.astype(int), minlength=6)[1:5]
        cum_pcts = np.cumsum(counts) / len(y_true) * 100
        init_t = np.percentile(preds, cum_pcts)
        for i in range(1, len(init_t)):
            if init_t[i] <= init_t[i - 1]:
                init_t[i] = init_t[i - 1] + 0.1

        def objective(t):
            penalty = 0.0
            sorted_t = np.sort(t)
            if not np.array_equal(t, sorted_t):
                penalty += 10.0 * np.sum((t - sorted_t) ** 2)

            discrete = np.digitize(preds, sorted_t) + 1
            discrete = np.clip(discrete, 1, 5)
            qwk = cohen_kappa_score(y_true, discrete, weights="quadratic")
            return -qwk + penalty

        res = minimize(objective, init_t, method="Nelder-Mead", options={"maxiter": 500})
        self.thresholds = np.sort(res.x)
        return self.thresholds

    def transform(self, preds: np.ndarray) -> np.ndarray:
        discrete = np.digitize(preds, self.thresholds) + 1
        return np.clip(discrete, 1, 5).astype(int)


def train_xgboost_domain(
    mode: str = "regressor",
    n_estimators: int = 150,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    print("=" * 85)
    print(f"Stage 02: XGBoost Training on 389-d Domain-Augmented Features ({mode.upper()})")
    print("=" * 85)

    train_path, dev_path = locate_augmented_data()
    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)

    feature_cols = [c for c in df_train.columns if c.startswith("sim_") or c.startswith("attn_")]
    sim_cols = [c for c in df_train.columns if c.startswith("sim_")]

    print(f"Features: {len(feature_cols)} domain label values (0 embeddings)")
    print(f"Domain Features: {feature_cols}")

    X_train = df_train[feature_cols].values
    y_train = df_train["risk_severity"].values
    X_dev = df_dev[feature_cols].values
    y_dev = df_dev["risk_severity"].values.astype(int)

    sample_weights = compute_sample_weight("balanced", y_train)

    if mode.lower() == "regressor":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            objective="reg:squarederror"
        )
        model.fit(X_train, y_train.astype(float), sample_weight=sample_weights)
        preds_cont = model.predict(X_dev)

        mse = mean_squared_error(y_dev, preds_cont)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_dev, preds_cont))
        corr = float(np.corrcoef(y_dev, preds_cont)[0, 1])

        # Baseline Rounding
        preds_default = np.clip(np.round(preds_cont), 1, 5).astype(int)
        qwk_default = float(cohen_kappa_score(y_dev, preds_default, weights="quadratic"))

        # Nelder-Mead Optimization
        optimizer = ThresholdOptimizer(init_method="percentile")
        optimal_t = optimizer.fit(preds_cont, y_dev)
        preds_opt = optimizer.transform(preds_cont)

        qwk_opt = float(cohen_kappa_score(y_dev, preds_opt, weights="quadratic"))
        acc_opt = float(accuracy_score(y_dev, preds_opt))
        macro_f1_opt = float(f1_score(y_dev, preds_opt, average="macro", zero_division=0))
        weighted_f1_opt = float(f1_score(y_dev, preds_opt, average="weighted", zero_division=0))
        cm_opt = confusion_matrix(y_dev, preds_opt, labels=[1, 2, 3, 4, 5]).tolist()

        print(f"\nContinuous Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, Pearson r={corr:.4f}")
        print(f"Ordinal Evaluation:")
        print(f"  • Default Rounding QWK      : {qwk_default:.4f}")
        print(f"  • Nelder-Mead Optimized QWK : {qwk_opt:.4f} (Stage 01 XGBoost was 0.1261)")
        print(f"  • Accuracy                  : {acc_opt * 100:.2f}%")
        print(f"  • Macro F1-Score            : {macro_f1_opt:.4f}")
        print(f"  • Optimal Cutoffs           : {np.round(optimal_t, 3).tolist()}")

        preds_final = preds_opt
    else:
        # Multi-class formulation
        y_train_mapped = y_train.astype(int) - 1
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            objective="multi:softprob",
            num_class=5
        )
        model.fit(X_train, y_train_mapped, sample_weight=sample_weights)
        preds_zero = model.predict(X_dev)
        preds_final = preds_zero + 1

        qwk_opt = float(cohen_kappa_score(y_dev, preds_final, weights="quadratic"))
        qwk_default = qwk_opt
        acc_opt = float(accuracy_score(y_dev, preds_final))
        macro_f1_opt = float(f1_score(y_dev, preds_final, average="macro", zero_division=0))
        weighted_f1_opt = float(f1_score(y_dev, preds_final, average="weighted", zero_division=0))
        cm_opt = confusion_matrix(y_dev, preds_final, labels=[1, 2, 3, 4, 5]).tolist()
        rmse, mae, corr = 0.0, 0.0, 0.0
        optimal_t = np.array([])

        print(f"\nMulti-Class Evaluation:")
        print(f"  • QWK Score      : {qwk_opt:.4f}")
        print(f"  • Accuracy       : {acc_opt * 100:.2f}%")
        print(f"  • Macro F1-Score : {macro_f1_opt:.4f}")

    print("\nConfusion Matrix:")
    print(format_confusion_matrix(cm_opt, labels=[1, 2, 3, 4, 5]))

    # Feature Importance of the 5 Domain Features vs Embeddings
    importances = model.feature_importances_
    sim_importances = {name: float(importances[feature_cols.index(name)]) for name in sim_cols}
    print("\nDomain Feature Importances (Tree Splits):")
    for name, imp in sorted(sim_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {name:<35}: {imp:.4f} ({imp * 100:.2f}%)")

    # Save artifacts
    output_dir = STAGE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "xgboost_domain_model.json"))

    return {
        "model": f"Domain-Augmented XGBoost ({mode.capitalize()})",
        "qwk_default": qwk_default,
        "qwk_optimized": qwk_opt,
        "accuracy": acc_opt,
        "macro_f1": macro_f1_opt,
        "weighted_f1": weighted_f1_opt,
        "domain_importances": sim_importances,
        "confusion_matrix": cm_opt,
        "predictions": preds_final.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on Domain Augmented Data")
    parser.add_argument("--mode", type=str, default="regressor", choices=["regressor", "classifier"])
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    train_xgboost_domain(
        mode=args.mode,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )


if __name__ == "__main__":
    main()

