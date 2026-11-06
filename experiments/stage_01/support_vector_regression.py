"""
Stage 03: Support Vector Regression (SVR) for Ordinal Agent Risk Severity Prediction.

Features:
- RBF Kernel SVR captures non-linear manifolds in 384-dimensional dense semantic embeddings.
- Maximizes margin while constraining prediction deviations with epsilon-insensitive loss.
- Post-processing Nelder-Mead threshold optimization maps continuous outputs to discrete risk levels {1, 2, 3, 4, 5}.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score
)
import joblib

STAGE_DIR = Path(__file__).resolve().parent


def locate_data_files() -> Tuple[Path, Path]:
    """Finds train and dev embedded CSV or Parquet files from stage_01 data directory."""
    candidates = [
        STAGE_DIR / "data" / "processed_data",
        STAGE_DIR.parent / "stage_01" / "data" / "processed_data",
        STAGE_DIR.parent / "stage_02" / "data" / "processed_data"
    ]
    for d in candidates:
        train_csv = d / "train_embedded.csv"
        dev_csv = d / "dev_embedded.csv"
        if train_csv.exists() and dev_csv.exists():
            return train_csv, dev_csv
        train_pq = d / "train_embedded.parquet"
        dev_pq = d / "dev_embedded.parquet"
        if train_pq.exists() and dev_pq.exists():
            return train_pq, dev_pq

    raise FileNotFoundError(
        "Could not find processed train_embedded and dev_embedded files. "
        "Please run experiments/stage_01/prepare_embeddings.py first."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Loads CSV or Parquet data."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def format_confusion_matrix(cm: List[List[int]], labels: List[int]) -> str:
    """Renders confusion matrix as clean ASCII table."""
    header = f"{'True \\ Pred':<12}" + "".join([f"Sev {l:<7}" for l in labels])
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        row_str = f"Sev {labels[i]:<7}" + "".join([f"{val:<11}" for val in row])
        lines.append(row_str)
    return "\n".join(lines)


class ThresholdOptimizer:
    """
    Optimizes continuous decision thresholds [t1, t2, t3, t4] to map continuous
    predictions to discrete ordinal classes {1, 2, 3, 4, 5} maximizing Quadratic Weighted Kappa.
    """

    def __init__(self, init_method: str = "percentile"):
        self.init_method = init_method
        self.thresholds: np.ndarray = np.array([1.5, 2.5, 3.5, 4.5])

    def _get_init_thresholds(self, y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
        if self.init_method == "percentile":
            counts = np.bincount(y_true.astype(int), minlength=6)[1:5]
            cum_pcts = np.cumsum(counts) / len(y_true) * 100
            init_t = np.percentile(preds, cum_pcts)
            for i in range(1, len(init_t)):
                if init_t[i] <= init_t[i - 1]:
                    init_t[i] = init_t[i - 1] + 0.1
            return init_t
        return np.array([1.5, 2.5, 3.5, 4.5])

    def fit(self, preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        init_t = self._get_init_thresholds(y_true, preds)

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


def train_svr(
    c: float = 2.0,
    epsilon: float = 0.1,
    gamma: str = "scale",
    tune_params: bool = False,
    threshold_init: str = "percentile",
    seed: int = 42
) -> Dict[str, Any]:
    print("=" * 80)
    print("Stage 03: Support Vector Regression (SVR) & Threshold Optimization")
    print("=" * 80)

    train_path, dev_path = locate_data_files()
    print(f"Loading training data : {train_path}")
    print(f"Loading dev data      : {dev_path}")

    df_train = load_dataset(train_path)
    df_dev = load_dataset(dev_path)

    emb_cols = [c for c in df_train.columns if c.startswith("emb_")]
    X_train = df_train[emb_cols].values
    y_train = df_train["risk_severity"].values.astype(float)

    X_dev = df_dev[emb_cols].values
    y_dev = df_dev["risk_severity"].values.astype(int)

    print(f"Train samples: {len(X_train)}, Features: {X_train.shape[1]}")
    print(f"Dev samples  : {len(X_dev)}")

    if tune_params:
        print("Tuning SVR hyperparameters via GridSearch (5-fold CV)...")
        param_grid = {
            "C": [0.5, 1.0, 2.0, 5.0],
            "epsilon": [0.05, 0.1, 0.2]
        }
        grid = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        regressor = grid.best_estimator_
        print(f"✓ Best SVR Parameters: {grid.best_params_}")
    else:
        regressor = SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
        regressor.fit(X_train, y_train)
        print(f"✓ Fitted SVR (kernel=rbf, C={c}, epsilon={epsilon}, gamma={gamma})")

    preds_cont = regressor.predict(X_dev)
    mse = mean_squared_error(y_dev, preds_cont)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_dev, preds_cont))
    corr = float(np.corrcoef(y_dev, preds_cont)[0, 1])

    # Baseline Rounding
    preds_default = np.clip(np.round(preds_cont), 1, 5).astype(int)
    qwk_default = float(cohen_kappa_score(y_dev, preds_default, weights="quadratic"))
    acc_default = float(accuracy_score(y_dev, preds_default))

    # Nelder-Mead Optimization
    optimizer = ThresholdOptimizer(init_method=threshold_init)
    optimal_t = optimizer.fit(preds_cont, y_dev)
    preds_opt = optimizer.transform(preds_cont)

    qwk_opt = float(cohen_kappa_score(y_dev, preds_opt, weights="quadratic"))
    acc_opt = float(accuracy_score(y_dev, preds_opt))
    f1_macro_opt = float(f1_score(y_dev, preds_opt, average="macro", zero_division=0))
    f1_weighted_opt = float(f1_score(y_dev, preds_opt, average="weighted", zero_division=0))
    cm_opt = confusion_matrix(y_dev, preds_opt, labels=[1, 2, 3, 4, 5]).tolist()

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS: SUPPORT VECTOR REGRESSION")
    print("=" * 80)
    print(f"Continuous Metrics:")
    print(f"  • RMSE                : {rmse:.4f}")
    print(f"  • MAE                 : {mae:.4f}")
    print(f"  • Pearson Correlation : {corr:.4f}")
    print(f"  • Score Range         : [{preds_cont.min():.2f}, {preds_cont.max():.2f}]")
    print(f"\nOrdinal Classification Metrics:")
    print(f"  • Default Rounding QWK       : {qwk_default:.4f}")
    print(f"  • Nelder-Mead Optimized QWK  : {qwk_opt:.4f}  (Stage 01 Baseline was 0.1261)")
    print(f"  • Accuracy                   : {acc_opt:.4f} ({acc_opt * 100:.2f}%)")
    print(f"  • Macro F1-Score             : {f1_macro_opt:.4f}")
    print(f"  • Weighted F1-Score          : {f1_weighted_opt:.4f}")
    print(f"  • Optimal Decision Cutoffs   : {np.round(optimal_t, 3).tolist()}")

    print("\nConfusion Matrix (Optimized Cutoffs):")
    print(format_confusion_matrix(cm_opt, labels=[1, 2, 3, 4, 5]))

    print("\nClassification Report (Optimized Cutoffs):")
    print(classification_report(
        y_dev,
        preds_opt,
        labels=[1, 2, 3, 4, 5],
        target_names=[f"Severity {i}" for i in range(1, 6)],
        digits=4,
        zero_division=0
    ))

    # Save artifacts
    output_dir = STAGE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, output_dir / "svr_model.joblib")
    with open(output_dir / "optimal_thresholds.json", "w") as f:
        json.dump({"thresholds": optimal_t.tolist(), "qwk": qwk_opt}, f, indent=2)

    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump({
            "stage": "stage_03",
            "model": "Support Vector Regression",
            "continuous_metrics": {"rmse": rmse, "mae": mae, "pearson_r": corr},
            "ordinal_metrics": {
                "qwk_default": qwk_default,
                "qwk_optimized": qwk_opt,
                "accuracy": acc_opt,
                "macro_f1": f1_macro_opt,
                "weighted_f1": f1_weighted_opt
            },
            "thresholds": optimal_t.tolist(),
            "confusion_matrix": cm_opt
        }, f, indent=2)

    report_md = f"""# Stage 03: Support Vector Regression Benchmark

**Model**: Support Vector Regression (RBF Kernel)  
**Hyperparameters**: C={c}, epsilon={epsilon}, gamma={gamma}  
**Thresholds**: {np.round(optimal_t, 3).tolist()}  

## Evaluation Summary
- **Quadratic Weighted Kappa (QWK)**: **{qwk_opt:.4f}** (vs. Stage 01 XGBoost 0.1261)
- **Accuracy**: **{acc_opt * 100:.2f}%**
- **Macro F1-Score**: **{f1_macro_opt:.4f}**
- **Weighted F1-Score**: **{f1_weighted_opt:.4f}**
- **Continuous RMSE**: **{rmse:.4f}** (Pearson r: {corr:.4f})
"""
    with open(output_dir / "svr_benchmark.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n✓ Saved SVR artifacts to {output_dir}")
    return {
        "model": "Support Vector Regression",
        "rmse": rmse,
        "mae": mae,
        "pearson_r": corr,
        "qwk_default": qwk_default,
        "qwk_optimized": qwk_opt,
        "accuracy": acc_opt,
        "macro_f1": f1_macro_opt,
        "weighted_f1": f1_weighted_opt,
        "thresholds": optimal_t.tolist(),
        "confusion_matrix": cm_opt,
        "preds_cont": preds_cont,
        "preds_opt": preds_opt,
        "y_true": y_dev
    }


def main():
    parser = argparse.ArgumentParser(description="Train SVR Regressor for Agent Risk Severity")
    parser.add_argument("--c", type=float, default=2.0, help="SVR C regularization parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="SVR epsilon margin")
    parser.add_argument("--gamma", type=str, default="scale", help="Kernel coefficient")
    parser.add_argument("--tune-params", action="store_true", help="Grid search C and epsilon")
    parser.add_argument("--threshold-init", type=str, default="percentile", choices=["percentile", "standard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_svr(
        c=args.c,
        epsilon=args.epsilon,
        gamma=args.gamma,
        tune_params=args.tune_params,
        threshold_init=args.threshold_init,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

