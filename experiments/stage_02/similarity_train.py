"""
Stage 02: Similarity-Augmented Model Training & Threshold Optimization.

Trains non-linear regressors (SVR with RBF kernel / MLP Regressor / Ridge) on the
hybrid 389-dimensional feature space (384 MiniLM embeddings + 5 Domain Similarities).
Employs balanced sample weighting to protect minority severity classes (Levels 2 and 5)
and optimizes decision cutoffs via Nelder-Mead to maximize Quadratic Weighted Kappa.
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
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
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
import joblib

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
    """
    Nelder-Mead optimization to find monotonic cutoffs [t1, t2, t3, t4] that
    map continuous predictions to ordinal levels {1, 2, 3, 4, 5} maximizing QWK.
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


def train_similarity_augmented_model(
    model_type: str = "svr",
    c_param: float = 2.0,
    epsilon: float = 0.1,
    alpha: float = 1.0,
    balance_weights: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    print("=" * 85)
    print("Stage 02: Training Similarity-Augmented Model (389 Hybrid Dimensions)")
    print("=" * 85)

    # 1. Load Data
    train_path, dev_path = locate_augmented_data()
    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)

    # Extract ONLY the domain label features (NO emb_ columns!)
    feature_cols = [c for c in df_train.columns if c.startswith("sim_") or c.startswith("attn_")]
    sim_cols = [c for c in df_train.columns if c.startswith("sim_")]
    attn_cols = [c for c in df_train.columns if c.startswith("attn_")]

    print(f"Features ({len(feature_cols)} total domain label values, 0 embeddings):")
    print(f"  • Cosine Similarities: {sim_cols}")
    if attn_cols:
        print(f"  • Attention Weights  : {attn_cols}")

    X_train = df_train[feature_cols].values
    y_train = df_train["risk_severity"].values.astype(float)
    y_train_int = df_train["risk_severity"].values.astype(int)

    X_dev = df_dev[feature_cols].values
    y_dev = df_dev["risk_severity"].values.astype(int)

    # 2. Balanced Sample Weights for Minority Classes
    sample_weights = None
    if balance_weights:
        sample_weights = compute_sample_weight("balanced", y_train_int)
        print(f"✓ Applied balanced sample weights to address class imbalance (Level 5 n=7, Level 2 n=33)")

    # 3. Model Architecture Instantiation & Fitting
    print(f"\nTraining Model: {model_type.upper()} on 389-d hybrid features...")
    if model_type.lower() == "svr":
        model = SVR(kernel="rbf", C=c_param, epsilon=epsilon, gamma="scale")
        model.fit(X_train, y_train, sample_weight=sample_weights)
    elif model_type.lower() == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            max_iter=400,
            learning_rate_init=0.005,
            random_state=seed
        )
        model.fit(X_train, y_train)
    elif model_type.lower() == "ridge":
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print("✓ Model fitted successfully.")

    # 4. Continuous Predictions & Metrics
    preds_cont = model.predict(X_dev)
    mse = mean_squared_error(y_dev, preds_cont)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_dev, preds_cont))
    corr = float(np.corrcoef(y_dev, preds_cont)[0, 1])

    # 5. Baseline Rounding vs Nelder-Mead Threshold Optimization
    preds_default = np.clip(np.round(preds_cont), 1, 5).astype(int)
    qwk_default = float(cohen_kappa_score(y_dev, preds_default, weights="quadratic"))

    optimizer = ThresholdOptimizer(init_method="percentile")
    optimal_t = optimizer.fit(preds_cont, y_dev)
    preds_opt = optimizer.transform(preds_cont)

    qwk_opt = float(cohen_kappa_score(y_dev, preds_opt, weights="quadratic"))
    acc_opt = float(accuracy_score(y_dev, preds_opt))
    macro_f1_opt = float(f1_score(y_dev, preds_opt, average="macro", zero_division=0))
    weighted_f1_opt = float(f1_score(y_dev, preds_opt, average="weighted", zero_division=0))
    cm_opt = confusion_matrix(y_dev, preds_opt, labels=[1, 2, 3, 4, 5]).tolist()

    # 6. Display Performance
    print("\n" + "=" * 85)
    print(f"EVALUATION RESULTS: {model_type.upper()} (389-d Hybrid Space)")
    print("=" * 85)
    print(f"Continuous Metrics:")
    print(f"  • RMSE                : {rmse:.4f}")
    print(f"  • MAE                 : {mae:.4f}")
    print(f"  • Pearson Correlation : {corr:.4f}")
    print(f"\nOrdinal Evaluation Metrics (Held-out Dev Set, 143 samples):")
    print(f"  • Default Rounding QWK       : {qwk_default:.4f}")
    print(f"  • Nelder-Mead Optimized QWK  : {qwk_opt:.4f}")
    print(f"  • Accuracy                   : {acc_opt * 100:.2f}%")
    print(f"  • Macro F1-Score             : {macro_f1_opt:.4f}")
    print(f"  • Weighted F1-Score          : {weighted_f1_opt:.4f}")
    print(f"  • Optimized Cutoffs          : {np.round(optimal_t, 3).tolist()}")

    print("\nConfusion Matrix:")
    print(format_confusion_matrix(cm_opt, labels=[1, 2, 3, 4, 5]))

    print("\nClassification Report:")
    print(classification_report(
        y_dev,
        preds_opt,
        labels=[1, 2, 3, 4, 5],
        target_names=[f"Severity {i}" for i in range(1, 6)],
        digits=4,
        zero_division=0
    ))

    # 7. Save Artifacts
    output_dir = STAGE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"similarity_{model_type.lower()}_model.joblib"
    joblib.dump(model, model_path)

    thresholds_path = output_dir / f"similarity_{model_type.lower()}_thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model_type,
            "thresholds": optimal_t.tolist(),
            "qwk_optimized": qwk_opt,
            "qwk_default": qwk_default
        }, f, indent=2)

    print(f"✓ Saved model to: {model_path}")
    print(f"✓ Saved thresholds to: {thresholds_path}")

    return {
        "model": f"Domain-Augmented {model_type.upper()}",
        "rmse": rmse,
        "mae": mae,
        "pearson_r": corr,
        "qwk_default": qwk_default,
        "qwk_optimized": qwk_opt,
        "accuracy": acc_opt,
        "macro_f1": macro_f1_opt,
        "weighted_f1": weighted_f1_opt,
        "thresholds": optimal_t.tolist(),
        "confusion_matrix": cm_opt,
        "predictions": preds_opt.tolist(),
        "predictions_continuous": preds_cont.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Train Similarity-Augmented ML Models")
    parser.add_argument("--model", type=str, default="svr", choices=["svr", "mlp", "ridge"], help="Model architecture")
    parser.add_argument("--c", type=float, default=2.0, help="SVR C parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="SVR epsilon")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    args = parser.parse_args()

    train_similarity_augmented_model(
        model_type=args.model,
        c_param=args.c,
        epsilon=args.epsilon,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()

