"""
Stage 09: Cross-Validation & Champion Severity Model Trainer.

Performs 5-Fold Stratified Cross-Validation on the 568 SemEval training samples
across diverse ordinal and gradient boosted architectures:
  1. HistGradientBoosting_Classifier
  2. FrankHall_Ordinal_HistGBDT
  3. HierarchicalGated_Classifier
  4. XGBoost_Classifier
  5. RandomForest_Classifier
  6. OrdinalThreshold_Regressor
  7. ConsolidatedSeverity_Ensemble

Evaluates:
  - Exact Accuracy (%)
  - Within-1 Accuracy (%)
  - Quadratic Weighted Kappa (QWK)
  - Mean Absolute Error (MAE)
  - Macro F1-score

Exports:
  - experiments/stage_09/models/best_severity_model_stage09.joblib
  - experiments/stage_09/output/train_cv_benchmark.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    f1_score,
    confusion_matrix
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier
)
import xgboost as xgb

STAGE_09_DIR = Path(__file__).resolve().parent
if str(STAGE_09_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_09_DIR))

from models import (
    StandardizedClassifierWrapper,
    FrankHallOrdinalClassifier,
    HierarchicalGatedClassifier,
    OrdinalThresholdRegressor,
    ConsolidatedSeverityEnsemble
)

DATA_DIR = STAGE_09_DIR / "data"
MODELS_DIR = STAGE_09_DIR / "models"
OUTPUT_DIR = STAGE_09_DIR / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_models() -> Dict[str, Any]:
    """Instantiates candidate models for CV benchmarking."""
    models = {
        "HistGradientBoosting_Classifier": HistGradientBoostingClassifier(random_state=42),
        "FrankHall_Ordinal_HistGBDT": FrankHallOrdinalClassifier(
            lambda k: HistGradientBoostingClassifier(random_state=42, min_samples_leaf=15, max_iter=100)
        ),
        "HierarchicalGated_Classifier": HierarchicalGatedClassifier(),
        "XGBoost_Classifier": StandardizedClassifierWrapper(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42
            )
        ),
        "RandomForest_Classifier": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        ),
        "OrdinalThreshold_Regressor": OrdinalThresholdRegressor(
            HistGradientBoostingRegressor(random_state=42)
        ),
        "ConsolidatedSeverity_Ensemble": ConsolidatedSeverityEnsemble(
            weights=[0.70, 0.20, 0.10]
        )
    }
    return models


def evaluate_cv(
    model_name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> Dict[str, Any]:
    """Runs n-fold stratified cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, within1s, qwks, maes, f1s = [], [], [], [], []

    oof_preds = np.zeros(len(y), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Clone / re-instantiate fresh model for this fold
        candidate_dict = get_candidate_models()
        fold_model = candidate_dict[model_name]

        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_val)
        oof_preds[val_idx] = preds

        accs.append(accuracy_score(y_val, preds))
        within1s.append(np.mean(np.abs(y_val - preds) <= 1))
        qwks.append(cohen_kappa_score(y_val, preds, weights="quadratic"))
        maes.append(mean_absolute_error(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

    cm = confusion_matrix(y, oof_preds, labels=[1, 2, 3, 4, 5]).tolist()

    return {
        "model_name": model_name,
        "cv_exact_acc_mean": float(np.mean(accs)),
        "cv_exact_acc_std": float(np.std(accs)),
        "cv_within1_acc_mean": float(np.mean(within1s)),
        "cv_within1_acc_std": float(np.std(within1s)),
        "cv_qwk_mean": float(np.mean(qwks)),
        "cv_qwk_std": float(np.std(qwks)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_macro_f1_mean": float(np.mean(f1s)),
        "oof_confusion_matrix": cm
    }


def main():
    print("=" * 70)
    print("Stage 09: 5-Fold Stratified CV Severity Benchmark & Model Training")
    print("=" * 70)

    train_csv = DATA_DIR / "train_severity_features_stage09.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train feature file: {train_csv}")

    df_train = pd.read_csv(train_csv)
    feature_cols = [c for c in df_train.columns if c not in ["example_id", "target_severity"]]
    X_train = df_train[feature_cols].values
    y_train = df_train["target_severity"].values

    print(f"Loaded {len(df_train)} training samples with {len(feature_cols)} features.")

    candidates = get_candidate_models()
    benchmark_results = {}

    print(f"\nRunning 5-Fold Stratified Cross-Validation across {len(candidates)} models...\n")
    print(f"{'Model Name':<32} | {'Exact Acc':<10} | {'Within-1':<10} | {'QWK':<8} | {'MAE':<8}")
    print("-" * 76)

    for name, model in candidates.items():
        res = evaluate_cv(name, model, X_train, y_train, n_splits=5)
        benchmark_results[name] = res
        print(
            f"{name:<32} | "
            f"{res['cv_exact_acc_mean'] * 100:>5.2f}%    | "
            f"{res['cv_within1_acc_mean'] * 100:>5.2f}%    | "
            f"{res['cv_qwk_mean']:>6.4f} | "
            f"{res['cv_mae_mean']:>6.4f}"
        )

    # Determine champion based on Exact Accuracy (with high QWK tiebreaker)
    champion_name = max(
        benchmark_results.keys(),
        key=lambda k: benchmark_results[k]["cv_exact_acc_mean"]
    )
    print("-" * 76)
    print(f"★ Champion Model Selected: {champion_name}")
    print(f"  CV Exact Acc: {benchmark_results[champion_name]['cv_exact_acc_mean'] * 100:.2f}%")
    print(f"  CV Within-1 Acc: {benchmark_results[champion_name]['cv_within1_acc_mean'] * 100:.2f}%")
    print(f"  CV QWK: {benchmark_results[champion_name]['cv_qwk_mean']:.4f}")
    print(f"  CV MAE: {benchmark_results[champion_name]['cv_mae_mean']:.4f}")

    # Retrain champion on full training data
    champion_model = get_candidate_models()[champion_name]
    champion_model.fit(X_train, y_train)

    # Save model artifact
    model_save_path = MODELS_DIR / "best_severity_model_stage09.joblib"
    joblib.dump({
        "model_name": champion_name,
        "model": champion_model,
        "feature_names": feature_cols,
        "cv_metrics": benchmark_results[champion_name]
    }, model_save_path)
    print(f"✓ Saved champion model checkpoint to {model_save_path}")

    # Export CV benchmark JSON
    cv_json_path = OUTPUT_DIR / "train_cv_benchmark.json"
    with open(cv_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "champion": champion_name,
            "benchmark": benchmark_results
        }, f, indent=2)
    print(f"✓ Exported CV benchmark to {cv_json_path}")


if __name__ == "__main__":
    main()

