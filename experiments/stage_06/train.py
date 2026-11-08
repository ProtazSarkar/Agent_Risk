"""
Stage 06: Training & Model Selection Suite for Risk Severity (Levels 1 to 5).

Trains and benchmarks candidate tree-based and regression models:
  - Evaluates via 5-Fold Stratified Cross-Validation on train_severity_features.csv
  - Selects the top-performing model based on Quadratic Weighted Kappa (QWK) & Accuracy
  - Fits final models and persists checkpoints to experiments/stage_06/models/
"""

import os
import sys
import json
import time
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    root_mean_squared_error,
    f1_score
)

STAGE_06_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE_06_DIR.parent.parent

sys.path.insert(0, str(STAGE_06_DIR))
from models import build_candidate_models

DATA_DIR = STAGE_06_DIR / "data"
MODELS_DIR = STAGE_06_DIR / "models"
OUTPUT_DIR = STAGE_06_DIR / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    tr_path = DATA_DIR / "train_severity_features.csv"
    if not tr_path.exists():
        from prepare_dataset import prepare_stage_06_severity_datasets
        prepare_stage_06_severity_datasets()

    df_tr = pd.read_csv(tr_path)
    meta_cols = ["example_id", "risk_severity", "risk_domains", "summary_text"]
    feature_cols = [c for c in df_tr.columns if c not in meta_cols]

    X = df_tr[feature_cols].values
    y = df_tr["risk_severity"].values.astype(int)

    return df_tr, X, y, feature_cols


def evaluate_cross_validation(models: Dict[str, Any], X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Dict[str, float]]:
    """Runs 5-fold stratified cross-validation on all candidate models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {}

    print(f"\nRunning {n_splits}-Fold Stratified Cross-Validation across candidate models...")
    for name, model_factory in models.items():
        accs, qwks, within1s, maes = [], [], [], []
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fresh_models = build_candidate_models()
            model = fresh_models[name]
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            qwk = cohen_kappa_score(y_val, preds, weights="quadratic")
            within1 = np.mean(np.abs(y_val - preds) <= 1)
            mae = mean_absolute_error(y_val, preds)

            accs.append(acc)
            qwks.append(qwk)
            within1s.append(within1)
            maes.append(mae)

        elapsed = time.time() - t0
        cv_results[name] = {
            "cv_accuracy": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs)),
            "cv_qwk": float(np.mean(qwks)),
            "cv_qwk_std": float(np.std(qwks)),
            "cv_within1": float(np.mean(within1s)),
            "cv_mae": float(np.mean(maes)),
            "cv_time_s": float(elapsed)
        }
        print(f"  • {name:<32} | QWK: {np.mean(qwks):.4f} | Acc: {np.mean(accs)*100:.2f}% | Within-1: {np.mean(within1s)*100:.2f}% | MAE: {np.mean(maes):.4f}")

    return cv_results


def train_and_select_best_models():
    print("=" * 105)
    print("STAGE 06: SEVERITY (LEVELS 1-5) TRAINING & MODEL SELECTION SUITE")
    print("Features: Multi-domain probabilities + compound aggregates + PCA latents + keywords")
    print("=" * 105)

    df_tr, X, y, feature_cols = load_training_data()
    print(f"Loaded Training Data: {len(df_tr)} samples, {len(feature_cols)} features.")
    print(f"Target Distribution:\n{df_tr['risk_severity'].value_counts().sort_index()}")

    candidate_models = build_candidate_models()
    cv_scores = evaluate_cross_validation(candidate_models, X, y)

    ranked_models = sorted(cv_scores.items(), key=lambda x: (x[1]["cv_qwk"], x[1]["cv_accuracy"]), reverse=True)
    best_model_name = ranked_models[0][0]
    print(f"\n🏆 Champion Model Selected: '{best_model_name}' (CV QWK = {cv_scores[best_model_name]['cv_qwk']:.4f})")

    print("\nFitting final models on full training set...")
    final_models = build_candidate_models()
    saved_checkpoints = {}

    for name, model in final_models.items():
        model.fit(X, y)
        ckpt_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({
            "model": model,
            "feature_cols": feature_cols,
            "model_name": name,
            "cv_metrics": cv_scores[name]
        }, ckpt_path)
        saved_checkpoints[name] = str(ckpt_path)

    best_ckpt_path = MODELS_DIR / "best_severity_model.joblib"
    joblib.dump({
        "model": final_models[best_model_name],
        "feature_cols": feature_cols,
        "model_name": best_model_name,
        "cv_metrics": cv_scores[best_model_name],
        "all_cv_results": cv_scores
    }, best_ckpt_path)
    print(f"✓ Saved champion model to: {best_ckpt_path}")

    cv_summary_path = OUTPUT_DIR / "training_cv_results.json"
    with open(cv_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "champion_model": best_model_name,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "ranked_models": ranked_models,
            "cv_scores": cv_scores
        }, f, indent=2)
    print(f"✓ Saved CV benchmark report to: {cv_summary_path}")

    return best_model_name, cv_scores


if __name__ == "__main__":
    train_and_select_best_models()

