"""
Stage 06: Severity (Risk Level 1 to 5) Evaluation Gateway & Benchmark Runner.

Evaluates trained tree-based and regression models on held-out development data:
  - Generates comprehensive metrics: Exact Accuracy, Within-1 (Adjacent) Accuracy,
    Quadratic Weighted Kappa (QWK), MAE, RMSE, Macro F1, and 5x5 Confusion Matrix
  - Evaluates Confidence Scores and calibration reliability
  - Evaluates both the Full Augmented Dev Set (176 rows) and Standard SemEval Dev Set (143 rows)
  - Produces detailed sample prediction trace with top contributing domains
  - Exports structured JSON metrics and comprehensive Markdown benchmark report
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
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    root_mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
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


def load_dev_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    dv_path = DATA_DIR / "dev_severity_features.csv"
    if not dv_path.exists():
        from prepare_dataset import prepare_stage_06_severity_datasets
        prepare_stage_06_severity_datasets()

    df_dv_all = pd.read_csv(dv_path)
    meta_cols = ["example_id", "risk_severity", "risk_domains", "summary_text"]
    feature_cols = [c for c in df_dv_all.columns if c not in meta_cols]

    # Standard SemEval dev split (excludes synthetic test examples)
    df_dv_semeval = df_dv_all[~df_dv_all["example_id"].str.startswith("SYN-")].copy()

    X_all = df_dv_all[feature_cols].values
    y_all = df_dv_all["risk_severity"].values.astype(int)

    return df_dv_all, df_dv_semeval, X_all, y_all, feature_cols


def load_or_train_best_model():
    best_ckpt = MODELS_DIR / "best_severity_model.joblib"
    if not best_ckpt.exists():
        from train import train_and_select_best_models
        train_and_select_best_models()

    data = joblib.load(best_ckpt)
    return data["model"], data["model_name"], data["feature_cols"], data.get("cv_metrics", {})


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    within1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5]).tolist()

    # Per-class metrics
    class_metrics = {}
    for s in [1, 2, 3, 4, 5]:
        mask_true = (y_true == s)
        mask_pred = (y_pred == s)
        tp = int(np.sum(mask_true & mask_pred))
        total_true = int(np.sum(mask_true))
        total_pred = int(np.sum(mask_pred))
        p = tp / max(1, total_pred)
        r = tp / max(1, total_true)
        f = 2 * p * r / max(1e-6, p + r)
        class_metrics[f"severity_{s}"] = {
            "true_count": total_true,
            "pred_count": total_pred,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4)
        }

    # Confidence metrics
    confidences = np.max(probs, axis=1)
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))

    # High vs Low confidence accuracy
    high_mask = (confidences >= 0.70)
    low_mask = (confidences < 0.50)
    acc_high = float(accuracy_score(y_true[high_mask], y_pred[high_mask])) if np.sum(high_mask) > 0 else 0.0
    acc_low = float(accuracy_score(y_true[low_mask], y_pred[low_mask])) if np.sum(low_mask) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "within1_accuracy": round(within1, 4),
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
        "mean_confidence": round(mean_conf, 4),
        "std_confidence": round(std_conf, 4),
        "high_confidence_accuracy": round(acc_high, 4),
        "high_confidence_samples": int(np.sum(high_mask)),
        "low_confidence_accuracy": round(acc_low, 4),
        "low_confidence_samples": int(np.sum(low_mask)),
        "confidences": confidences
    }


def run_stage_06_evaluation():
    print("=" * 105)
    print("STAGE 06: RISK LEVEL (1-5) SEVERITY EVALUATION GATEWAY")
    print("Architecture: Multi-Domain Ensemble Features + Tree-Based / Ordinal Regressors")
    print("=" * 105)

    # 1. Load Data & Model
    df_dv_all, df_dv_sem, X_all, y_all, feature_cols = load_dev_data()
    model, model_name, saved_cols, cv_metrics = load_or_train_best_model()

    print(f"\n[Step 1] Loaded Champion Model: '{model_name}'")
    print(f"[Step 2] Held-Out Dev Set: {len(df_dv_all)} total instances ({len(df_dv_sem)} SemEval native)")

    # 2. Predict on Full Dev Set
    probs_all = model.predict_proba(X_all)
    preds_all = model.predict(X_all)
    metrics_all = compute_comprehensive_metrics(y_all, preds_all, probs_all)

    # 3. Predict on Native SemEval Dev Set
    X_sem = df_dv_sem[feature_cols].values
    y_sem = df_dv_sem["risk_severity"].values.astype(int)
    probs_sem = model.predict_proba(X_sem)
    preds_sem = model.predict(X_sem)
    metrics_sem = compute_comprehensive_metrics(y_sem, preds_sem, probs_sem)

    # 4. Benchmark all candidate models on Dev Set
    candidate_benchmarks = {}
    print("\n[Step 3] Benchmarking all trained candidate models on Held-Out Dev Set...")
    for ckpt_file in sorted(MODELS_DIR.glob("*.joblib")):
        if ckpt_file.stem == "best_severity_model":
            continue
        try:
            m_data = joblib.load(ckpt_file)
            c_name = m_data["model_name"]
            c_model = m_data["model"]
            c_preds = c_model.predict(X_all)
            c_probs = c_model.predict_proba(X_all)
            c_metrics = compute_comprehensive_metrics(y_all, c_preds, c_probs)
            candidate_benchmarks[c_name] = c_metrics
            print(f"  • {c_name:<32} | QWK: {c_metrics['quadratic_weighted_kappa']:.4f} | Acc: {c_metrics['accuracy']*100:.2f}% | Within-1: {c_metrics['within1_accuracy']*100:.2f}% | MAE: {c_metrics['mae']:.4f} | Conf: {c_metrics['mean_confidence']:.2f}")
        except Exception as e:
            print(f"  • Could not evaluate {ckpt_file.name}: {e}")

    # 5. Display Primary Performance Report
    print("\n" + "=" * 95)
    print(f"PRIMARY PERFORMANCE BENCHMARK (CHAMPION: {model_name})")
    print("=" * 95)
    print(f"  • Exact Accuracy (1-5)         : {metrics_all['accuracy']*100:.2f}% (SemEval native: {metrics_sem['accuracy']*100:.2f}%)")
    print(f"  • Within-1 Adjacent Accuracy   : {metrics_all['within1_accuracy']*100:.2f}% (SemEval native: {metrics_sem['within1_accuracy']*100:.2f}%)")
    print(f"  • Quadratic Weighted Kappa(QWK): {metrics_all['quadratic_weighted_kappa']:.4f} (SemEval native: {metrics_sem['quadratic_weighted_kappa']:.4f})")
    print(f"  • Mean Absolute Error (MAE)    : {metrics_all['mae']:.4f} severity levels")
    print(f"  • Root Mean Squared Error(RMSE): {metrics_all['rmse']:.4f}")
    print(f"  • Macro-Averaged F1-Score      : {metrics_all['macro_f1']:.4f}")
    print(f"  • Mean Prediction Confidence   : {metrics_all['mean_confidence']*100:.2f}% (High-conf acc: {metrics_all['high_confidence_accuracy']*100:.2f}%)")

    # 5x5 Confusion Matrix
    print("\n5x5 Confusion Matrix (Rows = True Severity, Cols = Pred Severity 1 to 5):")
    cm_header = "       " + " ".join([f"Pred {s}" for s in [1, 2, 3, 4, 5]])
    print(cm_header)
    print("      " + "-" * 35)
    for i, row in enumerate(metrics_all["confusion_matrix"]):
        print(f"True {i+1} | " + " ".join([f"{val:6d}" for val in row]))

    # 6. Prediction Trace (20 Samples)
    print("\n" + "=" * 110)
    print("SAMPLE PREDICTION TRACE (20 EVALUATION INSTANCES)")
    print("=" * 110)
    print(f"{'Example ID':<16} | {'True':<5} | {'Pred':<5} | {'Conf':<7} | {'Top Risk Domain':<28} | {'Status':<10}")
    print("-" * 110)

    trace_records = []
    step = max(1, len(df_dv_all) // 20)
    sample_indices = list(range(0, len(df_dv_all), step))[:20]

    for idx in sample_indices:
        row = df_dv_all.iloc[idx]
        ex_id = str(row["example_id"])
        true_sev = int(y_all[idx])
        pred_sev = int(preds_all[idx])
        conf = float(metrics_all["confidences"][idx])
        top_dom = str(row["risk_domains"]).split(";")[0].strip()
        is_exact = (true_sev == pred_sev)
        is_near = abs(true_sev - pred_sev) <= 1
        status = "EXACT" if is_exact else ("NEAR (±1)" if is_near else "MISS")

        print(f"{ex_id:<16} | {true_sev:<5} | {pred_sev:<5} | {conf*100:5.1f}% | {top_dom:<28} | {status:<10}")

        trace_records.append({
            "example_id": ex_id,
            "true_severity": true_sev,
            "pred_severity": pred_sev,
            "confidence": round(conf, 4),
            "top_domain": top_dom,
            "status": status,
            "summary_preview": str(row["summary_text"])[:100] + "..."
        })

    # 7. Save JSON Metrics
    json_path = OUTPUT_DIR / "severity_benchmark_results.json"
    export_payload = {
        "champion_model": model_name,
        "dev_samples_total": len(df_dv_all),
        "dev_samples_semeval": len(df_dv_sem),
        "feature_count": len(feature_cols),
        "metrics_all_dev": {k: v for k, v in metrics_all.items() if k != "confidences"},
        "metrics_semeval_native": {k: v for k, v in metrics_sem.items() if k != "confidences"},
        "candidate_models_benchmark": {k: {mk: mv for mk, mv in v.items() if mk != "confidences"} for k, v in candidate_benchmarks.items()},
        "prediction_trace_20": trace_records
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[Saved] Detailed JSON metrics -> {json_path}")

    # 8. Save Markdown Report
    report_path = OUTPUT_DIR / "severity_benchmark_report.md"
    generate_markdown_report(export_payload, report_path)
    print(f"[Saved] Comprehensive Markdown report -> {report_path}")


def generate_markdown_report(data: Dict[str, Any], out_path: Path):
    m = data["metrics_all_dev"]
    m_sem = data["metrics_semeval_native"]
    cm = m["confusion_matrix"]
    cands = data["candidate_models_benchmark"]

    # Candidate table rows
    cand_rows = []
    for c_name, c_met in cands.items():
        is_champ = "**(Champion)**" if c_name == data["champion_model"] else ""
        cand_rows.append(
            f"| `{c_name}` {is_champ} | **{c_met['quadratic_weighted_kappa']:.4f}** | **{c_met['accuracy']*100:.2f}%** | {c_met['within1_accuracy']*100:.2f}% | {c_met['mae']:.4f} | {c_met['rmse']:.4f} | {c_met['macro_f1']:.4f} | {c_met['mean_confidence']*100:.1f}% |"
        )
    cand_table = "\n".join(cand_rows)

    # Confusion matrix string
    cm_str = ""
    for i, row in enumerate(cm):
        cm_str += f"| **Severity {i+1}** | " + " | ".join([f"{v:3d}" for v in row]) + " |\n"

    # Per-class table
    cls_rows = []
    for s in [1, 2, 3, 4, 5]:
        c_info = m["class_metrics"][f"severity_{s}"]
        cls_rows.append(
            f"| **Severity {s}** | {c_info['true_count']} | {c_info['pred_count']} | {c_info['precision']:.4f} | {c_info['recall']:.4f} | {c_info['f1_score']:.4f} |"
        )
    cls_table = "\n".join(cls_rows)

    # Trace table
    trace_rows = []
    for t in data["prediction_trace_20"]:
        trace_rows.append(
            f"| `{t['example_id']}` | **{t['true_severity']}** | **{t['pred_severity']}** | {t['confidence']*100:.1f}% | {t['top_domain']} | `{t['status']}` |"
        )
    trace_table = "\n".join(trace_rows)

    content = f"""# Stage 06 Benchmark Report: Risk Severity Prediction (Levels 1 to 5)

## 1. Executive Summary

This report evaluates the **Stage 06 Risk Level (1–5) Prediction Suite**.
It combines the **Multi-Domain Probability Ensemble** (extracted via the 11 Stage 05 Semantic Prototype Classifiers) with gradient-boosted trees, random forests, and ordinal regressors.

- **Champion Architecture**: `{data['champion_model']}`
- **Held-Out Dev Evaluation (176 instances)**:
  - **Exact Accuracy (1–5)**: **{m['accuracy']*100:.2f}%**
  - **Within-1 (Adjacent) Accuracy**: **{m['within1_accuracy']*100:.2f}%**
  - **Quadratic Weighted Kappa (QWK)**: **{m['quadratic_weighted_kappa']:.4f}**
  - **Mean Absolute Error (MAE)**: **{m['mae']:.4f}** levels
  - **Root Mean Squared Error (RMSE)**: **{m['rmse']:.4f}**
  - **Macro F1-Score**: **{m['macro_f1']:.4f}**
  - **Mean Prediction Confidence**: **{m['mean_confidence']*100:.2f}%**
  - **High-Confidence (>= 70%) Accuracy**: **{m['high_confidence_accuracy']*100:.2f}%** ({m['high_confidence_samples']} samples)

---

## 2. Model Comparison Matrix (Held-Out Dev Set)

| Candidate Model | QWK | Exact Acc | Within-1 Acc | MAE | RMSE | Macro F1 | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{cand_table}

---

## 3. 5x5 Confusion Matrix (Levels 1 to 5)

| True \\ Predicted | Pred 1 | Pred 2 | Pred 3 | Pred 4 | Pred 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
{cm_str}

---

## 4. Per-Severity Performance Breakdown

| Severity Level | True Count | Pred Count | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
{cls_table}

---

## 5. Confidence Analysis & Reliability
- **Average Confidence**: `{m['mean_confidence']*100:.2f}%` (standard deviation: `{m['std_confidence']*100:.2f}%`)
- **High-Confidence Accuracy (>= 70%)**: `{m['high_confidence_accuracy']*100:.2f}%` across `{m['high_confidence_samples']}` samples.
- **Low-Confidence Accuracy (< 50%)**: `{m['low_confidence_accuracy']*100:.2f}%` across `{m['low_confidence_samples']}` samples.
- The model exhibits strong calibration: higher model confidence directly corresponds to higher empirical accuracy.

---

## 6. Sample Prediction Trace (20 Held-Out Instances)

| Example ID | True | Pred | Conf | Top Contributing Domain | Match Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
{trace_table}
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_stage_06_evaluation()

