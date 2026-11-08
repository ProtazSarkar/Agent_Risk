"""
Stage 05: Domain 02 ('Privacy / Confidentiality') Evaluation Gateway & Benchmark Runner.

Evaluates the SemanticPrototypeClassifier trained on the balanced, summarized Privacy embeddings:
1. Loads the trained checkpoint from models/domain_02_classifier.pt
2. Evaluates performance on held-out development set (146 instances)
3. Displays 15-sample prediction trace for Privacy / Confidentiality
4. Exports structured JSON metrics and comprehensive Markdown report
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

STAGE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = STAGE_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from experiments.stage_05.domain_02.model import SemanticPrototypeClassifier
    from experiments.stage_05.domain_02.train import train_domain_02_model
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(STAGE_DIR / "domain_02"))
    from model import SemanticPrototypeClassifier
    from train import train_domain_02_model


def load_or_train_model() -> Tuple[SemanticPrototypeClassifier, float, Dict[str, Any]]:
    models_dir = STAGE_DIR / "models"
    model_path = models_dir / "domain_02_classifier.pt"

    if not model_path.exists():
        print(f"Checkpoint not found at {model_path}. Running training suite first...")
        model, metrics = train_domain_02_model(save_model=True)
        return model, metrics["optimal_threshold"], metrics

    print(f"Loading trained Domain 02 Classifier from: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    prototype_keys = checkpoint["prototype_keys"]
    num_pos = checkpoint.get("num_pos", 4)
    num_neg = checkpoint.get("num_neg", 4)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    opt_th = checkpoint.get("optimal_threshold", 0.50)
    saved_metrics = checkpoint.get("metrics", {})

    model = SemanticPrototypeClassifier(
        initial_prototype_embeddings=prototype_keys,
        num_pos=num_pos,
        num_neg=num_neg,
        hidden_dim=hidden_dim
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded successfully. Scale: {model.scale.item():.2f} | Bias: {model.bias.item():.2f}")
    return model, opt_th, saved_metrics


def run_domain_02_evaluation():
    print("=" * 95)
    print("STAGE 05: DOMAIN 02 ('PRIVACY / CONFIDENTIALITY') EVALUATION GATEWAY")
    print("Evaluating SemanticPrototypeClassifier on Balanced, Non-Truncated 384-d Embeddings")
    print("=" * 95)

    # 1. Load Dev Dataset
    data_dir = STAGE_DIR / "data"
    dev_path = data_dir / "dev_domain_02.csv"

    if not dev_path.exists():
        raise FileNotFoundError(f"Processed dev dataset not found at {dev_path}.")

    df_dev = pd.read_csv(dev_path)
    emb_cols = [c for c in df_dev.columns if c.startswith("emb_")]
    X_dv = torch.tensor(df_dev[emb_cols].values, dtype=torch.float32)
    y_dv = df_dev["label"].values.astype(int)

    print(f"\n[Step 1] Dev Dataset Loaded: {len(df_dev)} instances (Privacy: {(y_dv == 1).sum()}, Non-Privacy: {(y_dv == 0).sum()})")

    # 2. Load Model
    model, opt_th, train_metrics = load_or_train_model()

    # 3. Predict Probabilities & Labels
    dev_probs = model.predict_proba(X_dv)

    preds_05 = (dev_probs >= 0.50).astype(int)
    preds_opt = (dev_probs >= opt_th).astype(int)

    acc_05 = accuracy_score(y_dv, preds_05)
    prec_05 = precision_score(y_dv, preds_05, zero_division=0)
    rec_05 = recall_score(y_dv, preds_05, zero_division=0)
    f1_05 = f1_score(y_dv, preds_05, zero_division=0)

    acc_opt = accuracy_score(y_dv, preds_opt)
    prec_opt = precision_score(y_dv, preds_opt, zero_division=0)
    rec_opt = recall_score(y_dv, preds_opt, zero_division=0)
    f1_opt = f1_score(y_dv, preds_opt, zero_division=0)

    auc = roc_auc_score(y_dv, dev_probs)
    cm_05 = confusion_matrix(y_dv, preds_05, labels=[0, 1])
    cm_opt = confusion_matrix(y_dv, preds_opt, labels=[0, 1])

    # 4. Display Analysis
    print("\n[Step 2] Probability Distribution Analysis:")
    print(f"  • Min Probability : {dev_probs.min():.4f}")
    print(f"  • Max Probability : {dev_probs.max():.4f}")
    print(f"  • Mean Probability: {dev_probs.mean():.4f}")
    print(f"  • Std Probability : {dev_probs.std():.4f}")

    print("\n[Step 3] Benchmark Performance on Held-Out Dev Set (Domain 02):")
    print(f"  • Default Cutoff (0.50): Acc = {acc_05*100:5.2f}% | F1 = {f1_05:.4f} | Prec = {prec_05:.4f} | Rec = {rec_05:.4f} | AUC = {auc:.4f}")
    print(f"  • Tuned Cutoff ({opt_th:.3f}) : Acc = {acc_opt*100:5.2f}% | F1 = {f1_opt:.4f} | Prec = {prec_opt:.4f} | Rec = {rec_opt:.4f}")
    print(f"  • Confusion Matrix (0.50):\n    [[TN={cm_05[0][0]}, FP={cm_05[0][1]}],\n     [FN={cm_05[1][0]}, TP={cm_05[1][1]}]]")

    # 5. 15-Sample Prediction Trace
    print("\n[Step 4] Detailed Prediction Trace (15 Sample Instances on Dev Split):")
    print("-" * 90)
    print(f"{'ID':<14} | {'Prob(Privacy)':<14} | {'Pred':<12} | {'True Label':<12} | {'Status'}")
    print("-" * 90)

    traces = []
    for idx in range(min(15, len(df_dev))):
        ex_id = df_dev["example_id"].iloc[idx]
        p_val = float(dev_probs[idx])
        y_pred = int(preds_05[idx])
        y_true = int(y_dv[idx])
        status = "CORRECT" if y_pred == y_true else "MISS"

        pred_name = "Privacy (1)" if y_pred == 1 else "Other (0)"
        true_name = "Privacy (1)" if y_true == 1 else "Other (0)"

        print(f"{ex_id:<14} | {p_val:14.4f} | {pred_name:<12} | {true_name:<12} | {status}")

        traces.append({
            "example_id": str(ex_id),
            "predicted_probability": round(p_val, 4),
            "predicted_label": y_pred,
            "true_label": y_true,
            "status": status
        })

    # 6. Save JSON & Markdown Reports
    output_dir = STAGE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "stage": "Stage 05",
        "domain": "Domain 02: Privacy / Confidentiality",
        "model_architecture": "SemanticPrototypeClassifier",
        "dev_samples": len(df_dev),
        "dev_positives": int((y_dv == 1).sum()),
        "dev_negatives": int((y_dv == 0).sum()),
        "prob_min": float(dev_probs.min()),
        "prob_max": float(dev_probs.max()),
        "prob_mean": float(dev_probs.mean()),
        "default_threshold": 0.50,
        "optimal_threshold": float(opt_th),
        "accuracy_default": float(acc_05),
        "f1_default": float(f1_05),
        "precision_default": float(prec_05),
        "recall_default": float(rec_05),
        "accuracy_opt": float(acc_opt),
        "f1_opt": float(f1_opt),
        "precision_opt": float(prec_opt),
        "recall_opt": float(rec_opt),
        "roc_auc": float(auc),
        "confusion_matrix_default": cm_05.tolist(),
        "confusion_matrix_opt": cm_opt.tolist(),
        "sample_traces": traces
    }

    results_json = output_dir / "domain_02_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[Saved] Detailed evaluation metrics -> {results_json}")

    report_md = output_dir / "domain_02_report.md"
    generate_markdown_report(results_data, report_md)
    print(f"[Saved] Comprehensive Markdown report -> {report_md}")

    return results_data


def generate_markdown_report(data: Dict[str, Any], out_path: Path):
    cm = data["confusion_matrix_default"]
    cm_opt = data["confusion_matrix_opt"]

    content = f"""# Stage 05: Domain 02 ('Privacy / Confidentiality') Classifier Report

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **Domain 02: 'Privacy / Confidentiality'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Train: 580 instances (192 Privacy Violations, 388 Negative)
  - Dev  : 146 instances (48 Privacy Violations, 98 Negative)
- **Zero-Truncation Embeddings**: 384-dimensional dense vectors generated from distilled 60–90 word summaries via `all-MiniLM-L6-v2`.
- **Domain Prototypes**: Pre-computed SentenceTransformer key vectors covering:
  - Identity / PII data exfiltration
  - Medical records & patient health data (EHR) leakage
  - Excessive device & application permission escalation
  - Proprietary business documents and employee records
- **Architecture**:
  - `query_proj`: 384 $\\rightarrow$ 64 $\\rightarrow$ LayerNorm $\\rightarrow$ GELU $\\rightarrow$ Dropout
  - `key_proj`: 384 $\\rightarrow$ 64 $\\rightarrow$ LayerNorm
  - `fusion_head`: Attentively combines multi-prototype similarities with latent query features.
  - Temperature scaling ($\\gamma = 10.0$).

---

## 2. Benchmark Results on Held-Out Dev Set (146 Instances)

| Evaluation Mode | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Default Cutoff (0.50)** | `0.50` | **{data['accuracy_default']*100:.2f}%** | {data['precision_default']:.4f} | {data['recall_default']:.4f} | **{data['f1_default']:.4f}** | **{data['roc_auc']:.4f}** | `TN={cm[0][0]}, FP={cm[0][1]} / FN={cm[1][0]}, TP={cm[1][1]}` |
| **Optimal F1 Cutoff** | `{data['optimal_threshold']:.3f}` | **{data['accuracy_opt']*100:.2f}%** | {data['precision_opt']:.4f} | {data['recall_opt']:.4f} | **{data['f1_opt']:.4f}** | {data['roc_auc']:.4f} | `TN={cm_opt[0][0]}, FP={cm_opt[0][1]} / FN={cm_opt[1][0]}, TP={cm_opt[1][1]}` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `{data['prob_min']:.4f}`
- **Maximum Probability**: `{data['prob_max']:.4f}`
- **Mean Probability**   : `{data['prob_mean']:.4f}`
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_domain_02_evaluation()
