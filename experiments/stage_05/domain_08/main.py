"""
Stage 05: domain_08 ('Reputational / Social / Psychological') Evaluation Gateway & Benchmark Runner.

Evaluates the SemanticPrototypeClassifier trained on the balanced, summarized embeddings:
1. Loads the trained checkpoint from models/domain_08_classifier.pt
2. Evaluates performance on held-out development set
3. Displays 15-sample prediction trace for 'Reputational / Social / Psychological'
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
    from experiments.stage_05.domain_08.model import SemanticPrototypeClassifier
    from experiments.stage_05.domain_08.train import train_domain_08_model
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(STAGE_DIR / "domain_08"))
    from model import SemanticPrototypeClassifier
    from train import train_domain_08_model


def load_or_train_model() -> Tuple[SemanticPrototypeClassifier, float, Dict[str, Any]]:
    models_dir = STAGE_DIR / "models"
    model_path = models_dir / "domain_08_classifier.pt"

    if not model_path.exists():
        print(f"Checkpoint not found at {model_path}. Running training suite first...")
        model, metrics = train_domain_08_model(save_model=True)
        return model, metrics["optimal_threshold"], metrics

    print(f"Loading trained domain_08 Classifier from: {model_path}")
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


def run_domain_08_evaluation():
    print("=" * 95)
    print("STAGE 05: domain_08 ('Reputational / Social / Psychological') EVALUATION GATEWAY")
    print("Evaluating SemanticPrototypeClassifier on Balanced, Non-Truncated 384-d Embeddings")
    print("=" * 95)

    # 1. Load Dev Dataset
    data_dir = STAGE_DIR / "data"
    dev_path = data_dir / "dev_domain_08.csv"

    if not dev_path.exists():
        raise FileNotFoundError(f"Processed dev dataset not found at {dev_path}.")

    df_dev = pd.read_csv(dev_path)
    emb_cols = [c for c in df_dev.columns if c.startswith("emb_")]
    X_dv = torch.tensor(df_dev[emb_cols].values, dtype=torch.float32)
    y_dv = df_dev["label"].values.astype(int)

    print(f"\n[Step 1] Dev Dataset Loaded: {len(df_dev)} instances (Positives: {(y_dv == 1).sum()}, Negatives: {(y_dv == 0).sum()})")

    # 2. Load Model
    model, opt_th, train_metrics = load_or_train_model()

    # 3. Predict Probabilities & Labels
    dev_probs = model.predict_proba(X_dv)

    preds_05 = (dev_probs >= 0.50).astype(int)
    preds_opt = (dev_probs >= opt_th).astype(int)

    acc_05 = accuracy_score(y_dv, preds_05)
    f1_05 = f1_score(y_dv, preds_05, zero_division=0)
    prec_05 = precision_score(y_dv, preds_05, zero_division=0)
    rec_05 = recall_score(y_dv, preds_05, zero_division=0)
    auc = roc_auc_score(y_dv, dev_probs) if len(np.unique(y_dv)) > 1 else 0.5
    cm_05 = confusion_matrix(y_dv, preds_05).tolist()

    acc_opt = accuracy_score(y_dv, preds_opt)
    f1_opt = f1_score(y_dv, preds_opt, zero_division=0)
    prec_opt = precision_score(y_dv, preds_opt, zero_division=0)
    rec_opt = recall_score(y_dv, preds_opt, zero_division=0)
    cm_opt = confusion_matrix(y_dv, preds_opt).tolist()

    print(f"\n[Step 2] Probability Distribution Analysis:")
    print(f"  • Min Probability : {dev_probs.min():.4f}")
    print(f"  • Max Probability : {dev_probs.max():.4f}")
    print(f"  • Mean Probability: {dev_probs.mean():.4f}")
    print(f"  • Std Probability : {dev_probs.std():.4f}")

    print(f"\n[Step 3] Benchmark Performance on Held-Out Dev Set (domain_08):")
    print(f"  • Default Cutoff (0.50): Acc = {acc_05*100:.2f}% | F1 = {f1_05:.4f} | Prec = {prec_05:.4f} | Rec = {rec_05:.4f} | AUC = {auc:.4f}")
    print(f"  • Tuned Cutoff ({opt_th:.3f}) : Acc = {acc_opt*100:.2f}% | F1 = {f1_opt:.4f} | Prec = {prec_opt:.4f} | Rec = {rec_opt:.4f}")
    print(f"  • Confusion Matrix (0.50):")
    print(f"    [[TN={cm_05[0][0]}, FP={cm_05[0][1]}],")
    print(f"     [FN={cm_05[1][0]}, TP={cm_05[1][1]}]]")

    # 4. Detailed Sample Predictions Trace
    print(f"\n[Step 4] Detailed Prediction Trace (15 Sample Instances on Dev Split):")
    print("-" * 90)
    print(f"{'ID':<14} | {'Prob(Pos)':<14} | {'Pred':<12} | {'True Label':<12} | Status")
    print("-" * 90)

    trace_records = []
    step = max(1, len(df_dev) // 15)
    sample_indices = list(range(0, len(df_dev), step))[:15]

    for idx in sample_indices:
        ex_id = str(df_dev.iloc[idx].get("example_id", f"EX-{idx:03d}"))
        prob = float(dev_probs[idx])
        pred_label = "Reputational / Social / Psychological (1)" if prob >= 0.50 else "Other (0)"
        true_label = "Reputational / Social / Psychological (1)" if y_dv[idx] == 1 else "Other (0)"
        is_correct = (prob >= 0.50) == (y_dv[idx] == 1)
        status = "CORRECT" if is_correct else "MISS"

        print(f"{ex_id:<14} | {prob:14.4f} | {pred_label:<12} | {true_label:<12} | {status}")

        trace_records.append({
            "example_id": ex_id,
            "prob": round(prob, 4),
            "prediction": pred_label,
            "ground_truth": true_label,
            "status": status
        })

    # 5. Export JSON Results
    results_dir = STAGE_DIR / "output"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "domain_08_results.json"

    export_payload = {
        "domain_id": "domain_08",
        "domain_name": "Reputational / Social / Psychological",
        "dev_instances": len(df_dev),
        "dev_positives": int((y_dv == 1).sum()),
        "accuracy_default": float(acc_05),
        "f1_default": float(f1_05),
        "precision_default": float(prec_05),
        "recall_default": float(rec_05),
        "roc_auc": float(auc),
        "optimal_threshold": float(opt_th),
        "accuracy_opt": float(acc_opt),
        "f1_opt": float(f1_opt),
        "precision_opt": float(prec_opt),
        "recall_opt": float(rec_opt),
        "confusion_matrix_default": cm_05,
        "confusion_matrix_opt": cm_opt,
        "prob_min": float(dev_probs.min()),
        "prob_max": float(dev_probs.max()),
        "prob_mean": float(dev_probs.mean()),
        "prob_std": float(dev_probs.std()),
        "trace_samples": trace_records
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[Saved] Detailed evaluation metrics -> {json_path}")

    # 6. Export Markdown Report
    md_path = results_dir / "domain_08_report.md"
    generate_markdown_report(export_payload, md_path)
    print(f"[Saved] Comprehensive Markdown report -> {md_path}")


def generate_markdown_report(data: Dict[str, Any], out_path: Path):
    cm = data["confusion_matrix_default"]
    cm_opt = data["confusion_matrix_opt"]

    content = f"""# Stage 05 Benchmark Report: domain_08 ('Reputational / Social / Psychological')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_08: 'Reputational / Social / Psychological'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: {data['dev_instances']} ({data['dev_positives']} Positive, {data['dev_instances'] - data['dev_positives']} Negative)
- **Zero-Truncation Embeddings**: 384-dimensional dense vectors generated from distilled 60–90 word summaries via `all-MiniLM-L6-v2`.
- **Domain Prototypes**: Pre-computed SentenceTransformer key vectors covering positive risk manifestations and negative safe anchors.
- **Architecture**:
  - `query_proj`: 384 -> 64 -> LayerNorm -> GELU -> Dropout
  - `key_proj`: 384 -> 64 -> LayerNorm
  - `fusion_head`: Attentively combines multi-prototype similarities with latent query features.
  - Temperature scaling (gamma = 10.0).

---

## 2. Benchmark Results on Held-Out Dev Set

| Evaluation Mode | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Default Cutoff (0.50)** | `0.50` | **{data['accuracy_default']*100:.2f}%** | {data['precision_default']:.4f} | {data['recall_default']:.4f} | **{data['f1_default']:.4f}** | **{data['roc_auc']:.4f}** | `TN={cm[0][0]}, FP={cm[0][1]} / FN={cm[1][0]}, TP={cm[1][1]}` |
| **Optimal F1 Cutoff** | `{data['optimal_threshold']:.3f}` | **{data['accuracy_opt']*100:.2f}%** | {data['precision_opt']:.4f} | {data['recall_opt']:.4f} | **{data['f1_opt']:.4f}** | {data['roc_auc']:.4f} | `TN={cm_opt[0][0]}, FP={cm_opt[0][1]} / FN={cm_opt[1][0]}, TP={cm_opt[1][1]}` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `{data['prob_min']:.4f}`
- **Maximum Probability**: `{data['prob_max']:.4f}`
- **Mean Probability**   : `{data['prob_mean']:.4f}`
- **Std Deviation**     : `{data['prob_std']:.4f}`
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_domain_08_evaluation()
