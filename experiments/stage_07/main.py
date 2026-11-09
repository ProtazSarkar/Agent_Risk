"""
Stage 07 Evaluation Gateway: Domain 01 (No Material Harm) Benchmark.

Evaluates the champion Stage07DomainClassifier on the held-out SemEval dev set:
  - Exact Accuracy, Macro F1, Precision, Recall, and ROC-AUC
  - 2x2 Confusion Matrix
  - Dynamic Section Attention Weights breakdown (query, action, thought, observation)
  - Confidence calibration and 20-sample prediction trace
  - Exports output/domain_01_results.json and output/domain_01_report.md
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.stage_07.model import Stage07DomainClassifier

STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
MODELS_DIR = STAGE_DIR / "models"
OUTPUT_DIR = STAGE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODELS_DIR / "best_domain_01_model.pt"
DEV_DATA_PATH = DATA_DIR / "dev_sections.pt"


def run_evaluation(device_str: str = "cpu") -> Dict[str, Any]:
    print("=" * 70)
    print("Stage 07: Domain 01 (No Material Harm) Evaluation Gateway")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Run train.py first.")
    if not DEV_DATA_PATH.exists():
        raise FileNotFoundError(f"Dev data not found at {DEV_DATA_PATH}. Run prepare_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    # 1. Load Checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    proto_keys = checkpoint["prototype_keys"].to(device)
    num_pos = 4

    model = Stage07DomainClassifier(
        initial_prototype_keys=proto_keys,
        num_pos_prototypes=num_pos,
        embed_dim=384,
        hidden_dim=384,
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load Dev Data
    dev_data = torch.load(DEV_DATA_PATH, weights_only=False)
    sections = dev_data["sections_tensor"].to(device)  # (143, 4, 384)
    labels = dev_data["labels"].numpy().astype(int)     # (143,)
    example_ids = dev_data["example_ids"]
    severities = dev_data["severities"]
    domains_list = dev_data["domains_list"]
    total = len(labels)

    # 3. Model Inference
    with torch.no_grad():
        probs_t, logits_t, meta_t = model(sections)

    probs = probs_t.cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    sec_weights = meta_t["section_weights"].cpu().numpy()  # (143, 4)
    sims = meta_t["prototype_similarities"].cpu().numpy()

    # 4. Metric Calculations
    acc = accuracy_score(labels, preds)
    p_safe, r_safe, f1_safe, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary", zero_division=0)
    p_harm, r_harm, f1_harm, _ = precision_recall_fscore_support(labels, preds, pos_label=0, average="binary", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(labels, preds, labels=[1, 0])
    # cm: [[TP (Safe->Safe), FN (Safe->Harm)], [FP (Harm->Safe), TN (Harm->Harm)]]
    tp, fn = cm[0][0], cm[0][1]
    fp, tn = cm[1][0], cm[1][1]

    # Section Weights Analysis
    mean_weights = {
        "query": float(sec_weights[:, 0].mean()),
        "action": float(sec_weights[:, 1].mean()),
        "thought": float(sec_weights[:, 2].mean()),
        "observation": float(sec_weights[:, 3].mean())
    }

    # Confidence Calibration
    confidences = np.where(preds == 1, probs, 1.0 - probs)
    high_mask = confidences >= 0.70
    med_mask = (confidences >= 0.55) & (confidences < 0.70)
    low_mask = confidences < 0.55

    high_acc = float(accuracy_score(labels[high_mask], preds[high_mask])) if high_mask.sum() > 0 else 0.0
    med_acc = float(accuracy_score(labels[med_mask], preds[med_mask])) if med_mask.sum() > 0 else 0.0
    low_acc = float(accuracy_score(labels[low_mask], preds[low_mask])) if low_mask.sum() > 0 else 0.0

    print(f"\nHeld-Out SemEval Dev Evaluation ({total} samples):")
    print(f"  Exact Accuracy:     {acc * 100:.2f}%")
    print(f"  Macro F1:           {f1_macro:.4f}")
    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  Safe (1) F1:        {f1_safe:.4f} (Prec: {p_safe:.4f}, Rec: {r_safe:.4f})")
    print(f"  Harmful (0) F1:     {f1_harm:.4f} (Prec: {p_harm:.4f}, Rec: {r_harm:.4f})")
    print(f"  High-Conf (>=70%):  {high_acc * 100:.2f}% ({high_mask.sum()}/{total} samples)")
    print(f"\nMean Section Attention Distribution:")
    for k, v in mean_weights.items():
        print(f"  {k:12s}: {v * 100:.2f}%")

    print("\nConfusion Matrix [Safe=1, Harmful=0]:")
    print(f"  True Safe    -> Pred Safe: {tp:3d} | Pred Harm: {fn:3d}")
    print(f"  True Harmful -> Pred Safe: {fp:3d} | Pred Harm: {tn:3d}")

    # Build 20-Sample Trace
    trace_samples = []
    for i in range(min(20, total)):
        trace_samples.append({
            "example_id": example_ids[i],
            "true_label": "Safe (1)" if labels[i] == 1 else "Harmful (0)",
            "pred_label": "Safe (1)" if preds[i] == 1 else "Harmful (0)",
            "safe_prob": float(probs[i]),
            "confidence": float(confidences[i]),
            "risk_severity": int(severities[i]),
            "risk_domains": domains_list[i],
            "correct": bool(labels[i] == preds[i]),
            "weights": {
                "query": float(sec_weights[i, 0]),
                "action": float(sec_weights[i, 1]),
                "thought": float(sec_weights[i, 2]),
                "obs": float(sec_weights[i, 3])
            }
        })

    results = {
        "dataset": "SemEval-2027 AgentRisk Dev Split",
        "domain": "Domain 01: No Material Harm",
        "num_samples": total,
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "roc_auc": float(auc),
            "safe_precision": float(p_safe),
            "safe_recall": float(r_safe),
            "safe_f1": float(f1_safe),
            "harm_precision": float(p_harm),
            "harm_recall": float(r_harm),
            "harm_f1": float(f1_harm),
            "high_confidence_accuracy": high_acc,
            "high_confidence_count": int(high_mask.sum())
        },
        "mean_section_weights": mean_weights,
        "confusion_matrix": {
            "tp_safe": int(tp),
            "fn_safe": int(fn),
            "fp_harm": int(fp),
            "tn_harm": int(tn)
        },
        "sample_traces": trace_samples
    }

    # Save JSON results
    with open(OUTPUT_DIR / "domain_01_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Report
    report_md = f"""# Stage 07 Benchmark Report: Domain 01 (No Material Harm)
**Architecture**: Multi-Section Chunked Embedding Optimizer + Semantic Prototype Classifier  
**Dataset**: SemEval AgentRisk Held-Out Dev Set ({total} samples)

---

## 1. Executive Performance Summary

| Metric | Score | Baseline / Note |
| :--- | :---: | :--- |
| **Exact Accuracy** | **{acc * 100:.2f}%** | Binary Safe (1) vs Harmful (0) |
| **Macro F1-Score** | **{f1_macro:.4f}** | Harmonic mean across both classes |
| **ROC-AUC** | **{auc:.4f}** | Area under ROC curve |
| **Safe (1) Precision / Recall / F1** | **{p_safe:.4f} / {r_safe:.4f} / {f1_safe:.4f}** | Class count: {int((labels==1).sum())} |
| **Harmful (0) Precision / Recall / F1** | **{p_harm:.4f} / {r_harm:.4f} / {f1_harm:.4f}** | Class count: {int((labels==0).sum())} |
| **High-Confidence (>=70%) Accuracy** | **{high_acc * 100:.2f}%** | {high_mask.sum()} / {total} samples ({high_mask.sum()/total*100:.1f}%) |

---

## 2. Learned Section Attention Distribution

The dynamic `SectionEmbeddingOptimizer` learned the following average attention weights across the interaction:

| Section Tag | Mean Attention Weight | Role & Semantic Focus |
| :--- | :---: | :--- |
| **`[ACTION]`** (+ Proposed Action) | **{mean_weights['action']*100:.1f}%** | Direct tool invocations and pending execution commands |
| **`[OBSERVATION]`** | **{mean_weights['observation']*100:.1f}%** | Tool outputs, exfiltrated files, and system feedback |
| **`[USER]` (`query`)** | **{mean_weights['query']*100:.1f}%** | Initial prompt context and user instructions |
| **`[THOUGHT]`** | **{mean_weights['thought']*100:.1f}%** | Agent chain-of-thought and internal deliberation |

---

## 3. Confusion Matrix

| True \\ Predicted | Predicted Safe (1) | Predicted Harmful (0) | Total | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **True Safe (No Material Harm)** | **{tp}** (TP) | {fn} (FN) | {tp+fn} | {r_safe*100:.1f}% |
| **True Harmful (Material Harm Occurs)** | {fp} (FP) | **{tn}** (TN) | {fp+tn} | {r_harm*100:.1f}% |

---

## 4. 20-Sample Prediction Trace

| ID | True Class | Pred Class | Safe Prob | Conf. | Severity | Correct | Query Wt | Action Wt | Obs Wt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for t in trace_samples:
        corr_mark = "PASS" if t["correct"] else "FAIL"
        report_md += (
            f"| `{t['example_id']}` | {t['true_label']} | {t['pred_label']} | "
            f"{t['safe_prob']:.3f} | {t['confidence']*100:.1f}% | {t['risk_severity']} | "
            f"**{corr_mark}** | {t['weights']['query']*100:.1f}% | {t['weights']['action']*100:.1f}% | {t['weights']['obs']*100:.1f}% |\n"
        )

    with open(OUTPUT_DIR / "domain_01_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nReport generated at {OUTPUT_DIR / 'domain_01_report.md'}")
    return results


def main():
    run_evaluation()


if __name__ == "__main__":
    main()

