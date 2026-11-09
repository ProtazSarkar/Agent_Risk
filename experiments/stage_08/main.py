"""
Stage 08 Evaluation Gateway: 5-Channel Cross-Attentive Prototype Benchmark.

Evaluates Stage08OptimizedDomainClassifier on the held-out SemEval dev set:
  - Exact Accuracy, Macro F1, ROC-AUC, Precision, and Recall
  - 2x2 Confusion Matrix
  - 5-Channel Attention Pooling breakdown (query, thought, context_actions, observations, proposed_action)
  - Head-to-head comparison vs Stage 07 baseline
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

from experiments.stage_08.model import Stage08OptimizedDomainClassifier

STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
MODELS_DIR = STAGE_DIR / "models"
OUTPUT_DIR = STAGE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODELS_DIR / "best_domain_01_stage08.pt"
DEV_DATA_PATH = DATA_DIR / "dev_5channels.pt"
STAGE07_RESULTS = ROOT_DIR / "experiments" / "stage_07" / "output" / "domain_01_results.json"

CHANNELS = ["query", "thought", "context_actions", "observations", "proposed_action"]


def run_evaluation(device_str: str = "cpu") -> Dict[str, Any]:
    print("=" * 70)
    print("Stage 08: Domain 01 (5-Channel Cross-Attention) Evaluation Gateway")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Run train.py first.")
    if not DEV_DATA_PATH.exists():
        raise FileNotFoundError(f"Dev data not found at {DEV_DATA_PATH}. Run prepare_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    proto_keys = checkpoint["prototype_keys"].to(device)

    model = Stage08OptimizedDomainClassifier(
        initial_prototype_keys=proto_keys,
        num_pos_prototypes=4,
        embed_dim=384,
        hidden_dim=384,
        num_channels=5,
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dev_data = torch.load(DEV_DATA_PATH, weights_only=False)
    channels_tensor = dev_data["channels_tensor"].to(device)  # (143, 5, 384)
    labels = dev_data["labels"].numpy().astype(int)            # (143,)
    example_ids = dev_data["example_ids"]
    severities = dev_data["severities"]
    domains_list = dev_data["domains_list"]
    total = len(labels)

    with torch.no_grad():
        probs_t, logits_t, meta_t = model(channels_tensor)

    probs = probs_t.cpu().numpy()
    preds = (probs >= 0.50).astype(int)
    pool_w = meta_t["channel_pool_weights"].cpu().numpy()  # (143, 5)

    acc = accuracy_score(labels, preds)
    p_safe, r_safe, f1_safe, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary", zero_division=0)
    p_harm, r_harm, f1_harm, _ = precision_recall_fscore_support(labels, preds, pos_label=0, average="binary", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(labels, preds, labels=[1, 0])
    tp, fn = cm[0][0], cm[0][1]
    fp, tn = cm[1][0], cm[1][1]

    mean_ch_weights = {
        CHANNELS[i]: float(pool_w[:, i].mean())
        for i in range(5)
    }

    confidences = np.where(preds == 1, probs, 1.0 - probs)
    high_mask = confidences >= 0.70
    high_acc = float(accuracy_score(labels[high_mask], preds[high_mask])) if high_mask.sum() > 0 else 0.0

    print(f"\nHeld-Out SemEval Dev Evaluation ({total} samples):")
    print(f"  Exact Accuracy:     {acc * 100:.2f}%")
    print(f"  Macro F1:           {f1_macro:.4f}")
    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  Safe (1) F1:        {f1_safe:.4f} (Prec: {p_safe:.4f}, Rec: {r_safe:.4f})")
    print(f"  Harmful (0) F1:     {f1_harm:.4f} (Prec: {p_harm:.4f}, Rec: {r_harm:.4f})")
    print(f"  High-Conf (>=70%):  {high_acc * 100:.2f}% ({high_mask.sum()}/{total} samples)")
    print(f"\nMean 5-Channel Attention Distribution:")
    for k, v in mean_ch_weights.items():
        print(f"  {k:18s}: {v * 100:.2f}%")

    # Load Stage 07 results if available for comparative table
    stage07_acc = 0.0
    stage07_f1 = 0.0
    stage07_auc = 0.0
    if STAGE07_RESULTS.exists():
        with open(STAGE07_RESULTS, "r", encoding="utf-8") as f:
            s7 = json.load(f)
            stage07_acc = s7["metrics"]["accuracy"]
            stage07_f1 = s7["metrics"]["macro_f1"]
            stage07_auc = s7["metrics"]["roc_auc"]

    # Build 20 sample trace
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
            "proposed_action_wt": float(pool_w[i, 4]),
            "obs_wt": float(pool_w[i, 3])
        })

    results = {
        "dataset": "SemEval-2027 AgentRisk Dev Split",
        "domain": "Domain 01: No Material Harm (Stage 08 Optimized)",
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
        "mean_channel_weights": mean_ch_weights,
        "confusion_matrix": {
            "tp_safe": int(tp),
            "fn_safe": int(fn),
            "fp_harm": int(fp),
            "tn_harm": int(tn)
        },
        "comparison_vs_stage07": {
            "stage07_accuracy": stage07_acc,
            "stage08_accuracy": float(acc),
            "acc_delta": float(acc - stage07_acc),
            "stage07_macro_f1": stage07_f1,
            "stage08_macro_f1": float(f1_macro),
            "f1_delta": float(f1_macro - stage07_f1),
            "stage07_auc": stage07_auc,
            "stage08_auc": float(auc),
            "auc_delta": float(auc - stage07_auc)
        },
        "sample_traces": trace_samples
    }

    with open(OUTPUT_DIR / "domain_01_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_md = f"""# Stage 08 Benchmark Report: 5-Channel Cross-Attentive Classifier (Domain 01)
**Architecture**: 5-Channel Disentanglement + Self-Attention Bridge + Prototype Cross-Attention  
**Dataset**: SemEval AgentRisk Held-Out Dev Set ({total} samples)

---

## 1. Executive Performance Summary & Stage 07 Comparison

| Metric | Stage 07 Baseline | Stage 08 (Optimized) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Accuracy** | **{stage07_acc * 100:.2f}%** | **{acc * 100:.2f}%** | **{'+' if acc >= stage07_acc else ''}{(acc - stage07_acc)*100:+.2f}%** |
| **Macro F1-Score** | **{stage07_f1:.4f}** | **{f1_macro:.4f}** | **{'+' if f1_macro >= stage07_f1 else ''}{(f1_macro - stage07_f1):+.4f}** |
| **ROC-AUC** | **{stage07_auc:.4f}** | **{auc:.4f}** | **{'+' if auc >= stage07_auc else ''}{(auc - stage07_auc):+.4f}** |
| **Safe (1) F1-Score** | 0.8281 | **{f1_safe:.4f}** | {'+' if f1_safe >= 0.8281 else ''}{(f1_safe - 0.8281):+.4f} |
| **Harmful (0) F1-Score** | 0.8608 | **{f1_harm:.4f}** | {'+' if f1_harm >= 0.8608 else ''}{(f1_harm - 0.8608):+.4f} |
| **High-Confidence (>=70%) Accuracy** | 84.62% | **{high_acc * 100:.2f}%** | {'+' if high_acc >= 0.8462 else ''}{(high_acc - 0.8462)*100:+.2f}% |

---

## 2. 5-Channel Attention Allocation

The multi-head pooling network dynamically allocated the following attention across the 5 interaction channels:

| Semantic Channel | Mean Attention Weight | Key Diagnostic Role |
| :--- | :---: | :--- |
| **`proposed_action`** | **{mean_ch_weights['proposed_action']*100:.1f}%** | Next action about to be executed (highest safety vector) |
| **`observations`** | **{mean_ch_weights['observations']*100:.1f}%** | Environmental output, leaked credentials, file dumps |
| **`context_actions`** | **{mean_ch_weights['context_actions']*100:.1f}%** | Historical diagnostic tools & commands previously run |
| **`thought`** | **{mean_ch_weights['thought']*100:.1f}%** | Internal agent deliberation & planning traces |
| **`query`** | **{mean_ch_weights['query']*100:.1f}%** | User prompt context and initial request |

---

## 3. Confusion Matrix

| True \\ Predicted | Predicted Safe (1) | Predicted Harmful (0) | Total | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **True Safe (No Material Harm)** | **{tp}** (TP) | {fn} (FN) | {tp+fn} | {r_safe*100:.1f}% |
| **True Harmful (Material Harm Occurs)** | {fp} (FP) | **{tn}** (TN) | {fp+tn} | {r_harm*100:.1f}% |

---

## 4. Sample Prediction Trace (First 20 Held-Out Dev Samples)

| ID | True Class | Pred Class | Safe Prob | Conf. | Severity | Correct | Proposed Action Wt | Obs Wt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for t in trace_samples:
        corr_mark = "PASS" if t["correct"] else "FAIL"
        report_md += (
            f"| `{t['example_id']}` | {t['true_label']} | {t['pred_label']} | "
            f"{t['safe_prob']:.3f} | {t['confidence']*100:.1f}% | {t['risk_severity']} | "
            f"**{corr_mark}** | {t['proposed_action_wt']*100:.1f}% | {t['obs_wt']*100:.1f}% |\n"
        )

    with open(OUTPUT_DIR / "domain_01_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nReport generated at {OUTPUT_DIR / 'domain_01_report.md'}")
    return results


def main():
    run_evaluation()


if __name__ == "__main__":
    main()

