# Stage 05: Domain 02 ('Privacy / Confidentiality') Classifier Report

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
  - `query_proj`: 384 $\rightarrow$ 64 $\rightarrow$ LayerNorm $\rightarrow$ GELU $\rightarrow$ Dropout
  - `key_proj`: 384 $\rightarrow$ 64 $\rightarrow$ LayerNorm
  - `fusion_head`: Attentively combines multi-prototype similarities with latent query features.
  - Temperature scaling ($\gamma = 10.0$).

---

## 2. Benchmark Results on Held-Out Dev Set (146 Instances)

| Evaluation Mode | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Default Cutoff (0.50)** | `0.50` | **78.08%** | 0.6379 | 0.7708 | **0.6981** | **0.8027** | `TN=77, FP=21 / FN=11, TP=37` |
| **Optimal F1 Cutoff** | `0.220` | **78.08%** | 0.6379 | 0.7708 | **0.6981** | 0.8027 | `TN=77, FP=21 / FN=11, TP=37` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0048`
- **Maximum Probability**: `0.9931`
- **Mean Probability**   : `0.3961`
