# Stage 05 Benchmark Report: domain_07 ('Legal / Compliance')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_07: 'Legal / Compliance'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: 147 (8 Positive, 139 Negative)
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
| **Default Cutoff (0.50)** | `0.50` | **89.12%** | 0.1667 | 0.2500 | **0.2000** | **0.6529** | `TN=129, FP=10 / FN=6, TP=2` |
| **Optimal F1 Cutoff** | `0.790` | **91.16%** | 0.1429 | 0.1250 | **0.1333** | 0.6529 | `TN=133, FP=6 / FN=7, TP=1` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0708`
- **Maximum Probability**: `0.8513`
- **Mean Probability**   : `0.1568`
- **Std Deviation**     : `0.2026`
