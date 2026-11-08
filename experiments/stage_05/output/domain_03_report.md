# Stage 05 Benchmark Report: domain_03 ('Security / System Integrity')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_03: 'Security / System Integrity'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: 146 (19 Positive, 127 Negative)
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
| **Default Cutoff (0.50)** | `0.50` | **89.73%** | 0.5909 | 0.6842 | **0.6341** | **0.8417** | `TN=118, FP=9 / FN=6, TP=13` |
| **Optimal F1 Cutoff** | `0.200` | **86.99%** | 0.5000 | 0.6842 | **0.5778** | 0.8417 | `TN=114, FP=13 / FN=6, TP=13` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0030`
- **Maximum Probability**: `0.9963`
- **Mean Probability**   : `0.1571`
- **Std Deviation**     : `0.3406`
