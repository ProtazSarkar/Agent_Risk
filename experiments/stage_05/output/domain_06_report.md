# Stage 05 Benchmark Report: domain_06 ('Physical / Health')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_06: 'Physical / Health'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: 145 (14 Positive, 131 Negative)
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
| **Default Cutoff (0.50)** | `0.50` | **91.72%** | 0.5625 | 0.6429 | **0.6000** | **0.8059** | `TN=124, FP=7 / FN=5, TP=9` |
| **Optimal F1 Cutoff** | `0.200` | **91.72%** | 0.5625 | 0.6429 | **0.6000** | 0.8059 | `TN=124, FP=7 / FN=5, TP=9` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0032`
- **Maximum Probability**: `0.9963`
- **Mean Probability**   : `0.1143`
- **Std Deviation**     : `0.3077`
