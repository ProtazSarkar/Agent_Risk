# Stage 05 Benchmark Report: domain_11 ('Information Quality')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_11: 'Information Quality'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: 147 (5 Positive, 142 Negative)
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
| **Default Cutoff (0.50)** | `0.50` | **97.96%** | 1.0000 | 0.4000 | **0.5714** | **0.8056** | `TN=142, FP=0 / FN=3, TP=2` |
| **Optimal F1 Cutoff** | `0.200` | **97.96%** | 0.7500 | 0.6000 | **0.6667** | 0.8056 | `TN=141, FP=1 / FN=2, TP=3` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0102`
- **Maximum Probability**: `0.9804`
- **Mean Probability**   : `0.0309`
- **Std Deviation**     : `0.1198`
