# Stage 05 Benchmark Report: domain_08 ('Reputational / Social / Psychological')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_08: 'Reputational / Social / Psychological'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: 147 (6 Positive, 141 Negative)
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
| **Default Cutoff (0.50)** | `0.50` | **96.60%** | 0.6667 | 0.3333 | **0.4444** | **0.6253** | `TN=140, FP=1 / FN=4, TP=2` |
| **Optimal F1 Cutoff** | `0.200` | **96.60%** | 0.6667 | 0.3333 | **0.4444** | 0.6253 | `TN=140, FP=1 / FN=4, TP=2` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0206`
- **Maximum Probability**: `0.9542`
- **Mean Probability**   : `0.0357`
- **Std Deviation**     : `0.1010`
