# Stage 05 Benchmark Report: domain_05 ('Operational / Service')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_05: 'Operational / Service'**.

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
| **Default Cutoff (0.50)** | `0.50` | **93.79%** | 0.6667 | 0.7143 | **0.6897** | **0.9204** | `TN=126, FP=5 / FN=4, TP=10` |
| **Optimal F1 Cutoff** | `0.780` | **92.41%** | 0.6154 | 0.5714 | **0.5926** | 0.9204 | `TN=126, FP=5 / FN=6, TP=8` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0345`
- **Maximum Probability**: `0.9512`
- **Mean Probability**   : `0.1332`
- **Std Deviation**     : `0.2622`
