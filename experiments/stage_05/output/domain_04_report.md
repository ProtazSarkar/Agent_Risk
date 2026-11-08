# Stage 05 Benchmark Report: domain_04 ('Financial / Assets')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **domain_04: 'Financial / Assets'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: 146 (16 Positive, 130 Negative)
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
| **Default Cutoff (0.50)** | `0.50` | **86.30%** | 0.4167 | 0.6250 | **0.5000** | **0.7942** | `TN=116, FP=14 / FN=6, TP=10` |
| **Optimal F1 Cutoff** | `0.800` | **89.04%** | 0.5000 | 0.4375 | **0.4667** | 0.7942 | `TN=123, FP=7 / FN=9, TP=7` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `0.0725`
- **Maximum Probability**: `0.8772`
- **Mean Probability**   : `0.2049`
- **Std Deviation**     : `0.2568`
