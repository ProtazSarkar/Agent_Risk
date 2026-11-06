# Stage 05: Summarized Dataset & Zero-Truncation Semantic Classifier Report

## 1. Executive Summary

This report evaluates **Stage 05**, which completely eliminates the **100% action truncation defect** of Stage 01/04 through **Semantic Context Distillation**.

- **Upstream Problem Resolved**: Previous interaction strings spanned $400$ to $1,300$ words, cutting off $100\%$ of `proposed_action` at token 256.
- **Stage 05 Solution**: Distilled all 711 rows into dense 60–90 word summaries capturing:
  1. User Goal
  2. Untrusted Environmental Injections / Warnings
  3. Agent's Proposed Action & Safeguard Refusal Decision
- **Zero Truncation**: All summaries are strictly $\le 139$ words in dev and $\le 216$ words in train, completely fitting within the 256-token limit of `all-MiniLM-L6-v2`.
- **Dataset Split**:
  - Train: 568 instances (262 Positive, 306 Negative)
  - Dev  : 143 instances (65 Positive, 78 Negative)

---

## 2. Benchmark Comparison: Stage 04 vs Stage 05

| Metric | Stage 04 (Truncated Embeddings) | Stage 05 (Summarized, Non-Truncated) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy (Cutoff 0.50)** | **53.85%** | **82.52%** | **+28.67%** |
| **F1-Score (Cutoff 0.50)** | **0.4762** | **0.8062** | **+0.3300** |
| **ROC-AUC** | **0.5572** | **0.8714** | **+0.3142** |
| **Recall (Tuned Cutoff)** | **60.00%** | **81.54%** | **+21.54%** |
| **Action Truncation Rate** | 100.0% (Action dropped) | **0.0% (Zero Truncation)** | **100% Resolved** |

---

## 3. Confusion Matrix Breakdown (Dev Set: 143 Instances)

- **Default Cutoff (0.50)**:
  - True Negatives (TN): `66`
  - False Positives (FP): `12`
  - False Negatives (FN): `13`
  - True Positives (TP): `52`
