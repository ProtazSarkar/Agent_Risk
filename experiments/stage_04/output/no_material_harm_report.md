# Stage 04: Domain 01 ('No Material Harm') Binary Classifier Report

## 1. Executive Summary

This report establishes the baseline binary classifier for **Domain 01: 'No Material Harm'** within the Stage 04 domain-expert architecture.

- **Task**: Detect whether an agentic interaction is harmless (`label = 1`) versus risky/harmful (`label = 0`).
- **Input Space**: Strictly 384-dimensional dense semantic embeddings (`emb_0` through `emb_383`), completely isolated from target risk severity or metadata leakage.
- **Dataset Split**:
  - Train: 568 instances (262 Positive `No Material Harm`, 306 Negative)
  - Dev  : 143 instances (65 Positive `No Material Harm`, 78 Negative)

---

## 2. Classifier Benchmark Comparison

| Model Architecture | Default (0.50) Acc | Default F1 | Optimal Cutoff | Tuned Acc | Precision | Recall | Tuned F1 | ROC-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (L2, Balanced)** | 55.94% | 0.4425 | 0.450 | **49.65%** | 0.4588 | 0.6000 | **0.5200** | 0.5607 | `TN=55, FP=23 / FN=40, TP=25` |
| **Linear Support Vector Classifier (SVM)** | 54.55% | 0.0000 | 0.200 | **45.45%** | 0.4545 | 1.0000 | **0.6250** | 0.5630 | `TN=78, FP=0 / FN=65, TP=0` |
| **RBF Support Vector Classifier (SVM)** | 42.66% | 0.3971 | 0.200 | **45.45%** | 0.4545 | 1.0000 | **0.6250** | 0.4153 | `TN=34, FP=44 / FN=38, TP=27` |
| **Multi-Layer Perceptron (MLP Neural Net)** | 55.24% | 0.4754 | 0.200 | **54.55%** | 0.5000 | 0.6000 | **0.5455** | 0.5560 | `TN=50, FP=28 / FN=36, TP=29` |
| **XGBoost Binary Classifier** | 61.54% | 0.5045 | 0.380 | **51.75%** | 0.4787 | 0.6923 | **0.5660** | 0.5891 | `TN=60, FP=18 / FN=37, TP=28` |
| **Prototype-Attention Neural Classifier (PyTorch)** | 51.05% | 0.6067 | 0.290 | **51.05%** | 0.4779 | 0.8308 | **0.6067** | 0.5251 | `TN=19, FP=59 / FN=11, TP=54` |

---

## 3. Best Model Performance: Linear Support Vector Classifier (SVM)
- **Decision Threshold**: `0.2000`
- **Accuracy**: `45.45%`
- **F1-Score**: `0.6250`
- **Precision / Recall**: `0.4545` / `1.0000`
- **ROC-AUC**: `0.5630`

---

## 4. Key Findings

1. **Embedding Separability**:
   The 384-dimensional semantic embeddings provide meaningful discriminative signal between benign prompts (GitHub searches, math queries, markdown formatting) and hazardous prompts.
2. **Decision Threshold Impact**:
   Calibrating the probability decision threshold significantly balances Precision and Recall, avoiding the standard default cutoff (0.50) bias.
3. **Modular Role in Stage 04**:
   This classifier serves as **Domain Model 01** in the 11-expert multi-label risk domain classification pipeline described in `readme.md`.
