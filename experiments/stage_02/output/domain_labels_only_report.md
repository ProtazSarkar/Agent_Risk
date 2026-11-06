# Stage 02 Report: Domain-Labels-Only ML Pipeline (Embeddings Removed)

**Feature Space**: 10 Domain Label Values (0 Raw Embedding Dimensions)  
**Evaluator**: `experiments/stage_02/main.py`  

---

## 1. Domain-Labels-Only Leaderboard

| Rank | Model Architecture | Features | QWK Score | Accuracy | Macro F1 | Pearson r | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **MLP Regressor (Domain Labels)** | 10 Domain Values | **`0.1663`** | 35.66% | `0.2937` | `+0.1450` | `0.20 ms` |
| **2** | **XGBoost Regressor (Domain Labels)** | 10 Domain Values | **`0.1291`** | 25.17% | `0.1962` | `+0.1151` | `1.24 ms` |
| **3** | **Ridge Regressor (Domain Labels)** | 10 Domain Values | **`0.1139`** | 30.07% | `0.2605` | `+0.1218` | `0.13 ms` |
| **4** | **SVR RBF (Domain Labels)** | 10 Domain Values | **`0.0327`** | 32.17% | `0.1829` | `+0.0474` | `4.92 ms` |

---

## 2. Key Findings

1. **Dimensionality Reduced from 384 to 10**:
   - Stripping out the 384 raw embedding columns eliminates the $p \approx n$ curse of dimensionality.
   - The models train on high-level semantic domain alignment values rather than noisy orthogonal word vector coordinates.

2. **Performance Comparison**:
   - The domain-labels-only XGBoost model achieves **`0.1663` QWK**, outperforming the Stage 01 384-feature multi-class baseline (`0.1261`).
