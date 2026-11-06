# Stage 02 Evaluation Gateway Report: Domain-Aware Attention & Similarity

**Date**: November 2026  
**Dataset**: SemEval 2027 AgentRisk (568 Train, 143 Dev)  
**Evaluator**: `experiments/stage_02/main.py`  

---

## 1. Gateway Leaderboard

| Rank | Model Architecture | Feature Representation | QWK Score | Accuracy | Macro F1 | Pearson r | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Domain-Augmented XGBoost (389-d)** | Hybrid (384-d + 5 Domain) | **`0.2639`** | 39.86% | `0.2393` | `+0.2186` | `1.74 ms` |
| **2** | **SVR RBF (Raw 384-d)** | Raw Embeddings (384-d) | **`0.2419`** | 32.17% | `0.2165` | `+0.2719` | `12.67 ms` |
| **3** | **Domain-Augmented SVR (389-d)** | Hybrid (384-d + 5 Domain) | **`0.2364`** | 39.86% | `0.2309` | `+0.2630` | `14.35 ms` |
| **4** | **Domain-Augmented Ridge (389-d)** | Hybrid (384-d + 5 Domain) | **`0.2196`** | 33.57% | `0.1984` | `+0.2161` | `0.30 ms` |

---

## 2. Key Findings & Domain Feature Impact

1. **Domain-Augmented XGBoost Regressor Ranked #1 (`QWK = 0.2639`)**:
   - The addition of explicit prototype domain cosine similarities (`sim_emergency_risk`, `sim_authorization_access_risk`, etc.) provides high-level semantic anchor points that tree splitters can immediately branch on.
   - Boosted QWK from the Stage 01 baseline of **`0.1261`** to **`0.2639`** (a **+109.3% gain**).

2. **Continuous Loss Formulation Alignment**:
   - Transitioning from 5-class cross-entropy to continuous MSE regression directly aligns the learning gradients with the distance-based Quadratic Weighted Kappa penalty.

3. **Nelder-Mead Threshold Calibration**:
   - Adjusting decision cutoffs based on the cumulative empirical distribution restores detection sensitivity for rare severity classes (Severity Level 2 and Level 5).
