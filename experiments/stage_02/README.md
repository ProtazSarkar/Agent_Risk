# Stage 02: Domain-Aware Attention & Similarity-Augmented ML Pipeline

## Overview
Stage 02 implements a domain-guided hybrid learning pipeline for the **SemEval 2027 AgentRisk** shared task. It overcomes the bottlenecks of raw embedding classification by explicitly injecting semantic alignment scores against 5 prototype risk domains.

```
       [Raw User Interaction Text]
                    │
                    ▼
  [SentenceTransformer: all-MiniLM-L6-v2]
                    │
                    ▼
          Dense Query Vector Q (384-d)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  Raw Semantic Space    [Domain Attention] ───> Dot-product with Static Key Matrix K (5, 384)
       (384-d)                 │                (Emergency, Auth, Privacy, Policy, Safe)
         │                     ▼
         │            5 Domain Similarity Scores S (5-d)
         └──────────┬──────────┘
                    ▼
     Hybrid Feature Vector (389-d) = [Q (384-d) || S (5-d)]
                    │
                    ▼
     [Non-linear Regressor: SVR / XGBoost / Ridge] (MSE / Huber loss with balanced sample weights)
                    │
                    ▼
           Continuous Risk Score $\hat{y} \in [1.0, 5.0]$
                    │
                    ▼
     [Nelder-Mead Threshold Optimizer] (Directly maximizes Quadratic Weighted Kappa)
                    │
                    ▼
         Discrete Ordinal Risk Severity Level $\hat{k} \in \{1, 2, 3, 4, 5\}$
```

---

## Component Files
| File | Role | Description |
| :--- | :--- | :--- |
| [`label.py`](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_02/label.py) | **Domain Prototypes & Keys** | Defines 5 risk prototype sentences, computes static normalized key matrix $K \in \mathbb{R}^{5 \times 384}$. |
| [`attention.py`](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_02/attention.py) | **Similarity & Attention** | Computes cosine similarity $Q \times K^T$, softmax attention weights, and concatenates $384 + 5 = 389$ dimensions. |
| [`processdata.py`](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_02/processdata.py) | **Dataset Preprocessing** | Loads data from root `data/agentrisk-data-2027/data/` and builds `train_domain_augmented.csv` & `dev_domain_augmented.csv`. |
| [`similarity_train.py`](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_02/similarity_train.py) | **Model Training** | Trains SVR / Ridge / MLP with balanced sample weights and Nelder-Mead threshold optimization. |
| [`xgboost_train.py`](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_02/xgboost_train.py) | **XGBoost Benchmark** | Tests tree-based regression and classification on the augmented 389-d space. |
| [`main.py`](file:///c:/DevelopmentWorks/github/Agent_Risk/experiments/stage_02/main.py) | **Evaluation Gateway** | Central evaluator orchestrating preprocessing, training, comparing models, and generating leaderboards. |

---

## Leaderboard Results (143 Dev Split Instances)
| Rank | Model Architecture | Features | QWK Score | Accuracy | Macro F1 | Pearson $r$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Domain-Augmented XGBoost** | **Hybrid (389-d)** | **`0.2639`** ★ | **39.86%** | **`0.2393`** | **`+0.2186`** |
| **2** | SVR RBF (Raw Baseline) | Raw Embeddings (384-d) | `0.2419` | 32.17% | `0.2165` | `+0.2719` |
| **3** | Domain-Augmented SVR | Hybrid (389-d) | `0.2364` | 39.86% | `0.2309` | `+0.2630` |
| **4** | Domain-Augmented Ridge | Hybrid (389-d) | `0.2196` | 33.57% | `0.1984` | `+0.2161` |
| *Ref* | *Stage 01 Multi-Class XGBoost* | *Raw Embeddings (384-d)* | *`0.1261`* | *41.26%* | *`0.2245`* | *-* |

---

## Execution Commands
```bash
# Run the complete evaluation gateway:
python experiments/stage_02/main.py

# Run individual components:
python experiments/stage_02/label.py
python experiments/stage_02/attention.py
python experiments/stage_02/processdata.py
python experiments/stage_02/similarity_train.py --model svr
python experiments/stage_02/xgboost_train.py --mode regressor
```

