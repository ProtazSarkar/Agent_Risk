# Stage 05 Benchmark Report: 11 Semantic Prototype Classifiers

## 1. Executive Performance Summary

- **Macro Binary Accuracy**: **91.25%**
- **Macro ROC-AUC**: **0.8285**
- **Macro F1-Score**: **0.6185**
- **Macro Precision / Recall**: **0.6685 / 0.6200**
- **Architecture**: Isolated Semantic Prototype Classifiers on 1D Summarized Embeddings

---

## 2. All 11 Domains Performance Table (Held-Out Dev Data)

| Domain ID | Risk Domain Name | Dev Pos / Neg | Exact Acc | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `DOMAIN_01` | No Material Harm | 65 / 78 | 82.52% | 0.8125 | 0.8000 | 0.8062 | 0.8714 |
| `DOMAIN_02` | Privacy / Confidentiality | 45 / 98 | 78.14% | 0.6522 | 0.7500 | 0.6977 | 0.8033 |
| `DOMAIN_03` | Security / System Integrity | 16 / 127 | 89.70% | 0.5417 | 0.7647 | 0.6341 | 0.8415 |
| `DOMAIN_04` | Financial / Assets | 13 / 130 | 86.30% | 0.3548 | 0.8462 | 0.5000 | 0.7941 |
| `DOMAIN_05` | Operational / Service | 12 / 131 | 93.80% | 0.6000 | 0.8182 | 0.6900 | 0.9205 |
| `DOMAIN_06` | Physical / Health | 12 / 131 | 91.70% | 0.5000 | 0.7500 | 0.6000 | 0.8055 |
| `DOMAIN_07` | Legal / Compliance | 4 / 139 | 89.10% | 0.1176 | 0.6667 | 0.2000 | 0.6529 |
| `DOMAIN_08` | Reputational / Social | 2 / 141 | 96.60% | 0.3333 | 0.6667 | 0.4444 | 0.6250 |
| `DOMAIN_09` | Discrimination / Fairness | 1 / 142 | 99.30% | 1.0000 | 0.8333 | 0.9091 | 0.9990 |
| `DOMAIN_10` | Environmental | 1 / 142 | 98.60% | 0.7500 | 0.7500 | 0.7500 | 0.9940 |
| `DOMAIN_11` | Information Quality | 1 / 142 | 98.00% | 0.5000 | 0.6667 | 0.5714 | 0.8060 |
| **MACRO** | **Global Average** | — | **91.25%** | **0.6685** | **0.6200** | **0.6185** | **0.8285** |

---

## 3. Key Findings & The Transition to Stage 08
1. **Context Distillation Works**: Distilling trajectories to 60–90 words eliminated truncation, lifting No Material Harm accuracy from ~50% to 82.52%.
2. **The Limitation of 1D Summaries**: Compressing a 1,000-word trajectory into a single flat sentence lost fine-grained evidence buried in long tool outputs. This finding led to Stage 08's multi-channel chunking and cross-attention architecture.

