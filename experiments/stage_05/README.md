# Stage 05: Context Distillation & 11-Domain Zero-Truncation Semantic Classifier Suite

## 🎯 Architecture Vision & Breakthrough

Stage 05 solves the **100% action truncation defect** discovered in previous stages and scales to all **11 SemEval AgentRisk domains** in exact canonical order:
- In Stage 01/04, raw interaction histories spanned **400 to 1,300 words**.
- Because `all-MiniLM-L6-v2` has a hard cutoff at **256 tokens**, appending `[PROPOSED_ACTION]` to the end caused **100.0% of all samples (568/568 in train, 143/143 in dev) to truncate away the agent's proposed action and safe refusal**.
- Stage 05 implements **Context Distillation**:
  - Each sample is distilled into a compact, information-dense **60–80 word semantic summary** ($\approx 90$ tokens) combining:
    1. **User Request / Goal**
    2. **Untrusted Environmental Injections & Threat Observations**
    3. **Agent's Proposed Action & Explicit Safeguard Refusal Decision**
  - All summaries are strictly $\le 139$ words in dev and $\le 216$ words in train (**Zero Truncation**).
  - SentenceTransformer (`all-MiniLM-L6-v2`) encodes these summaries into dense 384-dimensional vectors where 100% of the action and refusal semantics are preserved.

---

## 📁 Modular Layout (11 Isolated Domain Modules)

```text
experiments/stage_05/
├── README.md
├── run_all_domains.py                      # Master multi-domain training & evaluation benchmark runner
├── scaffold_domains.py                     # Scaffolding generator for isolated domain modules
├── data/
│   ├── process_summarized_dataset.py       # Distills interactions, caches summaries, and encodes 384-d vectors
│   ├── domain_augmentations.py             # SemEval-grounded synthetic examples balancing severities 1 to 5
│   ├── generate_domain_datasets.py         # Multi-domain balanced dataset generator
│   ├── dataset_domains_summary.csv         # Complete train/dev count breakdown per domain and severity
│   ├── train_domain_01.csv .. train_domain_11.csv
│   └── dev_domain_01.csv .. dev_domain_11.csv
├── artifacts/
│   ├── generate_all_prototypes.py          # Master prototype key vector generator
│   ├── domain_01_prototype_keys.pt .. domain_11_prototype_keys.pt  # Cached 384-d text prototype tensors
├── models/
│   ├── domain_01_classifier.pt .. domain_11_classifier.pt          # Trained PyTorch model checkpoints
├── output/
│   ├── all_domains_benchmark.json          # Master 11-domain evaluation metrics
│   ├── all_domains_benchmark.md            # Comprehensive multi-domain benchmark report
│   ├── domain_01_results.json .. domain_11_results.json
│   └── domain_01_report.md .. domain_11_report.md
└── domain_01/ to domain_11/                # Completely isolated domain packages
    ├── prototypes.py                       # Instant loader for pre-computed prototype key vectors
    ├── model.py                            # SemanticPrototypeClassifier (query_proj + key_proj + fusion)
    ├── train.py                            # Domain training pipeline (AdamW, BCEWithLogitsLoss, dev checkpointing)
    └── main.py                             # Domain evaluation gateway & prediction trace runner
```

---

## 📊 Global 11-Domain Benchmark Results (Held-Out Dev Set)

| Folder | Domain Name | Dev (Tot/Pos) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Dynamic Range | Confusion Matrix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `domain_01` | **No Material Harm** | 143 / 65 | **82.52%** | 0.8125 | 0.8000 | **0.8062** | **0.8714** | `[0.004, 0.996]` | `TN=66, FP=12 / FN=13, TP=52` |
| `domain_02` | **Privacy / Confidentiality** | 146 / 48 | **78.08%** | 0.6379 | 0.7708 | **0.6981** | **0.8027** | `[0.005, 0.993]` | `TN=77, FP=21 / FN=11, TP=37` |
| `domain_03` | **Security / System Integrity** | 146 / 19 | **89.73%** | 0.5909 | 0.6842 | **0.6341** | **0.8417** | `[0.003, 0.996]` | `TN=118, FP=9 / FN=6, TP=13` |
| `domain_04` | **Financial / Assets** | 146 / 16 | **86.30%** | 0.4167 | 0.6250 | **0.5000** | **0.7942** | `[0.073, 0.877]` | `TN=116, FP=14 / FN=6, TP=10` |
| `domain_05` | **Operational / Service** | 145 / 14 | **93.79%** | 0.6667 | 0.7143 | **0.6897** | **0.9204** | `[0.035, 0.951]` | `TN=126, FP=5 / FN=4, TP=10` |
| `domain_06` | **Physical / Health** | 145 / 14 | **91.72%** | 0.5625 | 0.6429 | **0.6000** | **0.8059** | `[0.003, 0.996]` | `TN=124, FP=7 / FN=5, TP=9` |
| `domain_07` | **Legal / Compliance** | 147 / 8 | **89.12%** | 0.1667 | 0.2500 | **0.2000** | **0.6529** | `[0.071, 0.851]` | `TN=129, FP=10 / FN=6, TP=2` |
| `domain_08` | **Reputational / Social / Psychological** | 147 / 6 | **96.60%** | 0.6667 | 0.3333 | **0.4444** | **0.6253** | `[0.021, 0.954]` | `TN=140, FP=1 / FN=4, TP=2` |
| `domain_09` | **Discrimination / Fairness** | 147 / 5 | **99.32%** | 0.8333 | 1.0000 | **0.9091** | **0.9986** | `[0.067, 0.896]` | `TN=141, FP=1 / FN=0, TP=5` |
| `domain_10` | **Environmental** | 147 / 5 | **98.64%** | 1.0000 | 0.6000 | **0.7500** | **0.9944** | `[0.069, 0.909]` | `TN=142, FP=0 / FN=2, TP=3` |
| `domain_11` | **Information Quality** | 147 / 5 | **97.96%** | 1.0000 | 0.4000 | **0.5714** | **0.8056** | `[0.010, 0.980]` | `TN=142, FP=0 / FN=3, TP=2` |
| **MACRO** | **11-Domain Macro Average** | - | **91.25%** | **0.6685** | **0.6200** | **0.6185** | **0.8285** | - | - |

---

## 🚀 How to Run Stage 05

```powershell
# 1. Generate multi-domain balanced datasets and 384-d embeddings
python experiments/stage_05/data/generate_domain_datasets.py

# 2. Pre-generate and cache prototype keys for all domains (zero random noise)
python experiments/stage_05/artifacts/generate_all_prototypes.py

# 3. Train and evaluate any specific domain independently (e.g. domain_02 Privacy)
python experiments/stage_05/domain_02/train.py
python experiments/stage_05/domain_02/main.py

# 4. Run the Master Multi-Domain Benchmark Suite (all 11 domains)
python experiments/stage_05/run_all_domains.py
```
