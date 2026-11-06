# Stage 05: Context Distillation & Zero-Truncation Semantic Classifier

## 🎯 Architecture Vision & Breakthrough

Stage 05 solves the **100% action truncation defect** discovered in previous stages:
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

## 📁 Modular Layout (Clean Separation of Concerns)

```text
experiments/stage_05/
├── README.md
├── data/
│   ├── process_summarized_dataset.py       # Distills interactions, caches summaries, and encodes 384-d vectors
│   ├── train_summarized_embedded.csv       # 568 rows x 389 columns (emb_0..emb_383 + metadata + label)
│   ├── train_summarized_embedded.parquet
│   ├── dev_summarized_embedded.csv         # 143 rows x 389 columns
│   ├── dev_summarized_embedded.parquet
│   └── cache/
│       └── summaries_cache.json            # Persistent JSON cache for all 711 distilled summaries
├── artifacts/
│   └── domain_01_prototype_keys.pt         # 384-d multi-prototype keys (4 Harm, 4 Safe)
├── models/
│   └── semantic_prototype_classifier.pt    # Trained PyTorch checkpoint (query_proj + key_proj + fusion)
├── output/
│   ├── domain_01_results.json              # Structured evaluation metrics
│   └── domain_01_report.md                 # Detailed report
└── domain_01/                              # Domain 01: 'No Material Harm'
    ├── prototypes.py                       # Loads pre-computed prototype key vectors
    ├── model.py                            # SemanticPrototypeClassifier
    ├── train.py                            # Training pipeline (AdamW, BCEWithLogitsLoss)
    └── main.py                             # Evaluation gateway & trace runner
```

---

## 📊 Benchmark Comparison: Stage 04 vs Stage 05

| Metric | Stage 04 (Truncated Embeddings) | Stage 05 (Zero-Truncation Summarized) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy (Default 0.50)** | **53.85%** | **82.52%** | **+28.67%** |
| **F1-Score** | **0.4762** | **0.8062** | **+0.3300** |
| **Precision** | 49.18% | **81.25%** | **+32.07%** |
| **Recall on Safe Cases** | 46.15% (30/65) | **80.00% (52/65)** | **+33.85% (22 more caught)** |
| **ROC-AUC** | **0.5572** | **0.8714** | **+0.3142** |
| **`AGENTRISK-019` Pred** | $0.0533$ (Harmful 0 - MISS) | **$0.9953$ (No Harm 1 - CORRECT)** | **Correctly Resolved** |
| **Action Truncation Rate** | 100.0% (Action dropped) | **0.0% (Zero Truncation)** | **100% Fixed** |

---

## 🚀 How to Run Stage 05

```powershell
# 1. Generate summarized dataset and 384-d embeddings (using Groq / distillation cache)
python experiments/stage_05/data/process_summarized_dataset.py

# 2. Train the SemanticPrototypeClassifier
python experiments/stage_05/domain_01/train.py

# 3. Run the Evaluation Gateway
python experiments/stage_05/domain_01/main.py
```

