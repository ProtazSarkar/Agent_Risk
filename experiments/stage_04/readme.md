# Stage 04: Domain-Expert Binary Classifiers & Multi-Label Risk Pipeline

## 🎯 Architecture Vision
Stage 04 implements independent domain-expert binary classifiers for the risk domains. Each domain classifier maps prompt query vectors $Q$ against semantic prototype key vectors $K$ to learn calibrated alignment scores.

---

## 📁 Modular Layout (Clean Separation of Concerns)

```text
experiments/stage_04/
├── data/
│   ├── process_dataset.py           # Extracts embeddings-only + domain target labels
│   ├── train_no_material_harm.csv   # 568 rows x 386 columns (emb_0..emb_383 + label)
│   └── dev_no_material_harm.csv     # 143 rows x 386 columns
├── artifacts/
│   └── domain_01_prototype_keys.pt  # 384-d prototype vectors generated via SentenceTransformer
├── models/
│   └── semantic_prototype_classifier.pt # Trained PyTorch checkpoint (query_proj + key_proj + scale)
├── output/
│   ├── domain_01_results.json       # Structured evaluation metrics
│   └── domain_01_report.md          # Comprehensive analysis report
└── domain_01/                       # Domain 01: 'No Material Harm'
    ├── prototypes.py                # Pre-computes & caches text prototype key vectors (no random noise)
    ├── model.py                     # SemanticPrototypeClassifier (query_proj + key_proj + scaled dot product)
    ├── train.py                     # Model training loop, AdamW optimizer, loss, and checkpoint saving
    └── main.py                      # Primary evaluation gateway, metrics calculation, and trace reporting
```

---

## 🚀 How to Run Domain 01

```powershell
# 1. Pre-compute text prototypes from SentenceTransformer (zero random noise)
python experiments/stage_04/domain_01/prototypes.py

# 2. Train the SemanticPrototypeClassifier
python experiments/stage_04/domain_01/train.py

# 3. Run the Evaluation Gateway
python experiments/stage_04/domain_01/main.py
```
