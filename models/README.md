# `models/` - Serialized Model Checkpoints

## 🎯 Purpose
The `models/` directory stores serialized model artifacts, tabular classifier checkpoints, and fine-tuned LoRA adapter weights. Heavy binary files are git-ignored to keep the repository lightweight.

---

## 📁 Directory Content
- **`xgboost/`**: Serialized tabular risk classifiers (`risk_classifier.json` / `.bin`). Loaded at runtime by `XGBoostShield`.
- **`lora_adapters/`**: Checkpoints and adapter weights for fine-tuned open-source models (e.g., Llama 3 8B fine-tuned for structured DAG planning).

---

## 🔄 Bootstrapping Models
If no serialized model exists on disk, `XGBoostShield` automatically initializes a lightweight baseline heuristic classifier so pipelines and unit tests never fail due to missing model weights. To train a fresh model on your data:
```bash
python experiments/stage_01/02_train_xgboost.py
```

