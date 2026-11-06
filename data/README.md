# `data/` - Datasets & Local Storage

## 🎯 Purpose
The `data/` directory is reserved for local datasets, task traces, and downloaded benchmark suites (such as `AgentRisk/agentrisk-data-2027`).

> **Note on Version Control:**
> All datasets and data artifacts inside this directory are **strictly ignored by Git** (`.gitignore`) to prevent heavy data binaries or raw benchmark files from being committed to GitHub. Only this documentation file and directory placeholder are tracked.

---

## 📁 Directory Content (Local Only)
- **`agentrisk-data-2027/`**: Local clone or cache of the official SemEval 2027 dataset. (Git-ignored)
- **`raw/`**: Raw prompt datasets and downloaded benchmarks. (Git-ignored)
- **`processed/`**: Extracted task DAGs and psycholinguistic marker features ready for model training. (Git-ignored)
- **`ground_truth/`**: Hand-annotated risk labels, harm domains, and necessity ground truths. (Git-ignored)
- **`sample/`**: Local sample prompts and sample DAGs for testing. (Git-ignored)

---

## 📥 Ingesting SemEval 2027 Data
To download and ingest the official SemEval 2027 dataset into `data/`:
```bash
python scripts/fetch_semeval_data.py --dataset AgentRisk/agentrisk-data-2027 --split train
```

Or load directly in Python via the Hugging Face `datasets` library:
```python
from datasets import load_dataset
ds = load_dataset("AgentRisk/agentrisk-data-2027")
```
