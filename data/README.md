# `data/` - Datasets & Schemas

## 🎯 Purpose
The `data/` directory houses raw input prompts, parsed task DAGs, ground-truth annotations, and turnkey sample datasets. Large artifacts and raw downloads are git-ignored, while directory structures are maintained via `.gitkeep`.

---

## 📁 Directory Content
- **`raw/`**: Raw prompt datasets and downloaded benchmarks (e.g., `data/raw/agentrisk_data_2027/`). Git-ignored except `.gitkeep`.
- **`processed/`**: Extracted task DAGs and psycholinguistic marker features ready for model training (`stage_01_dags.json`, `semeval_2027_dags.json`). Git-ignored except `.gitkeep`.
- **`ground_truth/`**: Hand-annotated risk labels, harm domains, and necessity ground truths. Git-ignored except `.gitkeep`.
- **`sample/`**: Turnkey sample schemas tracked in git so tests and pipelines run out-of-the-box without external downloads:
  - `sample_prompts.json`: 10 balanced benign, edge-case, and adversarial prompts.
  - `sample_dag.json`: Concrete multi-route PlanDAG schema instance.

---

## 📥 Ingesting SemEval 2027 Data
To download and ingest the official SemEval 2027 dataset into `data/`:
```bash
python scripts/fetch_semeval_data.py --dataset AgentRisk/agentrisk-data-2027 --split train
```
