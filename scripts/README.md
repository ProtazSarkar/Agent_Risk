# `scripts/` - Automation & Ingestion Utilities

## 🎯 Purpose
The `scripts/` directory houses standalone command-line scripts, dataset downloaders, benchmark evaluators, and automation utilities.

---

## 📁 Directory Content
- **`fetch_semeval_data.py`**: Command-line tool to fetch, cache, and convert the official SemEval 2027 `AgentRisk/agentrisk-data-2027` dataset directly into `PlanDAG` graphs stored in `data/processed/semeval_2027_dags.json`.

---

## 🚀 Usage Examples
```bash
# Ingest official training split from Hugging Face
python scripts/fetch_semeval_data.py --dataset AgentRisk/agentrisk-data-2027 --split train

# Run in offline mode (uses cached or high-fidelity synthetic benchmark instances)
python scripts/fetch_semeval_data.py --offline
```

