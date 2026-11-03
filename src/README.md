# `src/` - Source Code Directory

## 🎯 Purpose
The `src/` directory contains all production code, reusable modules, and experimental agent implementations for the Agentic Risk Analysis framework. It uses a clean, standard Python packaging layout.

---

## 📁 Directory Content
- **[`agent_risk/`](agent_risk/)**: Production framework library containing:
  - `planner/`: Task decomposition & multi-route DAG generation (API + Local Llama LoRA).
  - `risk_engine/`: Psycholinguistic feature extraction, XGBoost risk shield, and necessity filter.
  - `decision_theory/`: Markov state transition modeling and Stackelberg security games.
  - `data/`: Ingestion, schema adapter, and conversion for SemEval 2027 (`AgentRisk/agentrisk-data-2027`).
  - `utils/`: Telemetry profiling (CPU, RAM, latency) and statistical evaluation metrics.
  - `config.py`: Centralized environment variables, directory paths, and safeguard thresholds.
- **[`dev/`](dev/)**: Prototyping workspace for designing custom experimental AI agents, multi-agent pipelines, and application-specific monitors.

---

## 📦 Installation
To install the `src/` packages in editable development mode:
```bash
pip install -e .
```
