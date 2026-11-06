# `src/agent_risk/` - Core Framework Package

## 🎯 Purpose
`agent_risk` is the production Python package for the repository. It is namespaced to prevent top-level import conflicts and provides fully typed (PEP 561) reusable components for agent task decomposition, threat detection, and decision-theoretic safeguards.

---

## 📁 Submodules & Architecture

```text
agent_risk/
├── __init__.py          # Exposes package version and key exports
├── py.typed             # PEP 561 marker for static type checking
├── config.py            # Pydantic BaseSettings loading .env, seed, paths, thresholds
├── planner/             # Task decomposition & multi-route DAG generation
│   ├── schema.py        # Pydantic TaskNode, Route, PlanDAG + NetworkX conversion
│   ├── api_planner.py   # Groq / OpenAI LLM API interface with deterministic mock
│   └── local_planner.py # Hugging Face transformers + PEFT LoRA local inference
├── risk_engine/         # Threat detection and shielding
│   ├── feature_extractor.py # Psycholinguistic markers + graph topological metrics
│   ├── xgboost_shield.py    # Tabular risk classifier predicting SAFE/WARN/BLOCKED
│   └── necessity_filter.py  # Reachability & dead-branch attack surface pruner
├── decision_theory/     # Advanced safety modeling
│   ├── markov_chain.py  # MDP trajectory compromise probability modeling
│   └── game_theory.py   # Stackelberg leader-follower equilibrium solver
├── data/                # Data ingestion & normalization
│   └── loader.py        # Adapter for SemEval 2027 AgentRisk/agentrisk-data-2027
└── utils/               # Telemetry and benchmarks
    ├── metrics.py       # Context manager measuring latency (ms), CPU (%), RAM (MB)
    └── evaluator.py     # Accuracy, Precision, Recall, F1, ROC-AUC calculator
```

---

## 💻 Import Examples
```python
from agent_risk.planner import APIPlanner
from agent_risk.risk_engine import FeatureExtractor, XGBoostShield, NecessityFilter
from agent_risk.decision_theory import MarkovRiskModel, StackelbergShield
from agent_risk.data import AgentRiskDatasetLoader
```

