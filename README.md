# Agentic Risk Analysis 🛡️

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-green.svg)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/Architecture-DAG%20%2B%20XGBoost%20%2B%20GameTheory-purple.svg)](docs/arch.md)

A modular, production-grade security and reliability framework for **Autonomous LLM Agents**. The framework decomposes complex agent goals into multi-route Directed Acyclic Graphs (Task DAGs), scores task necessity to reduce attack surfaces, extracts psycholinguistic textual markers and topological graph features, and enforces runtime safety using an **XGBoost Risk Shield** and **Stackelberg Game Policies**.

---

## 📑 Repository Structure (Optimized)

```text
Agent_Risk/
├── pyproject.toml              # Modern PEP 517/621 packaging, tool configs (pytest, ruff, black)
├── setup.py                    # Compatibility layer for editable install (pip install -e .)
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development, testing, and linting dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # Comprehensive git-ignore for ML models, datasets & checkpoints
├── LICENSE                     # Apache 2.0 open-source license
├── README.md                   # Project documentation and quickstart
│
├── docs/                       # Research documentation and publication drafts
│   ├── arch.md                 # System architecture details & mathematical formulations
│   ├── semeval_integration.md  # Mapping Psycholinguistic markers to XGBoost features
│   └── reports/                # Experiment writeups, evaluation charts, and benchmarks
│
├── data/                       # Datasets (git-ignored except sample schemas)
│   ├── raw/                    # Raw prompt datasets (Benign, Edge Cases, Adversarial)
│   ├── processed/              # Extracted task DAGs, Psycholinguistic markers
│   ├── ground_truth/           # Annotated risk labels and necessity ground truths
│   └── sample/                 # Immediate runnable sample datasets
│       ├── sample_prompts.json # Sample benign, edge case, and adversarial prompts
│       └── sample_dag.json     # Concrete multi-route PlanDAG schema instance
│
├── models/                     # Saved model artifacts & checkpoints (git-ignored)
│   ├── xgboost/                # Serialized XGBoost risk classifiers (.json / .bin)
│   └── lora_adapters/          # Fine-tuned Llama LoRA weights (e.g., stage_02_lora/)
│
├── scripts/                    # Automation and data ingestion CLI tools
│   ├── README.md               # Script documentation and execution instructions
│   └── fetch_semeval_data.py   # Download & convert AgentRisk/agentrisk-data-2027 into DAGs
│
├── tasks/                      # SemEval 2027 Shared Task evaluation submissions
│   ├── README.md               # Task submission instructions & format guide
│   └── semeval_2027/           # Subtask 1, 2, 3 predictions & benchmarks
│
├── src/                        # Core source code directory
│   ├── README.md               # Source layout and installation guide
│   ├── dev/                    # Prototyping workspace for developing custom AI agents
│   │   ├── README.md           # Guide for building experimental agents
│   │   ├── __init__.py
│   │   └── sample_agent.py     # Base agent and RiskAnalysisAgent prototype
│   └── agent_risk/             # Core production framework package
│       ├── README.md           # Module reference & API guide
│       ├── __init__.py
│       ├── py.typed            # PEP 561 type marker
│       ├── config.py           # Global settings, paths, seed setup, API keys
│       ├── data/               # SemEval 2027 dataset loader & schema adapter
│       │   ├── __init__.py
│       │   └── loader.py       # AgentRisk/agentrisk-data-2027 ingestion to PlanDAGs
│       ├── planner/            # Task decomposition & multi-route DAG generation
│       │   ├── __init__.py
│       │   ├── schema.py       # Pydantic models for structured JSON graphs
│       │   ├── api_planner.py  # LLM API interface (Groq / OpenAI-compatible / Mock)
│       │   └── local_planner.py# Local Llama + LoRA inference engine
│       ├── risk_engine/        # Custom Risk Analysis Layer
│       │   ├── __init__.py
│       │   ├── feature_extractor.py # Psycholinguistic & topological feature extractor
│       │   ├── xgboost_shield.py    # Unified XGBoost tabular classifier shield
│       │   └── necessity_filter.py  # Task necessity & dead-branch pruner
│       ├── decision_theory/    # Advanced decision and policy components
│       │   ├── __init__.py
│       │   ├── markov_chain.py # MDP state transitions & trajectory risk modeling
│       │   └── game_theory.py  # Stackelberg / Adversarial shield equilibrium policy
│       └── utils/              # Telemetry & statistical evaluation
│           ├── __init__.py
│           ├── metrics.py      # Latency, RAM, CPU load profiling context manager
│           └── evaluator.py    # F1, Precision, Recall, ROC-AUC, schema accuracy
│
├── experiments/                # Research experiments and empirical workshops
│   ├── README.md               # Experiments documentation and execution flow
│   └── stage_01/               # Stage 1: API Planner + Unified XGBoost Classifier
│       ├── README.md           # Stage 01 workflow guide
│       ├── config.yaml
│       ├── 01_generate_dags.py
│       ├── 02_train_xgboost.py
│       └── 03_evaluate_baseline.py
│
└── tests/                      # Unit and integration test suite
    ├── README.md               # Test running instructions and test map
    ├── __init__.py
    ├── conftest.py             # Fixtures (PlanDAG, TaskNode)
    ├── test_config.py          # Settings validation
    ├── test_planner.py         # Schema, acyclic validation, planner mocks
    ├── test_feature_extraction.py # Keyword markers & topological metrics
    ├── test_xgboost_shield.py  # Classifier inference, verdicts, necessity pruning
    ├── test_decision_theory.py # Markov matrices & Stackelberg equilibrium
    └── test_data_loader.py     # SemEval 2027 dataset loader & adapter tests
```

---

## 🚀 Key Architectural Optimizations Over Initial Proposal

1. **Standard Namespaced Source Tree (`src/agent_risk/`)**:
   - *Issue in raw `src/planner`, `src/utils`*: Causes top-level import namespace collisions with third-party libraries (especially `utils`).
   - *Optimization*: Namespaced under `src/agent_risk/` for clean imports (`from agent_risk.planner import APIPlanner`), PEP 561 compliance (`py.typed`), and reliable packaging with `setuptools`/`flit`/`hatchling`.
2. **Modern Packaging (PEP 517 / 621)**:
   - Added `pyproject.toml` with configuration for dependencies, optional groups (`[project.optional-dependencies]`), and tool configurations (`pytest`, `black`, `ruff`).
3. **Turnkey Sample Datasets (`data/sample/`)**:
   - Included immediately runnable sample prompts (`sample_prompts.json`) and a validated sample DAG schema (`sample_dag.json`) so researchers and CI can execute pipelines without prior external downloads.
4. **Resilient Offline Fallbacks**:
   - The `APIPlanner` and `XGBoostShield` include deterministic offline fallbacks and bootstrap initializers so unit tests and experiments run out-of-the-box even without active API keys or GPU environments.
5. **Complete Test Suite & Fixtures**:
   - Comprehensive test suite covering graph acyclicity, psycholinguistic keyword detection, necessity pruning, stochastic transition matrices, and Stackelberg search.

---

## ⚡ Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ProtazSarkar/Agent_Risk.git
cd Agent_Risk

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Configure environment variables
cp .env.example .env
```

### 2. Run Test Suite

```bash
pytest
```

### 3. Load SemEval 2027 Dataset (`AgentRisk/agentrisk-data-2027`)

```bash
# Ingest and convert SemEval interaction traces into task DAGs
python scripts/fetch_semeval_data.py --dataset AgentRisk/agentrisk-data-2027 --split train
```

Or directly in Python:
```python
from agent_risk.data import AgentRiskDatasetLoader

# Ingest and convert to PlanDAGs
instances = AgentRiskDatasetLoader.load(dataset_name="AgentRisk/agentrisk-data-2027", split="train")
dags = [inst.to_plan_dag() for inst in instances]
```

---

## 🔬 Running Stage-by-Stage Experiments

### Running Stage 01 Experiments

```bash
# 1. Generate Task DAGs from sample prompts (uses offline mock if no API key is set)
python experiments/stage_01/01_generate_dags.py --mock

# 2. Extract features and train the XGBoost Shield
python experiments/stage_01/02_train_xgboost.py

# 3. Evaluate precision, recall, F1, and ROC-AUC
python experiments/stage_01/03_evaluate_baseline.py
```

---

## 🛠️ Python API Usage Example

```python
from agent_risk.planner import APIPlanner
from agent_risk.risk_engine import FeatureExtractor, XGBoostShield, NecessityFilter

# 1. Decompose goal into a task DAG
planner = APIPlanner()
plan = planner.generate_plan("Read dataset sales.csv, summarize revenue, and notify lead.")
route = plan.primary_route()

# 2. Filter redundant or unneeded steps
filter_engine = NecessityFilter(necessity_threshold=0.40)
necessary_tasks, pruned = filter_engine.filter_unnecessary_tasks(route)
print(f"Kept {len(necessary_tasks)} necessary tasks, pruned {len(pruned)} dead-end tasks.")

# 3. Extract features & predict risk verdict
extractor = FeatureExtractor()
df = extractor.extract_route_dataframe(route)

shield = XGBoostShield()
shield.load_model()
probs = shield.predict_proba(df)

for task, prob in zip(route.tasks, probs):
    verdict, score = shield.evaluate_verdict(prob)
    print(f"Task '{task.name}' -> Score: {score:.3f} | Verdict: {verdict.value}")
```

---

## 📜 License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

