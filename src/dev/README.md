# `src/dev/` - Experimental AI Agents & Prototyping

## 🎯 Purpose
The `src/dev/` directory serves as an active experimental development workspace. Here, researchers and developers can design, prototype, and benchmark custom **AI agents** specialized in:
- Autonomous risk analysis and threat intelligence.
- Multi-agent collaboration frameworks (e.g., Researcher $\rightarrow$ Critic $\rightarrow$ Shield $\rightarrow$ Executor).
- Domain-specific application agents (e.g., automated code security auditors, workflow interceptors, and policy compliance monitors).

---

## 📁 Directory Content
- **`__init__.py`**: Exposes experimental agent modules for easy import.
- **`sample_agent.py`**: A clean, modular base implementation featuring:
  - `BaseDevAgent`: Abstract agent interface with context management.
  - `RiskAnalysisAgent`: An autonomous agent that intercepts queries, decomposes them into task DAGs, inspects every action via `XGBoostShield`, and renders policy decisions.

---

## 🚀 How to Build a New Agent in `src/dev/`

1. Create a new python file, e.g. `src/dev/my_custom_agent.py`.
2. Inherit from `BaseDevAgent` or integrate components from `agent_risk`:

```python
from src.dev.sample_agent import BaseDevAgent
from agent_risk.risk_engine import FeatureExtractor, XGBoostShield

class MyCustomAgent(BaseDevAgent):
    def act(self, input_query: str):
        # Your custom agent reasoning and tool usage here
        return {"status": "success"}
```

3. Test your experimental agent directly:
```bash
python -c "from src.dev.sample_agent import RiskAnalysisAgent; agent = RiskAnalysisAgent(); print(agent.act('Inspect system memory and report usage'))"
```

