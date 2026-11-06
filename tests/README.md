# `tests/` - Automated Test Suite

## 🎯 Purpose
The `tests/` directory provides comprehensive unit and integration testing across all framework layers. The test suite is dual-compatible: it runs with standard `python -m unittest` (no external test runner needed) as well as with `pytest`.

---

## 📁 Directory Content
- **`conftest.py`**: Shared test fixtures (`sample_task_node`, `sample_plan_dag`).
- **`test_config.py`**: Validates configuration loading, path resolution, and safeguard threshold ordering.
- **`test_planner.py`**: Validates DAG schemas, NetworkX conversion, cycle detection, and offline planner fallbacks.
- **`test_feature_extraction.py`**: Tests psycholinguistic keyword matching, Shannon entropy, and graph topological extraction.
- **`test_xgboost_shield.py`**: Tests model predictions, baseline initialization, and necessity dead-branch pruning.
- **`test_decision_theory.py`**: Tests Markov transition stochastic matrices, absorbing states, and Stackelberg search.
- **`test_data_loader.py`**: Validates SemEval 2027 schema parsing, instance conversion, and PlanDAG adaptation.

---

## 🧪 Running the Tests
```bash
# Using pytest (if installed)
pytest

# Or using standard Python unittest
python -m unittest discover -s tests -v
```

