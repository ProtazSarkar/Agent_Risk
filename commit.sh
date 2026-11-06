#!/usr/bin/env bash
# ==============================================================================
# Git Commit Script: Stages 04 & 05 Updates
# ==============================================================================
set -e

echo "Staging files for commit..."

# 1. Configuration & tooling
git add .gitignore pyproject.toml

# 2. Experiments documentation
git add experiments/README.md

# 3. Stage 01-03 updates & benchmarks
git add experiments/stage_01/
git add experiments/stage_02/
git add experiments/stage_03/

# 4. Stage 04: Multi-Prototype Anchor Classifier & Diagnostics
git add experiments/stage_04/README.md
git add experiments/stage_04/data/*.py
git add experiments/stage_04/domain_01/*.py
git add experiments/stage_04/output/

# 5. Stage 05: Context Distillation & Zero-Truncation Semantic Classifier
git add experiments/stage_05/README.md
git add experiments/stage_05/data/*.py
git add experiments/stage_05/domain_01/*.py
git add experiments/stage_05/output/

echo "Checking staged status..."
git status --short

echo "Committing changes..."
git commit -m "feat(experiments): add Stage 04 domain prototypes and Stage 05 context distillation pipeline

- Stage 04 (Multi-Prototype Anchors):
  * Implemented SemanticPrototypeClassifier for Domain 01 ('No Material Harm')
  * Added multi-prototype anchors covering both benign tasks and safe agent refusals
  * Fixed probability stuck at ~0.46 via calibrated temperature scaling (scale = 10.0)
  * Diagnosed root causes of false negatives on safe refusal cases

- Upstream Defect Diagnosis:
  * Discovered 100% action truncation defect where all-MiniLM-L6-v2 cut off text at
    token 256, dropping proposed_action and agent refusals across all 711 rows

- Stage 05 (Context Distillation & Zero Truncation):
  * Added process_summarized_dataset.py with Groq API & deterministic extraction
  * Distilled interactions into 60-90 word summaries with 0.0% action truncation
  * Lifted accuracy from 53.85% to 82.52% (+28.67% absolute improvement)
  * Increased F1 from 0.4762 to 0.8062 and ROC-AUC from 0.5572 to 0.8714
  * Resolved AGENTRISK-019 to No Harm (1) with 99.53% confidence

- Infrastructure & Tooling:
  * Updated .gitignore to ignore large CSV/Parquet datasets, weight tensors, and
    caches while preserving .py preprocessing code
  * Configured pyright extraPaths in pyproject.toml"

echo "✓ Successfully committed Stage 04 and Stage 05 changes!"

