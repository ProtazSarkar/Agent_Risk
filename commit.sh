#!/usr/bin/env bash
# ==============================================================================
# Bash Git Commit Script: Stage 05, 06, 07, & 08
# ==============================================================================
set -e

echo "Staging files for commit..."

# 1. Configuration & tooling
git add .gitignore pyproject.toml commit.ps1 commit.sh

# 2. Source utilities
git add src/utils/summerizer.py

# 3. Experiments documentation
git add experiments/README.md

# 4. Stages 01 to 08
git add experiments/stage_01/
git add experiments/stage_02/
git add experiments/stage_03/
git add experiments/stage_04/
git add experiments/stage_05/
git add experiments/stage_06/
git add experiments/stage_07/
git add experiments/stage_08/

echo -e "\nStaged changes:"
git status --short

COMMIT_MSG="feat(stage_07,stage_08): implement 5-channel cross-attentive prototype network across all 11 domains

- Utility (src/utils/summerizer.py):
  * Multi-section interaction parsing: [USER], [ACTION], [THOUGHT], [OBSERVATION], proposed_action
  * Sliding-window word chunking (120w, 20w overlap) with mean-pooling for zero truncation
  * Bulk vectorized sentence encoder and learnable SectionEmbeddingOptimizer

- Stage 07 (Multi-Section Chunked Embedding Optimizer):
  * Disentangles 4 interaction roles with dynamic neural attention
  * Evaluated on SemEval Domain 01 (No Material Harm): 84.62% Acc, 0.8483 ROC-AUC

- Stage 08 (5-Channel Cross-Attentive Prototype Network Suite):
  * 5-channel disentanglement isolating pending proposed_action from history
  * Self-Attention Bridge connecting observations to proposed actions
  * Domain prototypes act as query probes cross-attending across interaction tokens
  * Benchmark across all 11 SemEval domains: 91.48% Macro Acc, 0.7086 Macro F1 (+9.01% gain), 0.8505 Macro ROC-AUC
  * Added complete research documentation & flowcharts in docs/ for Stages 05, 06, 07, and 08"

git commit -m "$COMMIT_MSG"
echo -e "\nCommit completed successfully!"
