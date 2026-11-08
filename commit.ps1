# ==============================================================================
# PowerShell Git Commit Script: Stage 05 & Stage 06 Unified Pipeline
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "Staging files for commit..." -ForegroundColor Cyan

# 1. Configuration & tooling
git add .gitignore pyproject.toml commit.ps1 commit.sh

# 2. Experiments documentation
git add experiments/README.md

# 3. Stage 01-04
git add experiments/stage_01/
git add experiments/stage_02/
git add experiments/stage_03/
git add experiments/stage_04/

# 4. Stage 05: All 11 Domains & Multi-Domain Pipeline
git add experiments/stage_05/README.md
git add experiments/stage_05/run_all_domains.py
git add experiments/stage_05/scaffold_domains.py
git add experiments/stage_05/artifacts/.gitkeep
git add experiments/stage_05/artifacts/*.py
git add experiments/stage_05/data/*.py
git add experiments/stage_05/output/*.json
git add experiments/stage_05/output/*.md
git add experiments/stage_05/domain_01/
git add experiments/stage_05/domain_02/
git add experiments/stage_05/domain_03/
git add experiments/stage_05/domain_04/
git add experiments/stage_05/domain_05/
git add experiments/stage_05/domain_06/
git add experiments/stage_05/domain_07/
git add experiments/stage_05/domain_08/
git add experiments/stage_05/domain_09/
git add experiments/stage_05/domain_10/
git add experiments/stage_05/domain_11/

# 5. Stage 06: Consolidated Multi-Domain Probability Ensemble & Severity Prediction Suite
git add experiments/stage_06/README.md
git add experiments/stage_06/prepare_dataset.py
git add experiments/stage_06/models.py
git add experiments/stage_06/train.py
git add experiments/stage_06/main.py
git add experiments/stage_06/data/.gitkeep
git add experiments/stage_06/models/.gitkeep
git add experiments/stage_06/output/*.json
git add experiments/stage_06/output/*.md

Write-Host "`nStaged changes:" -ForegroundColor Yellow
git status --short

$commitMessage = @"
feat(stage_05,stage_06): implement all 11 prototype domain classifiers and consolidated tree/regression severity suite

- Stage 05 (11 Domain Semantic Prototype Classifiers):
  * Built balanced datasets and prototype keys for all 11 SemEval risk domains
  * Implemented query projection, prototype key projection, and scaled dot-product similarity
  * Automated batch training and benchmarking across all 11 domains (Macro Acc: 91.25%, Macro ROC-AUC: 0.8285)

- Stage 06 (Consolidated Multi-Domain Ensemble & Severity 1-5 Models):
  * Extracted 37 tabular features across all 11 domain prototype models + interaction text signals
  * Implemented and benchmarked XGBoost, HistGradientBoosting, Random Forest, Ordinal Regressors, and Hybrid Ensemble
  * Added Nelder-Mead ordinal threshold boundary optimization and calibrated confidence estimation
  * Champion model achieves 85.80% Within-1 Accuracy, 0.6895 QWK, 0.5966 MAE, and 89.80% High-Confidence Accuracy on held-out dev data
  * Exported severity_benchmark_results.json and comprehensive severity_benchmark_report.md
"@

git commit -m $commitMessage
Write-Host "`nCommit completed successfully!" -ForegroundColor Green
