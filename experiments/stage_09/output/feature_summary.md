# Stage 09: 5-Channel Disentangled Severity Feature Suite

This document summarizes the 39 tabular features extracted from the 11 trained Stage 08 5-channel models.

## Dataset Overview

- **Train Samples**: 568
- **Dev Samples**: 143
- **Total Extracted Features**: 39

## Severity Class Distribution

| Severity Level | Train Count | Train % | Dev Count | Dev % |
|---|---|---|---|---|
| Level 1 | 265 | 46.65% | 66 | 46.15% |
| Level 2 | 33 | 5.81% | 8 | 5.59% |
| Level 3 | 127 | 22.36% | 32 | 22.38% |
| Level 4 | 136 | 23.94% | 35 | 24.48% |
| Level 5 | 7 | 1.23% | 2 | 1.40% |

## Top Feature Correlations with Severity (Pearson r)

| Feature Name | Pearson Correlation (r) | Feature Category |
|---|---|---|
| `sim_margin_dom01` | -0.8629 | Channel Signal |
| `harm_dom01` | +0.8592 | Harm Probability |
| `prob_dom01_safe` | -0.8592 | Harm Probability |
| `safe_prob` | -0.8592 | Harm Probability |
| `pool_weight_query` | -0.7476 | Channel Signal |
| `max_harm_prob` | +0.6710 | Harm Probability |
| `std_harm_prob` | +0.6349 | Harm Probability |
| `prob_dom02_privacy` | +0.6128 | Harm Probability |
| `harm_dom02_privacy` | +0.6128 | Harm Probability |
| `mean_harm_prob` | +0.5727 | Harm Probability |
| `sum_harm_prob` | +0.5727 | Harm Probability |
| `sim_thought_action` | -0.5255 | Channel Signal |
| `count_domains_gt_07` | +0.5224 | Channel Signal |
| `count_domains_gt_05` | +0.4890 | Channel Signal |
| `count_domains_gt_03` | +0.4763 | Channel Signal |