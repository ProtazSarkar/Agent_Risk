# Stage 06: Multi-Domain Probability Ensemble & Feature Summary Report

## 1. Executive Summary

This report documents the **Stage 06 feature preparation pipeline** that feeds directly into **Stage 07** risk level (1–5) severity models.

- **Pipeline Purpose**: Transforms raw interactions into an information-dense feature space using all 11 calibrated domain probability outputs from Stage 05.
- **Total Features**: 37 engineered tabular features.
- **Dataset Dimensions**:
  - Train: `639` rows $\times$ `41` columns
  - Dev  : `176` rows $\times$ `41` columns
- **Execution Time**: `1.03s`

---

## 2. Severity Target Distribution (Risk Levels 1 to 5)

| Risk Severity | Train Samples | Dev Samples | Description |
| :---: | :---: | :---: | :--- |
| **Severity 1** | `289` | `76` | Negligible / Public Info / Safe Refusal |
| **Severity 2** | `45` | `15` | Minor / Local Config / Non-Destructive |
| **Severity 3** | `131` | `35` | Moderate / Reversible Service Disruption |
| **Severity 4** | `141` | `38` | Serious / Data Exfiltration / Financial Loss |
| **Severity 5** | `33` | `12` | Critical / Catastrophic / Bodily Danger |
| **TOTAL** | **`639`** | **`176`** | Held-out validation split |

---

## 3. Top Predictive Features (Correlations with `risk_severity`)

| Feature Name | Pearson Correlation ($r$) |
| :--- | :---: |
| `prob_no_material_harm` | -0.7728 |
| `std_harm_prob` | +0.7650 |
| `max_harm_prob` | +0.7560 |
| `top1_harm_prob` | +0.7560 |
| `harm_to_safe_ratio` | +0.6715 |
| `mean_harm_prob` | +0.6617 |
| `sum_harm_prob` | +0.6617 |
| `num_domains_ge_05` | +0.6427 |
| `num_domains_ge_02` | +0.5896 |
| `refusal_keyword_count` | -0.5427 |
| `top2_harm_prob` | +0.4973 |
| `prob_privacy` | +0.4891 |
| `text_word_count` | -0.4045 |
| `pca_5` | -0.3466 |
| `is_physical_high` | +0.3329 |

---

## 4. Feature Groups Breakdown
1. **11 Domain Probabilities**: Calibrated outputs from `domain_01` to `domain_11`.
2. **Harm Aggregate Statistics**: `max_harm_prob`, `sum_harm_prob`, `harm_to_safe_ratio`, `domain_entropy`.
3. **Multi-Domain Counters**: `num_domains_ge_05`, `num_domains_ge_02`.
4. **Severity Flags**: `is_physical_high`, `is_security_high`, `is_financial_high`.
5. **PCA Latents**: 8 principal components capturing dense semantic context.
6. **Text & Refusal Keywords**: `refusal_keyword_count`, `critical_keyword_count`, `text_word_count`.
