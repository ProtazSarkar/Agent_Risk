# Stage 06 Architecture: Multi-Domain Probability Ensemble & Severity Predictor

## 1. Architectural Motivation

Predicting the ordinal **Risk Severity (Levels 1 to 5)** directly from raw text fails because text-level regressors lack grounding in specific domain hazards.
Stage 06 addresses this by constructing a **Two-Tier Hierarchical Framework**:
1. **Tier 1 (Domain Expert Feature Extractor)**: Interaction embeddings are evaluated against all 11 Domain Prototype Classifiers, extracting 11 calibrated domain risk probabilities plus 26 engineered statistical & context features (37 tabular features total).
2. **Tier 2 (Ordinal Risk Predictor)**: Tree-based ensembles (XGBoost, HistGradientBoosting, Random Forest) and continuous ordinal regressors map the 37-dimensional feature representation into discrete severity levels 1–5 with calibrated confidence.

---

## 2. End-to-End Architectural Flow Diagram

```mermaid
flowchart TD
    subgraph InputTrajectory["1. Input Trajectory"]
        Text["Interaction Trajectory & Proposed Action"]
        Enc["SentenceTransformer (all-MiniLM-L6-v2) -> e in R^384"]
        Text --> Enc
    end

    subgraph Tier1DomainExperts["2. Tier 1: 11 Domain Expert Prototype Classifiers"]
        D1["Domain 01: No Material Harm -> P(NoHarm)"]
        D2["Domain 02: Privacy -> P(Privacy)"]
        D3["Domain 03: Security -> P(Security)"]
        D4["Domain 04: Financial -> P(Financial)"]
        D5["Domain 05: Operational -> P(Operational)"]
        D6["Domain 06: Physical -> P(Physical)"]
        D7["Domain 07: Legal -> P(Legal)"]
        D8["Domain 08: Reputational -> P(Reputational)"]
        D9["Domain 09: Discrimination -> P(Discrimination)"]
        D10["Domain 10: Environmental -> P(Environmental)"]
        D11["Domain 11: Information Quality -> P(InfoQuality)"]

        Enc --> D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 & D9 & D10 & D11
    end

    subgraph FeatureEngineering["3. Compound Tabular Feature Formulation (37 Features)"]
        Probs["11 Domain Risk Probabilities: [p_1, ..., p_11]"]
        D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 & D9 & D10 & D11 --> Probs

        Stats["Compound Harm Statistics:<br/>* max_harm_prob<br/>* sum_harm_prob<br/>* std_harm_prob<br/>* harm_to_safe_ratio<br/>* domain_entropy"]
        Thresholds["Sparsity Counters:<br/>* num_domains_ge_05<br/>* num_domains_ge_07"]
        PCA["Interaction Text Geometry:<br/>* PCA Latents (5 dims) from embedding"]
        Lexical["Action & Refusal Keywords:<br/>* refusal_keyword_count<br/>* destructive_keyword_count"]

        Probs --> Stats & Thresholds
        Enc --> PCA & Lexical

        FeatVector["Consolidated Tabular Vector: v in R^37"]
        Probs & Stats & Thresholds & PCA & Lexical --> FeatVector
    end

    subgraph Tier2SeverityModels["4. Tier 2: Candidate Risk Severity Predictors (Levels 1-5)"]
        HGBR["HistGradientBoosting_Regressor (Champion)"]
        RFR["RandomForest_Regressor"]
        XGBC["XGBoost_Classifier"]
        XGBR["XGBoost_Ordinal_Regressor"]
        RidgeR["Ridge_Ordinal_Regressor"]
        Hybrid["Hybrid_XGB_Ensemble"]

        FeatVector --> HGBR & RFR & XGBC & XGBR & RidgeR & Hybrid
    end

    subgraph CalibrationOutput["5. Threshold Calibration & Output Gateway"]
        Continuous["Continuous Latent Severity: y_cont in [1.0, 5.0]"]
        ThresholdsOpt["Nelder-Mead Ordinal Threshold Boundaries: [t1, t2, t3, t4]"]
        HGBR --> Continuous --> ThresholdsOpt
        Discrete["Predicted Risk Severity Level (1, 2, 3, 4, 5)"]
        ThresholdsOpt --> Discrete
        Conf["Prediction Confidence: max_k P(y = k) in [0.0, 1.0]"]
        Discrete & Conf --> FinalOutput["Final Output: Risk Level + Confidence + Explanation"]
    end
```

---

## 3. Mathematical Specification
1. **Domain Feature Vector**:
   $\mathbf{p} = [p_{\text{noharm}}, p_{\text{privacy}}, \dots, p_{\text{info}}] \in [0, 1]^{11}$
2. **Compound Statistics**:
   - `max_harm_prob`: $\max_{k \ge 2} p_k$
   - `std_harm_prob`: $\text{Std}(p_2, \dots, p_{11})$
   - `harm_to_safe_ratio`: $\frac{\sum_{k \ge 2} p_k}{p_{\text{noharm}} + 10^{-4}}$
   - `domain_entropy`: $-\sum_{k=1}^{11} \tilde{p}_k \log(\tilde{p}_k + 10^{-6})$
3. **Continuous-to-Ordinal Discretization**:
   $$\hat{Y} = 1 + \sum_{j=1}^4 \mathbb{I}(y_{\text{cont}} \ge \theta_j)$$
   Where boundaries $\boldsymbol{\theta} = [\theta_1, \theta_2, \theta_3, \theta_4]$ are optimized via Nelder-Mead simplex to maximize Quadratic Weighted Kappa (QWK).

