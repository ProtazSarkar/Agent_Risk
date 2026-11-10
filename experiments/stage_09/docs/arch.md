# Stage 09: Architectural Blueprint & Ordinal Decision Pipeline

## 1. High-Level Concept
Predicting agent interaction risk severity ($Y \in \{1, 2, 3, 4, 5\}$) is an inherently **ordinal classification problem** with a strong class imbalance:
- Level 1 (Safe): ~46.6%
- Level 2 (Minor/Nuisance): ~5.7%
- Level 3 (Moderate Harm): ~22.4%
- Level 4 (Severe/Systemic Harm): ~24.1%
- Level 5 (Catastrophic Harm): ~1.3%

Stage 09 resolves the exact accuracy bottleneck by replacing single continuous regression or unconstrained multi-class loss with:
1. **5-Channel Disentangled Cross-Attentive Feature Ingestion** from the 11 trained Stage 08 prototype networks.
2. **Frank & Hall (2001) Cumulative Link Ordinal Decomposition** converting the problem into 4 monotonic binary threshold classifiers.
3. **Calibrated Soft Consensus Gating** ensuring accurate decision boundaries between adjacent levels (especially Level 3 vs Level 4).

---

## 2. End-to-End System Flowchart

```mermaid
flowchart TD
    subgraph S1["1. Raw Multi-Turn Interaction"]
        raw_text["Formatted Agent Interaction\n(Step 1..N [USER], [THOUGHT], [ACTION], [OBSERVATION])"]
        prop_act["Proposed Next Action\n(Pending Execution)"]
    end

    subgraph S2["2. 5-Channel Disentangled Tensors (Stage 08)"]
        ch0["Channel 0: Query\n(384-d MiniLM)"]
        ch1["Channel 1: Thought\n(384-d MiniLM)"]
        ch2["Channel 2: Context Actions\n(384-d MiniLM)"]
        ch3["Channel 3: Observations\n(384-d MiniLM)"]
        ch4["Channel 4: Proposed Action\n(384-d MiniLM)"]
    end

    subgraph S3["3. 11 Pre-Trained Cross-Attentive Domain Models"]
        m01["Domain 01: No Material Harm\n(P(Safe), Channel Weights, Margin)"]
        m02["Domain 02: Privacy / Confidentiality\n(P(Privacy))"]
        m03["Domain 03: Security / System Integrity\n(P(Security))"]
        m04["Domain 04: Financial / Assets\n(P(Financial))"]
        m05["Domain 05..11: Operational, Physical, Legal, etc."]
    end

    subgraph S4["4. Tabular Feature Engineering (39 Dimensions)"]
        dom_probs["11 Calibrated Domain Probabilities"]
        harm_stats["5 Harm Statistics: Max, Sum, Mean, Std, Safe"]
        dom_counts["3 Multi-Domain Counts: (>0.3, >0.5, >0.7)"]
        ch_sims["3 Channel Cosine Similarities: Query-Action, Thought-Action, Obs-Action"]
        attn_weights["5 Channel Pooling Weights from Domain 01"]
        margin["1 Positive vs Negative Prototype Margin"]
    end

    subgraph S5["5. Ordinal Decision Framework (Frank & Hall 2001)"]
        c1["Classifier C1:\nP(Y > 1 | X)\n(Safe vs Harm)"]
        c2["Classifier C2:\nP(Y > 2 | X)\n(Minor vs Moderate+)"]
        c3["Classifier C3:\nP(Y > 3 | X)\n(Moderate vs Severe+)"]
        c4["Classifier C4:\nP(Y > 4 | X)\n(Severe vs Catastrophic)"]
    end

    subgraph S6["6. Monotonic Probability Projection & Calibration"]
        mono["Monotonic Ordering Enforcement:\n1 >= P(Y > 1) >= P(Y > 2) >= P(Y > 3) >= P(Y > 4) >= 0"]
        exact_p["Exact Level Probabilities:\nP(Y = 1) = 1 - P(Y > 1)\nP(Y = 2) = P(Y > 1) - P(Y > 2)\nP(Y = 3) = P(Y > 2) - P(Y > 3)\nP(Y = 4) = P(Y > 3) - P(Y > 4)\nP(Y = 5) = P(Y > 4)"]
    end

    subgraph S7["7. Final Inference & Confidence"]
        pred["Predicted Severity Level:\ny_hat = argmax_k P(Y = k) in {1, 2, 3, 4, 5}"]
        conf["Calibrated Confidence Metric:\nconf = max_k P(Y = k) in [0, 1]"]
    end

    raw_text --> ch0
    raw_text --> ch1
    raw_text --> ch2
    raw_text --> ch3
    prop_act --> ch4

    ch0 & ch1 & ch2 & ch3 & ch4 --> m01 & m02 & m03 & m04 & m05

    m01 & m02 & m03 & m04 & m05 --> dom_probs & harm_stats & dom_counts & ch_sims & attn_weights & margin

    dom_probs & harm_stats & dom_counts & ch_sims & attn_weights & margin --> c1 & c2 & c3 & c4

    c1 & c2 & c3 & c4 --> mono --> exact_p --> pred & conf
```

---

## 3. Mathematical Formulation of Frank & Hall Decomposition

Given an ordinal target $y_i \in \{1, 2, \dots, K\}$ with $K = 5$:
1. The target is partitioned into $K - 1 = 4$ binary indicators:
   $$z_{ik} = \mathbb{I}(y_i > k) \quad \text{for } k \in \{1, 2, 3, 4\}$$
2. Four binary classifiers $C_1, C_2, C_3, C_4$ are trained independently on the entire dataset:
   $$P_k(x) = P(Y > k \mid X = x) = C_k(x)$$
3. Because probabilities must be monotonically non-increasing as thresholds increase, we enforce:
   $$\tilde{P}_1(x) = P_1(x)$$
   $$\tilde{P}_k(x) = \min(\tilde{P}_{k-1}(x), P_k(x)) \quad \text{for } k \in \{2, 3, 4\}$$
4. The exact probability distribution over the discrete ordinal levels $\{1, 2, 3, 4, 5\}$ is given by:
   $$P(Y = 1 \mid x) = 1 - \tilde{P}_1(x)$$
   $$P(Y = k \mid x) = \tilde{P}_{k-1}(x) - \tilde{P}_k(x) \quad \text{for } k \in \{2, 3, 4\}$$
   $$P(Y = 5 \mid x) = \tilde{P}_4(x)$$
5. The final prediction $\hat{y}$ and confidence score $\mathcal{C}$ are:
   $$\hat{y} = \arg\max_{k \in \{1, \dots, 5\}} P(Y = k \mid x)$$
   $$\mathcal{C} = \max_{k \in \{1, \dots, 5\}} P(Y = k \mid x)$$

---

## 4. Key Strengths Over Stage 06 Continuous Regression
- **Eliminates Regression Median Hedging**: Continuous regression with MSE loss naturally hedges predictions towards the sample mean, causing severe confusion between adjacent levels (e.g. predicting 3.4 when true is 4).
- **Domain Specialization**: Classifier $C_1$ specializes strictly in Safe vs Harm; Classifier $C_3$ specializes strictly in Moderate vs High-Consequence harm.
- **Calibrated Uncertainty**: Samples where $P(Y=3) \approx P(Y=4)$ receive low confidence scores, enabling automated escalation to human supervisors.

