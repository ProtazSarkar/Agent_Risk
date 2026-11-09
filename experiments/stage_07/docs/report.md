# Stage 07 Benchmark Report & Empirical Analysis

## 1. Executive Performance on Held-Out SemEval Dev Data (143 Samples)

- **Exact Binary Accuracy**: **84.62%** (Baseline before chunking was ~50%)
- **Macro F1-Score**: **0.8444**
- **ROC-AUC**: **0.8483**
- **Safe (1) F1 / Precision / Recall**: **0.8281** / 0.8413 / 0.8154 (Class count: 65)
- **Harmful (0) F1 / Precision / Recall**: **0.8608** / 0.8500 / 0.8718 (Class count: 78)
- **High-Confidence (>=70%) Accuracy**: **84.62%** (143/143 samples)

---

## 2. Learned Section Attention Distribution

The dynamic `SectionEmbeddingOptimizer` converged on the following empirical weights:
* **`[ACTION]` (+ proposed action)**: **42.98%**
* **`[THOUGHT]`**: **43.64%**
* **`[OBSERVATION]`**: **8.23%**
* **`[USER]` (`query`)**: **5.15%**

---

## 3. Confusion Matrix (Safe = 1, Harmful = 0)

| True \ Predicted | Pred Safe (1) | Pred Harmful (0) | Total | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **True Safe (1)** | **53** (TP) | 12 (FN) | 65 | 81.5% |
| **True Harmful (0)** | 10 (FP) | **68** (TN) | 78 | 87.2% |

---

## 4. Research Discussion: Why Did Stage 07 Jump to 84.62%?

1. **Resolution of Truncation Defect**:
   In Stage 01–04, encoding raw text with `all-MiniLM-L6-v2` truncated after 256 tokens, lopping off the final tool commands and proposed actions in 100% of samples longer than 300 words. Because sliding chunking embeds all chunks and averages them, 100% of trajectory tokens were preserved.
2. **Dominance of Action and Thought**:
   The attention network placed **86.62% of its weight on `[ACTION]` and `[THOUGHT]`**. This demonstrates that the agent's internal intent (`thought`) and tool executions (`action`) contain the core semantic signals distinguishing routine tasks from destructive operations.
3. **Identified Failure Mode (The Motivation for Stage 08)**:
   Stage 07 still suffered 12 false negatives and 10 false positives. Qualitative inspection revealed that because `proposed_action` was concatenated directly into `context_actions`, historical diagnostic commands diluted the pending action. This finding directly motivated Stage 08's 5-channel disentanglement.
