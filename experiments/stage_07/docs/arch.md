# Stage 07 Architecture: Multi-Section Chunked Embedding Optimizer

## 1. Architectural Overview & Design Rationale

Stage 07 was designed to investigate whether disentangling the interaction trajectory into 4 structural sections (`[USER]`, `[ACTION]`, `[THOUGHT]`, and `[OBSERVATION]`) combined with sliding-window chunking could overcome the 256-token cutoff defect of standard dense sentence encoders.

---

## 2. End-to-End Architectural Flow Diagram

```mermaid
flowchart TD
    subgraph InputProcessing["1. Interaction Parsing & Sliding Chunking"]
        Raw["Interaction Trajectory (SemEval-2027)"]
        Raw --> SecU["query: [USER] Text"]
        Raw --> SecA["action: [ACTION] Steps + proposed_action"]
        Raw --> SecT["thought: [THOUGHT] Reasoning"]
        Raw --> SecO["observation: [OBSERVATION] Tool Outputs"]

        ChunkU["Sliding Chunks (120w, 20w overlap)"]
        ChunkA["Sliding Chunks (120w, 20w overlap)"]
        ChunkT["Sliding Chunks (120w, 20w overlap)"]
        ChunkO["Sliding Chunks (120w, 20w overlap)"]

        SecU --> ChunkU
        SecA --> ChunkA
        SecT --> ChunkT
        SecO --> ChunkO
    end

    subgraph DenseEncoding["2. Dense Sentence Encoding & Mean Pooling"]
        Enc["SentenceTransformer: all-MiniLM-L6-v2 (d = 384)"]
        ChunkU & ChunkA & ChunkT & ChunkO --> Enc
        MeanU["Mean Pool -> e_query in R^384"]
        MeanA["Mean Pool -> e_action in R^384"]
        MeanT["Mean Pool -> e_thought in R^384"]
        MeanO["Mean Pool -> e_obs in R^384"]
        Enc --> MeanU & MeanA & MeanT & MeanO
        Stack["Stacked Section Tensor: X in R^(B x 4 x 384)"]
        MeanU & MeanA & MeanT & MeanO --> Stack
    end

    subgraph NeuralOptimizer["3. SectionEmbeddingOptimizer"]
        AttnNet["Dynamic Section-Attention Network:<br/>Linear(384->256) -> LayerNorm -> Tanh -> Linear(256->1)"]
        Stack --> AttnNet
        Softmax["Softmax across 4 sections -> Weights alpha in R^4"]
        AttnNet --> Softmax
        WeightedPool["Weighted Pooling: sum(alpha_s * x_s) in R^384"]
        Softmax & Stack --> WeightedPool
        Refine["Non-linear Projection + LayerNorm + GELU + Residual Gating"]
        WeightedPool --> Refine
        FusedQ["Fused Query Vector: q_fused in R^384"]
        Refine --> FusedQ
    end

    subgraph PrototypeClassification["4. Prototype Alignment & Output Head"]
        QProj["Query Projection: Linear(384->384) -> LayerNorm -> ReLU"]
        FusedQ --> QProj
        ProtoKeys["Pre-Computed Prototype Keys Buffer:<br/>4 Safe Keys ('No Material Harm'), 4 Harm Keys in R^(8 x 384)"]
        KeyProj["Key Projection: Linear(384->384)"]
        ProtoKeys --> KeyProj
        Sim["Scaled Dot-Product Cosine Similarity in R^8"]
        QProj & KeyProj --> Sim
        Head["Classification Head: Linear(8->32) -> ReLU -> Linear(32->1)"]
        Sim --> Head
        Output["Calibrated Probability P(Safe = 1) via Sigmoid"]
        Head --> Output
    end
```

---

## 3. Mathematical Formulation
* **Section Representations**: For section $s \in \{\text{query}, \text{action}, \text{thought}, \text{obs}\}$, sliding chunks $C_{s,1}, \dots, C_{s,M}$ are encoded and mean-pooled:
  $$\mathbf{x}_s = \text{Normalize}\left(\frac{1}{M}\sum_{m=1}^M \text{Encoder}(C_{s,m})\right) \in \mathbb{R}^{384}$$
* **Dynamic Section Attention**:
  $$\alpha_s = \frac{\exp(\mathbf{w}_a^T \tanh(\mathbf{W}_1 \mathbf{x}_s + \mathbf{b}_1))}{\sum_{k=1}^4 \exp(\mathbf{w}_a^T \tanh(\mathbf{W}_1 \mathbf{x}_k + \mathbf{b}_1))}$$
* **Prototype Dot-Product Similarity**:
  $$\mathbf{s}[j] = \frac{1}{\sqrt{d}} \left(\frac{\mathbf{W}_q \mathbf{q}_{\text{fused}}}{\|\mathbf{W}_q \mathbf{q}_{\text{fused}}\|} \cdot \frac{\mathbf{W}_k \mathbf{k}_j}{\|\mathbf{W}_k \mathbf{k}_j\|}\right)$$

