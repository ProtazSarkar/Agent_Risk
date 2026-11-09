# Stage 07: Multi-Section Chunked Interaction Embedding & Domain 01 Classifier

## 1. Overview
Stage 07 implements a **Multi-Section Chunked Embedding Optimizer** paired with a **Semantic Prototype Classifier** for **Domain 01: No Material Harm** (Binary classification: `Safe (1)` vs `Harmful (0)`).

Standard dense encoders (such as `all-MiniLM-L6-v2`) truncate inputs at 256 tokens. Because real agent interactions average 450–1,300 words with extensive environment observations, truncation destroys 70–80% of downstream tool outputs and action arguments where safety hazards occur.

Stage 07 resolves this by:
1. **Section Disentanglement**: Isolates `[USER]` (`query`), `[ACTION]` (+ pending proposed action), `[THOUGHT]`, and `[OBSERVATION]`.
2. **Sliding Chunking & Mean Pooling**: Chunks sections that exceed 120 words with 20-word overlap, generating zero-truncation dense representations via `all-MiniLM-L6-v2` and mean pooling chunk vectors per section.
3. **Learnable Section Fusion Optimizer**: A neural attention module (`SectionEmbeddingOptimizer`) dynamically assigns importance weights across the 4 sections based on interaction context.
4. **Semantic Prototype Classification**: Evaluates scaled dot-product similarity against pre-computed `domain_01` positive and negative prototype keys.

---

## 2. Directory Structure

```
experiments/stage_07/
├── data/
│   ├── .gitkeep
│   ├── train_sections.pt        # (568, 4, 384) cached section tensors
│   └── dev_sections.pt          # (143, 4, 384) held-out SemEval dev tensors
├── models/
│   ├── .gitkeep
│   └── best_domain_01_model.pt  # Champion checkpoint
├── output/
│   ├── training_history.json
│   ├── domain_01_results.json
│   └── domain_01_report.md      # Detailed benchmark report
├── prepare_dataset.py           # Vectorized section extraction & chunked embedding
├── model.py                     # SectionEmbeddingOptimizer & Stage07DomainClassifier
├── train.py                     # End-to-end training pipeline
├── main.py                      # Evaluation gateway & prediction trace
└── README.md
```

---

## 3. Workflow & Usage

### Step 1: Extract & Cache Section Embeddings
```powershell
python -u experiments/stage_07/prepare_dataset.py
```

### Step 2: Train Model with Section Attention
```powershell
python -u experiments/stage_07/train.py
```

### Step 3: Run Evaluation Gateway
```powershell
python -u experiments/stage_07/main.py
```

