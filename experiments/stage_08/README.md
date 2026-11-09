# Stage 08: 5-Channel Cross-Attentive Prototype Classifier (Domain 01)

## 1. Executive Summary
Stage 08 is the **Optimized Advanced Architecture** for **Domain 01: No Material Harm** binary classification (`Safe (1)` vs `Harmful (0)`).

Building on the multi-section findings from Stage 07, Stage 08 addresses semantic dilution by introducing four architectural optimizations:
1. **5-Channel Disentanglement**:
   Explicitly isolates the pending execution step (`proposed_action`) from historical actions (`context_actions`).
   - Channel 0: `[USER]` (`query`)
   - Channel 1: `[THOUGHT]` (`thought`)
   - Channel 2: `[ACTION]` (`context_actions`)
   - Channel 3: `[OBSERVATION]` (`observations`)
   - Channel 4: `proposed_action` (pending execution command)
2. **Channel-Type Learnable Positional Embeddings**:
   Injects dedicated role vectors to each interaction token.
3. **Domain-Prototype Cross-Attention**:
   Positive (Safe) and Negative (Harmful) prototype keys act as active queries cross-attending across the 5 interaction tokens, calculating exact alignment with safety guidelines.
4. **Contrastive Prototype Margin Loss**:
   Combined loss function ($L = L_{\text{BCE}} + \lambda_{\text{margin}} L_{\text{margin}}$) that pushes harmful interactions away from the safe prototype manifold.

---

## 2. Directory Structure

```
experiments/stage_08/
├── data/
│   ├── .gitkeep
│   ├── train_5channels.pt       # (568, 5, 384) 5-channel training tensors
│   └── dev_5channels.pt         # (143, 5, 384) 5-channel held-out SemEval dev tensors
├── models/
│   ├── .gitkeep
│   └── best_domain_01_stage08.pt# Optimized champion checkpoint
├── output/
│   ├── training_history.json
│   ├── domain_01_results.json
│   └── domain_01_report.md      # Detailed benchmark report & Stage 07 comparison
├── prepare_dataset.py           # 5-channel extraction & chunked bulk encoding
├── model.py                     # 5-Channel Cross-Attentive Prototype Network
├── train.py                     # Combined BCE + Prototype Margin training suite
├── main.py                      # Evaluation gateway & comparative analysis
└── README.md
```

---

## 3. Workflow & Usage

### Step 1: Extract 5-Channel Tensors
```powershell
python -u experiments/stage_08/prepare_dataset.py
```

### Step 2: Train Model with Margin Loss & Cross-Attention
```powershell
python -u experiments/stage_08/train.py
```

### Step 3: Run Evaluation Gateway
```powershell
python -u experiments/stage_08/main.py
```

