# Stage 03: Hybrid 394-Dimensional Representation Pipeline

Stage 03 systematically implements the hybrid feature representation and range calibration architecture designed to overcome the accuracy and prediction compression bottlenecks discovered in Stage 02.

---

## 1. Core Architecture & Improvements

| Bottleneck Identified in Stage 02 | Solution Implemented in Stage 03 |
| :--- | :--- |
| **Extreme Information Loss** (dropping 384 embeddings caused model to lose fine-grained token/tool context) | **Hybrid Feature Representation**: Combines the 384 dense semantic embeddings with 5 enhanced domain similarities and 5 attention weights ($384 + 5 + 5 = 394$ dimensions). |
| **Feature Overlap** (generic prototype descriptions caused high similarity overlap across all risk classes) | **Multi-Perspective Domain Prototypes**: Detailed operational descriptions covering critical physical disasters, cyber exploitation/root escalations, PII/health data leaks, policy bypasses, and harmless routine tasks. |
| **Prediction Collapse Under Fixed Thresholds** (uncalibrated MSE caused outputs to cluster in $[1.68, 3.16]$, predicting zero samples in Level 1) | **Dynamic Range Calibration**: An affine scaling layer ($val_{\text{cal}} = \alpha \cdot val + \beta$) ensures continuous outputs span $[0.8, 4.8]$, properly populating all 5 user decision intervals: `(0, 1.5) -> 1, [1.5, 2.5) -> 2, [2.5, 3.5) -> 3, [3.5, 4.5) -> 4, >= 4.5 -> 5`. |
| **Class Imbalance** (Level 5 has only 7 samples; Level 2 has 33) | **Class-Balanced Weighting**: Applied across all deep neural networks and regressors. |

---

## 2. Directory Layout

```text
experiments/stage_03/
├── README.md               # Stage 03 architecture & overview
├── label.py                # Enhanced domain prototype descriptions and key matrix generation
├── attention.py            # Domain-aware attention, cosine similarity, and range mapping
├── processdata.py          # Dataset builder generating 394-d hybrid tabular datasets
├── hybrid_model.py         # PyTorch Dual-Branch Deep Neural Regressor & Classifier
├── main.py                 # Primary evaluation gateway and benchmarking suite
├── data/
│   └── processed/          # train_hybrid.csv / dev_hybrid.csv (.parquet)
├── artifacts/              # domain_keys.npy, saved model checkpoints (.pt)
└── output/                 # stage_03_benchmark_results.json, stage_03_report.md
```

---

## 3. How to Run

To run the complete preprocessing, training, and benchmark evaluation suite:

```powershell
python experiments/stage_03/main.py
```

