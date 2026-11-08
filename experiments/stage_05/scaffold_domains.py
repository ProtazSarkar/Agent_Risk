"""
Stage 05: Scaffolding Generator for Domains 03 to 11.

Generates modular, isolated domain packages using simple token replacement:
  - domain_03: Security / System Integrity
  - domain_04: Financial / Assets
  - domain_05: Operational / Service
  - domain_06: Physical / Health
  - domain_07: Legal / Compliance
  - domain_08: Reputational / Social / Psychological
  - domain_09: Discrimination / Fairness
  - domain_10: Environmental
  - domain_11: Information Quality
"""

from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent

DOMAINS = [
    ("domain_03", "Security / System Integrity"),
    ("domain_04", "Financial / Assets"),
    ("domain_05", "Operational / Service"),
    ("domain_06", "Physical / Health"),
    ("domain_07", "Legal / Compliance"),
    ("domain_08", "Reputational / Social / Psychological"),
    ("domain_09", "Discrimination / Fairness"),
    ("domain_10", "Environmental"),
    ("domain_11", "Information Quality")
]

MODEL_TEMPLATE = '''"""
Stage 05: __DOM_ID__ ('__DOM_NAME__') Model Architecture.

Implements the SemanticPrototypeClassifier trained on summarized embeddings:
- Multi-prototype cross-attention with calibrated temperature scaling (scale = 10.0)
- Trainable Query Projection Network (query_proj) and Key Projection Network (key_proj)
- Attentive Fusion Head combining prototype similarity signals with query latent features
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPrototypeClassifier(nn.Module):
    """
    Semantic Prototype Classifier for __DOM_ID__: '__DOM_NAME__'.
    Projects 384-d summarized input query embeddings and pre-computed prototype key embeddings
    into a shared latent metric space, computing temperature-scaled dot-product similarity.
    """

    def __init__(
        self,
        initial_prototype_embeddings: torch.Tensor,
        num_pos: int = 4,
        num_neg: int = 4,
        hidden_dim: int = 64,
        initial_scale: float = 10.0,
        dropout_rate: float = 0.20
    ):
        super().__init__()
        self.embedding_dim = initial_prototype_embeddings.shape[-1]
        self.num_prototypes = initial_prototype_embeddings.shape[0]
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.hidden_dim = hidden_dim

        # 1. Registered Buffer: Pre-computed text prototype embeddings (zero random noise)
        self.register_buffer("prototype_keys", initial_prototype_embeddings.clone().detach())

        # 2. Trainable Query Projection Network: Q -> Q_h
        self.query_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 3. Trainable Key Projection Network: K -> K_h
        self.key_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 4. Calibrated Scale & Bias
        self.scale = nn.Parameter(torch.tensor(initial_scale))
        self.bias = nn.Parameter(torch.tensor(0.0))

        # 5. Attentive Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(self.num_prototypes + hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def compute_projected_embeddings(
        self,
        q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q_h = self.query_proj(q)
        q_norm = F.normalize(q_h, p=2, dim=-1)

        k_h = self.key_proj(self.prototype_keys)
        k_norm = F.normalize(k_h, p=2, dim=-1)
        return q_h, q_norm, k_norm

    def compute_prototype_similarities(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_h, q_norm, k_norm = self.compute_projected_embeddings(q)
        sim = torch.matmul(q_norm, k_norm.t()) * self.scale
        return sim, q_h

    def forward_logits(self, q: torch.Tensor) -> torch.Tensor:
        sim, q_h = self.compute_prototype_similarities(q)
        fused = torch.cat([sim, q_h], dim=-1)
        logits = self.fusion_head(fused).squeeze(-1) + self.bias
        return logits

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(q)
        return torch.sigmoid(logits)

    def predict_proba(self, q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q, dtype=torch.float32)
            probs = self.forward(q)
            return probs.cpu().numpy()

    def predict(
        self,
        q: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.50
    ) -> np.ndarray:
        probs = self.predict_proba(q)
        return (probs >= threshold).astype(int)
'''

PROTOTYPES_TEMPLATE = '''"""
Stage 05: __DOM_ID__ ('__DOM_NAME__') Prototype Key Loader & Generator.

Loads pre-computed 384-dimensional dense semantic key vectors from:
  artifacts/__DOM_ID___prototype_keys.pt
Zero random noise is used.
"""

from pathlib import Path
from typing import Dict, Any
import torch

STAGE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = STAGE_DIR / "artifacts" / "__DOM_ID___prototype_keys.pt"


def generate___DOM_ID___prototype_keys(
    force_regenerate: bool = False,
    cache_path: Path = CACHE_PATH
) -> Dict[str, Any]:
    if not force_regenerate and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    # If cache does not exist, trigger generator
    from experiments.stage_05.artifacts.generate_all_prototypes import generate_all_prototype_keys
    generate_all_prototype_keys(force_regenerate=True)
    return torch.load(cache_path, weights_only=False)


if __name__ == "__main__":
    payload = generate___DOM_ID___prototype_keys()
    keys = payload["all_keys"]
    print(f"Loaded prototype keys for __DOM_ID__ ('__DOM_NAME__'): shape {keys.shape} ({payload['num_pos']} Pos, {payload['num_neg']} Neg)")
'''

TRAIN_TEMPLATE = '''"""
Stage 05: __DOM_ID__ ('__DOM_NAME__') Training Suite.

Trains the SemanticPrototypeClassifier for __DOM_ID__:
- Loads 384-d dense semantic embeddings from data/train___DOM_ID__.csv and data/dev___DOM_ID__.csv
- Loads pre-computed prototype key vectors (zero random noise)
- Uses class-weighted BCE loss with AdamW and weight decay
- Optimizes decision threshold on training set
- Persists trained model weights to experiments/stage_05/models/__DOM_ID___classifier.pt
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

STAGE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = STAGE_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from experiments.stage_05.__DOM_ID__.model import SemanticPrototypeClassifier
    from experiments.stage_05.__DOM_ID__.prototypes import generate___DOM_ID___prototype_keys
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(STAGE_DIR / "__DOM_ID__"))
    from model import SemanticPrototypeClassifier
    from prototypes import generate___DOM_ID___prototype_keys


def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    data_dir = STAGE_DIR / "data"
    tr_path = data_dir / "train___DOM_ID__.csv"
    dv_path = data_dir / "dev___DOM_ID__.csv"

    if not tr_path.exists() or not dv_path.exists():
        from generate_domain_datasets import generate_all_domain_datasets
        generate_all_domain_datasets()

    df_tr = pd.read_csv(tr_path)
    df_dv = pd.read_csv(dv_path)

    emb_cols = [c for c in df_tr.columns if c.startswith("emb_")]
    X_tr = torch.tensor(df_tr[emb_cols].values, dtype=torch.float32)
    y_tr = torch.tensor(df_tr["label"].values, dtype=torch.float32)

    X_dv = torch.tensor(df_dv[emb_cols].values, dtype=torch.float32)
    y_dv = torch.tensor(df_dv["label"].values, dtype=torch.float32)

    return X_tr, y_tr, X_dv, y_dv, df_tr, df_dv


def optimize_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Finds threshold in [0.20, 0.80] that maximizes F1 score."""
    best_th = 0.50
    best_f1 = -1.0
    for th in np.linspace(0.20, 0.80, 61):
        preds = (probs >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return round(best_th, 4)


def train___DOM_ID___model(
    epochs: int = 160,
    lr: float = 0.002,
    weight_decay: float = 1e-2,
    hidden_dim: int = 64,
    save_model: bool = True
) -> Tuple[SemanticPrototypeClassifier, Dict[str, Any]]:
    print("=" * 85)
    print("STAGE 05: TRAINING CLASSIFIER FOR __DOM_ID__ ('__DOM_NAME__')")
    print("Features: 384-d dense embeddings (query) + SentenceTransformer prototypes (keys)")
    print("=" * 85)

    # 1. Load Data
    X_tr, y_tr, X_dv, y_dv, df_tr, df_dv = load_dataset()
    n_pos = int((y_tr == 1).sum().item())
    n_neg = int((y_tr == 0).sum().item())
    print(f"Loaded Datasets:")
    print(f"  • Train: {len(X_tr)} samples | Positives ('__DOM_NAME__'): {n_pos} | Negatives: {n_neg}")
    print(f"  • Dev  : {len(X_dv)} samples | Positives ('__DOM_NAME__'): {int((y_dv == 1).sum().item())}")

    # 2. Load Pre-Computed Text Prototype Key Vectors
    proto_payload = generate___DOM_ID___prototype_keys()
    prototype_keys = proto_payload["all_keys"]
    num_pos = proto_payload["num_pos"]
    num_neg = proto_payload["num_neg"]
    print(f"Loaded Prototype Key Vectors: shape {prototype_keys.shape} ({num_pos} Pos, {num_neg} Neg)")

    # 3. Initialize Model & Class-Balanced Loss
    torch.manual_seed(42)
    model = SemanticPrototypeClassifier(
        initial_prototype_embeddings=prototype_keys,
        num_pos=num_pos,
        num_neg=num_neg,
        hidden_dim=hidden_dim,
        initial_scale=10.0,
        dropout_rate=0.20
    )

    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training Loop
    print(f"\\nTraining for {epochs} epochs with AdamW (lr={lr}, weight_decay={weight_decay})...")
    start_time = time.time()
    loss_history = []

    best_dev_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model.forward_logits(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        # Periodic evaluation
        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                dv_probs = model.predict_proba(X_dv)
                dv_preds = (dv_probs >= 0.50).astype(int)
                y_dv_np = y_dv.numpy().astype(int)
                dv_f1 = f1_score(y_dv_np, dv_preds, zero_division=0)
                dv_acc = accuracy_score(y_dv_np, dv_preds)

                if dv_f1 > best_dev_f1:
                    best_dev_f1 = dv_f1
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                p_min = float(dv_probs.min())
                p_max = float(dv_probs.max())
                print(f"  [Epoch {epoch:3d}/{epochs}] Loss: {loss.item():.4f} | Scale: {model.scale.item():.2f} | Dev Acc: {dv_acc*100:.2f}% | Dev F1: {dv_f1:.4f} | Probs [min={p_min:.3f}, max={p_max:.3f}]")

    elapsed = time.time() - start_time
    print(f"✓ Training finished in {elapsed:.2f}s")

    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"✓ Loaded best checkpoint with Dev F1 = {best_dev_f1:.4f}")

    # 5. Evaluate on Dev
    model.eval()
    with torch.no_grad():
        tr_probs = model.predict_proba(X_tr)
        dv_probs = model.predict_proba(X_dv)

    opt_th = optimize_threshold(tr_probs, y_tr.numpy().astype(int))

    y_dv_np = y_dv.numpy().astype(int)
    preds_05 = (dv_probs >= 0.50).astype(int)
    preds_opt = (dv_probs >= opt_th).astype(int)

    acc_05 = accuracy_score(y_dv_np, preds_05)
    f1_05 = f1_score(y_dv_np, preds_05, zero_division=0)
    prec_05 = precision_score(y_dv_np, preds_05, zero_division=0)
    rec_05 = recall_score(y_dv_np, preds_05, zero_division=0)
    auc = roc_auc_score(y_dv_np, dv_probs) if len(np.unique(y_dv_np)) > 1 else 0.5
    cm_05 = confusion_matrix(y_dv_np, preds_05).tolist()

    acc_opt = accuracy_score(y_dv_np, preds_opt)
    f1_opt = f1_score(y_dv_np, preds_opt, zero_division=0)
    prec_opt = precision_score(y_dv_np, preds_opt, zero_division=0)
    rec_opt = recall_score(y_dv_np, preds_opt, zero_division=0)
    cm_opt = confusion_matrix(y_dv_np, preds_opt).tolist()

    print(f"\\nEvaluation Results on Dev Set ('__DOM_NAME__'):")
    print(f"  • Probability Dynamic Range: [{dv_probs.min():.4f}, {dv_probs.max():.4f}] (mean={dv_probs.mean():.4f})")
    print(f"  • Standard Cutoff (0.50): Acc = {acc_05*100:.2f}% | F1 = {f1_05:.4f} | Prec = {prec_05:.4f} | Rec = {rec_05:.4f} | AUC = {auc:.4f}")
    print(f"  • Tuned Cutoff ({opt_th:.3f}): Acc = {acc_opt*100:.2f}% | F1 = {f1_opt:.4f} | Prec = {prec_opt:.4f} | Rec = {rec_opt:.4f}")
    print(f"  • Confusion Matrix (0.50):")
    print(f"    [[TN={cm_05[0][0]}, FP={cm_05[0][1]}],")
    print(f"     [FN={cm_05[1][0]}, TP={cm_05[1][1]}]]")

    metrics = {
        "domain_id": "__DOM_ID__",
        "domain_name": "__DOM_NAME__",
        "train_samples": len(X_tr),
        "train_positives": n_pos,
        "dev_samples": len(X_dv),
        "dev_positives": int((y_dv == 1).sum().item()),
        "accuracy_default": float(acc_05),
        "f1_default": float(f1_05),
        "precision_default": float(prec_05),
        "recall_default": float(rec_05),
        "roc_auc": float(auc),
        "optimal_threshold": float(opt_th),
        "accuracy_opt": float(acc_opt),
        "f1_opt": float(f1_opt),
        "precision_opt": float(prec_opt),
        "recall_opt": float(rec_opt),
        "confusion_matrix_default": cm_05,
        "confusion_matrix_opt": cm_opt,
        "prob_min": float(dv_probs.min()),
        "prob_max": float(dv_probs.max()),
        "prob_mean": float(dv_probs.mean()),
        "trained_scale": float(model.scale.item()),
        "trained_bias": float(model.bias.item())
    }

    # 6. Save Model
    if save_model:
        models_dir = STAGE_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "__DOM_ID___classifier.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "prototype_keys": prototype_keys,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "hidden_dim": hidden_dim,
            "optimal_threshold": opt_th,
            "metrics": metrics,
            "loss_history": loss_history
        }
        torch.save(checkpoint, model_path)
        print(f"\\n✓ Saved model weights & checkpoint to: {model_path}")

    return model, metrics


if __name__ == "__main__":
    train___DOM_ID___model(save_model=True)
'''

MAIN_TEMPLATE = '''"""
Stage 05: __DOM_ID__ ('__DOM_NAME__') Evaluation Gateway & Benchmark Runner.

Evaluates the SemanticPrototypeClassifier trained on the balanced, summarized embeddings:
1. Loads the trained checkpoint from models/__DOM_ID___classifier.pt
2. Evaluates performance on held-out development set
3. Displays 15-sample prediction trace for '__DOM_NAME__'
4. Exports structured JSON metrics and comprehensive Markdown report
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

STAGE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = STAGE_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from experiments.stage_05.__DOM_ID__.model import SemanticPrototypeClassifier
    from experiments.stage_05.__DOM_ID__.train import train___DOM_ID___model
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(STAGE_DIR / "__DOM_ID__"))
    from model import SemanticPrototypeClassifier
    from train import train___DOM_ID___model


def load_or_train_model() -> Tuple[SemanticPrototypeClassifier, float, Dict[str, Any]]:
    models_dir = STAGE_DIR / "models"
    model_path = models_dir / "__DOM_ID___classifier.pt"

    if not model_path.exists():
        print(f"Checkpoint not found at {model_path}. Running training suite first...")
        model, metrics = train___DOM_ID___model(save_model=True)
        return model, metrics["optimal_threshold"], metrics

    print(f"Loading trained __DOM_ID__ Classifier from: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    prototype_keys = checkpoint["prototype_keys"]
    num_pos = checkpoint.get("num_pos", 4)
    num_neg = checkpoint.get("num_neg", 4)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    opt_th = checkpoint.get("optimal_threshold", 0.50)
    saved_metrics = checkpoint.get("metrics", {})

    model = SemanticPrototypeClassifier(
        initial_prototype_embeddings=prototype_keys,
        num_pos=num_pos,
        num_neg=num_neg,
        hidden_dim=hidden_dim
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded successfully. Scale: {model.scale.item():.2f} | Bias: {model.bias.item():.2f}")
    return model, opt_th, saved_metrics


def run___DOM_ID___evaluation():
    print("=" * 95)
    print("STAGE 05: __DOM_ID__ ('__DOM_NAME__') EVALUATION GATEWAY")
    print("Evaluating SemanticPrototypeClassifier on Balanced, Non-Truncated 384-d Embeddings")
    print("=" * 95)

    # 1. Load Dev Dataset
    data_dir = STAGE_DIR / "data"
    dev_path = data_dir / "dev___DOM_ID__.csv"

    if not dev_path.exists():
        raise FileNotFoundError(f"Processed dev dataset not found at {dev_path}.")

    df_dev = pd.read_csv(dev_path)
    emb_cols = [c for c in df_dev.columns if c.startswith("emb_")]
    X_dv = torch.tensor(df_dev[emb_cols].values, dtype=torch.float32)
    y_dv = df_dev["label"].values.astype(int)

    print(f"\\n[Step 1] Dev Dataset Loaded: {len(df_dev)} instances (Positives: {(y_dv == 1).sum()}, Negatives: {(y_dv == 0).sum()})")

    # 2. Load Model
    model, opt_th, train_metrics = load_or_train_model()

    # 3. Predict Probabilities & Labels
    dev_probs = model.predict_proba(X_dv)

    preds_05 = (dev_probs >= 0.50).astype(int)
    preds_opt = (dev_probs >= opt_th).astype(int)

    acc_05 = accuracy_score(y_dv, preds_05)
    f1_05 = f1_score(y_dv, preds_05, zero_division=0)
    prec_05 = precision_score(y_dv, preds_05, zero_division=0)
    rec_05 = recall_score(y_dv, preds_05, zero_division=0)
    auc = roc_auc_score(y_dv, dev_probs) if len(np.unique(y_dv)) > 1 else 0.5
    cm_05 = confusion_matrix(y_dv, preds_05).tolist()

    acc_opt = accuracy_score(y_dv, preds_opt)
    f1_opt = f1_score(y_dv, preds_opt, zero_division=0)
    prec_opt = precision_score(y_dv, preds_opt, zero_division=0)
    rec_opt = recall_score(y_dv, preds_opt, zero_division=0)
    cm_opt = confusion_matrix(y_dv, preds_opt).tolist()

    print(f"\\n[Step 2] Probability Distribution Analysis:")
    print(f"  • Min Probability : {dev_probs.min():.4f}")
    print(f"  • Max Probability : {dev_probs.max():.4f}")
    print(f"  • Mean Probability: {dev_probs.mean():.4f}")
    print(f"  • Std Probability : {dev_probs.std():.4f}")

    print(f"\\n[Step 3] Benchmark Performance on Held-Out Dev Set (__DOM_ID__):")
    print(f"  • Default Cutoff (0.50): Acc = {acc_05*100:.2f}% | F1 = {f1_05:.4f} | Prec = {prec_05:.4f} | Rec = {rec_05:.4f} | AUC = {auc:.4f}")
    print(f"  • Tuned Cutoff ({opt_th:.3f}) : Acc = {acc_opt*100:.2f}% | F1 = {f1_opt:.4f} | Prec = {prec_opt:.4f} | Rec = {rec_opt:.4f}")
    print(f"  • Confusion Matrix (0.50):")
    print(f"    [[TN={cm_05[0][0]}, FP={cm_05[0][1]}],")
    print(f"     [FN={cm_05[1][0]}, TP={cm_05[1][1]}]]")

    # 4. Detailed Sample Predictions Trace
    print(f"\\n[Step 4] Detailed Prediction Trace (15 Sample Instances on Dev Split):")
    print("-" * 90)
    print(f"{'ID':<14} | {'Prob(Pos)':<14} | {'Pred':<12} | {'True Label':<12} | Status")
    print("-" * 90)

    trace_records = []
    step = max(1, len(df_dev) // 15)
    sample_indices = list(range(0, len(df_dev), step))[:15]

    for idx in sample_indices:
        ex_id = str(df_dev.iloc[idx].get("example_id", f"EX-{idx:03d}"))
        prob = float(dev_probs[idx])
        pred_label = "__DOM_NAME__ (1)" if prob >= 0.50 else "Other (0)"
        true_label = "__DOM_NAME__ (1)" if y_dv[idx] == 1 else "Other (0)"
        is_correct = (prob >= 0.50) == (y_dv[idx] == 1)
        status = "CORRECT" if is_correct else "MISS"

        print(f"{ex_id:<14} | {prob:14.4f} | {pred_label:<12} | {true_label:<12} | {status}")

        trace_records.append({
            "example_id": ex_id,
            "prob": round(prob, 4),
            "prediction": pred_label,
            "ground_truth": true_label,
            "status": status
        })

    # 5. Export JSON Results
    results_dir = STAGE_DIR / "output"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "__DOM_ID___results.json"

    export_payload = {
        "domain_id": "__DOM_ID__",
        "domain_name": "__DOM_NAME__",
        "dev_instances": len(df_dev),
        "dev_positives": int((y_dv == 1).sum()),
        "accuracy_default": float(acc_05),
        "f1_default": float(f1_05),
        "precision_default": float(prec_05),
        "recall_default": float(rec_05),
        "roc_auc": float(auc),
        "optimal_threshold": float(opt_th),
        "accuracy_opt": float(acc_opt),
        "f1_opt": float(f1_opt),
        "precision_opt": float(prec_opt),
        "recall_opt": float(rec_opt),
        "confusion_matrix_default": cm_05,
        "confusion_matrix_opt": cm_opt,
        "prob_min": float(dev_probs.min()),
        "prob_max": float(dev_probs.max()),
        "prob_mean": float(dev_probs.mean()),
        "prob_std": float(dev_probs.std()),
        "trace_samples": trace_records
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\\n[Saved] Detailed evaluation metrics -> {json_path}")

    # 6. Export Markdown Report
    md_path = results_dir / "__DOM_ID___report.md"
    generate_markdown_report(export_payload, md_path)
    print(f"[Saved] Comprehensive Markdown report -> {md_path}")


def generate_markdown_report(data: Dict[str, Any], out_path: Path):
    cm = data["confusion_matrix_default"]
    cm_opt = data["confusion_matrix_opt"]

    content = f"""# Stage 05 Benchmark Report: __DOM_ID__ ('__DOM_NAME__')

## 1. Executive Summary

This report evaluates the **SemanticPrototypeClassifier** for **__DOM_ID__: '__DOM_NAME__'**.

- **Dataset**: Balanced summarized dataset using SemEval reference data plus high-precision severity augmentations.
  - Dev Instances: {data['dev_instances']} ({data['dev_positives']} Positive, {data['dev_instances'] - data['dev_positives']} Negative)
- **Zero-Truncation Embeddings**: 384-dimensional dense vectors generated from distilled 60–90 word summaries via `all-MiniLM-L6-v2`.
- **Domain Prototypes**: Pre-computed SentenceTransformer key vectors covering positive risk manifestations and negative safe anchors.
- **Architecture**:
  - `query_proj`: 384 -> 64 -> LayerNorm -> GELU -> Dropout
  - `key_proj`: 384 -> 64 -> LayerNorm
  - `fusion_head`: Attentively combines multi-prototype similarities with latent query features.
  - Temperature scaling (gamma = 10.0).

---

## 2. Benchmark Results on Held-Out Dev Set

| Evaluation Mode | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Default Cutoff (0.50)** | `0.50` | **{data['accuracy_default']*100:.2f}%** | {data['precision_default']:.4f} | {data['recall_default']:.4f} | **{data['f1_default']:.4f}** | **{data['roc_auc']:.4f}** | `TN={cm[0][0]}, FP={cm[0][1]} / FN={cm[1][0]}, TP={cm[1][1]}` |
| **Optimal F1 Cutoff** | `{data['optimal_threshold']:.3f}` | **{data['accuracy_opt']*100:.2f}%** | {data['precision_opt']:.4f} | {data['recall_opt']:.4f} | **{data['f1_opt']:.4f}** | {data['roc_auc']:.4f} | `TN={cm_opt[0][0]}, FP={cm_opt[0][1]} / FN={cm_opt[1][0]}, TP={cm_opt[1][1]}` |

---

## 3. Probability Dynamic Range Analysis

- **Minimum Probability**: `{data['prob_min']:.4f}`
- **Maximum Probability**: `{data['prob_max']:.4f}`
- **Mean Probability**   : `{data['prob_mean']:.4f}`
- **Std Deviation**     : `{data['prob_std']:.4f}`
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run___DOM_ID___evaluation()
'''


def scaffold_all():
    print("Scaffolding domain folders domain_03 through domain_11...")
    for dom_id, dom_name in DOMAINS:
        dom_dir = STAGE_DIR / dom_id
        dom_dir.mkdir(parents=True, exist_ok=True)

        # 1. model.py
        model_path = dom_dir / "model.py"
        model_code = MODEL_TEMPLATE.replace("__DOM_ID__", dom_id).replace("__DOM_NAME__", dom_name)
        model_path.write_text(model_code, encoding="utf-8")

        # 2. prototypes.py
        proto_path = dom_dir / "prototypes.py"
        proto_code = PROTOTYPES_TEMPLATE.replace("__DOM_ID__", dom_id).replace("__DOM_NAME__", dom_name)
        proto_path.write_text(proto_code, encoding="utf-8")

        # 3. train.py
        train_path = dom_dir / "train.py"
        train_code = TRAIN_TEMPLATE.replace("__DOM_ID__", dom_id).replace("__DOM_NAME__", dom_name)
        train_path.write_text(train_code, encoding="utf-8")

        # 4. main.py
        main_path = dom_dir / "main.py"
        main_code = MAIN_TEMPLATE.replace("__DOM_ID__", dom_id).replace("__DOM_NAME__", dom_name)
        main_path.write_text(main_code, encoding="utf-8")

        print(f"  ✓ Scaffolding complete for {dom_id} ('{dom_name}')")

    print("All domain directories successfully scaffolded!")


if __name__ == "__main__":
    scaffold_all()
