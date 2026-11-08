"""
Stage 05: domain_05 ('Operational / Service') Training Suite.

Trains the SemanticPrototypeClassifier for domain_05:
- Loads 384-d dense semantic embeddings from data/train_domain_05.csv and data/dev_domain_05.csv
- Loads pre-computed prototype key vectors (zero random noise)
- Uses class-weighted BCE loss with AdamW and weight decay
- Optimizes decision threshold on training set
- Persists trained model weights to experiments/stage_05/models/domain_05_classifier.pt
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
    from experiments.stage_05.domain_05.model import SemanticPrototypeClassifier
    from experiments.stage_05.domain_05.prototypes import generate_domain_05_prototype_keys
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(STAGE_DIR / "domain_05"))
    from model import SemanticPrototypeClassifier
    from prototypes import generate_domain_05_prototype_keys


def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    data_dir = STAGE_DIR / "data"
    tr_path = data_dir / "train_domain_05.csv"
    dv_path = data_dir / "dev_domain_05.csv"

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


def train_domain_05_model(
    epochs: int = 160,
    lr: float = 0.002,
    weight_decay: float = 1e-2,
    hidden_dim: int = 64,
    save_model: bool = True
) -> Tuple[SemanticPrototypeClassifier, Dict[str, Any]]:
    print("=" * 85)
    print("STAGE 05: TRAINING CLASSIFIER FOR domain_05 ('Operational / Service')")
    print("Features: 384-d dense embeddings (query) + SentenceTransformer prototypes (keys)")
    print("=" * 85)

    # 1. Load Data
    X_tr, y_tr, X_dv, y_dv, df_tr, df_dv = load_dataset()
    n_pos = int((y_tr == 1).sum().item())
    n_neg = int((y_tr == 0).sum().item())
    print(f"Loaded Datasets:")
    print(f"  • Train: {len(X_tr)} samples | Positives ('Operational / Service'): {n_pos} | Negatives: {n_neg}")
    print(f"  • Dev  : {len(X_dv)} samples | Positives ('Operational / Service'): {int((y_dv == 1).sum().item())}")

    # 2. Load Pre-Computed Text Prototype Key Vectors
    proto_payload = generate_domain_05_prototype_keys()
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
    print(f"\nTraining for {epochs} epochs with AdamW (lr={lr}, weight_decay={weight_decay})...")
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

    print(f"\nEvaluation Results on Dev Set ('Operational / Service'):")
    print(f"  • Probability Dynamic Range: [{dv_probs.min():.4f}, {dv_probs.max():.4f}] (mean={dv_probs.mean():.4f})")
    print(f"  • Standard Cutoff (0.50): Acc = {acc_05*100:.2f}% | F1 = {f1_05:.4f} | Prec = {prec_05:.4f} | Rec = {rec_05:.4f} | AUC = {auc:.4f}")
    print(f"  • Tuned Cutoff ({opt_th:.3f}): Acc = {acc_opt*100:.2f}% | F1 = {f1_opt:.4f} | Prec = {prec_opt:.4f} | Rec = {rec_opt:.4f}")
    print(f"  • Confusion Matrix (0.50):")
    print(f"    [[TN={cm_05[0][0]}, FP={cm_05[0][1]}],")
    print(f"     [FN={cm_05[1][0]}, TP={cm_05[1][1]}]]")

    metrics = {
        "domain_id": "domain_05",
        "domain_name": "Operational / Service",
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
        model_path = models_dir / "domain_05_classifier.pt"

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
        print(f"\n✓ Saved model weights & checkpoint to: {model_path}")

    return model, metrics


if __name__ == "__main__":
    train_domain_05_model(save_model=True)
