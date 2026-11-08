"""
Stage 05: Domain 02 ('Privacy / Confidentiality') Training Suite.

Trains the SemanticPrototypeClassifier for Privacy & Confidentiality:
- Loads 384-dimensional dense semantic embeddings from data/train_domain_02.csv
- Loads pre-computed prototype key vectors (zero random noise)
- Uses class-weighted BCE loss with AdamW and weight decay
- Optimizes decision threshold on training set
- Persists trained model weights to experiments/stage_05/models/domain_02_classifier.pt
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
    from experiments.stage_05.domain_02.model import SemanticPrototypeClassifier
    from experiments.stage_05.domain_02.prototypes import generate_domain_02_prototype_keys
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(STAGE_DIR / "domain_02"))
    from model import SemanticPrototypeClassifier
    from prototypes import generate_domain_02_prototype_keys


def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    data_dir = STAGE_DIR / "data"
    tr_path = data_dir / "train_domain_02.csv"
    dv_path = data_dir / "dev_domain_02.csv"

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


def train_domain_02_model(
    epochs: int = 160,
    lr: float = 0.002,
    weight_decay: float = 1e-2,
    hidden_dim: int = 64,
    save_model: bool = True
) -> Tuple[SemanticPrototypeClassifier, Dict[str, Any]]:
    print("=" * 85)
    print("STAGE 05: TRAINING CLASSIFIER FOR DOMAIN 02 ('PRIVACY / CONFIDENTIALITY')")
    print("Features: 384-d dense embeddings (query) + SentenceTransformer prototypes (keys)")
    print("=" * 85)

    # 1. Load Data
    X_tr, y_tr, X_dv, y_dv, df_tr, df_dv = load_dataset()
    n_pos = int((y_tr == 1).sum().item())
    n_neg = int((y_tr == 0).sum().item())
    print(f"Loaded Datasets:")
    print(f"  • Train: {len(X_tr)} samples | Positives ('Privacy / Confidentiality'): {n_pos} | Negatives: {n_neg}")
    print(f"  • Dev  : {len(X_dv)} samples | Positives ('Privacy / Confidentiality'): {int((y_dv == 1).sum().item())}")

    # 2. Load Pre-Computed Text Prototype Key Vectors
    proto_payload = generate_domain_02_prototype_keys()
    prototype_keys = proto_payload["all_keys"]
    num_pos = proto_payload["num_pos"]
    num_neg = proto_payload["num_neg"]
    print(f"Loaded Prototype Key Vectors: shape {prototype_keys.shape} ({num_pos} Privacy, {num_neg} Non-Privacy)")

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

                print(
                    f"  [Epoch {epoch:3d}/{epochs}] Loss: {loss.item():.4f} | "
                    f"Scale: {model.scale.item():.2f} | "
                    f"Dev Acc: {dv_acc*100:5.2f}% | Dev F1: {dv_f1:.4f} | "
                    f"Probs [min={dv_probs.min():.3f}, max={dv_probs.max():.3f}]"
                )

    train_time = time.time() - start_time
    print(f"✓ Training finished in {train_time:.2f}s")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"✓ Loaded best checkpoint with Dev F1 = {best_dev_f1:.4f}")

    # 5. Evaluate Probability Spread on Dev Set
    model.eval()
    dev_probs = model.predict_proba(X_dv)
    y_dv_np = y_dv.numpy().astype(int)
    y_tr_np = y_tr.numpy().astype(int)

    # Optimize threshold on Train
    tr_probs = model.predict_proba(X_tr)
    opt_th = optimize_threshold(tr_probs, y_tr_np)

    # Predictions under default 0.50 and optimal threshold
    preds_05 = (dev_probs >= 0.50).astype(int)
    preds_opt = (dev_probs >= opt_th).astype(int)

    acc_05 = accuracy_score(y_dv_np, preds_05)
    f1_05 = f1_score(y_dv_np, preds_05, zero_division=0)
    prec_05 = precision_score(y_dv_np, preds_05, zero_division=0)
    rec_05 = recall_score(y_dv_np, preds_05, zero_division=0)

    acc_opt = accuracy_score(y_dv_np, preds_opt)
    f1_opt = f1_score(y_dv_np, preds_opt, zero_division=0)
    prec_opt = precision_score(y_dv_np, preds_opt, zero_division=0)
    rec_opt = recall_score(y_dv_np, preds_opt, zero_division=0)
    auc = roc_auc_score(y_dv_np, dev_probs)
    cm_05 = confusion_matrix(y_dv_np, preds_05, labels=[0, 1])
    cm_opt = confusion_matrix(y_dv_np, preds_opt, labels=[0, 1])

    metrics = {
        "domain": "Domain 02: Privacy / Confidentiality",
        "model_name": "SemanticPrototypeClassifier",
        "epochs": epochs,
        "learned_scale": float(model.scale.item()),
        "learned_bias": float(model.bias.item()),
        "prob_min": float(dev_probs.min()),
        "prob_max": float(dev_probs.max()),
        "prob_mean": float(dev_probs.mean()),
        "default_threshold": 0.50,
        "optimal_threshold": opt_th,
        "accuracy_05": float(acc_05),
        "f1_05": float(f1_05),
        "precision_05": float(prec_05),
        "recall_05": float(rec_05),
        "accuracy_opt": float(acc_opt),
        "f1_opt": float(f1_opt),
        "precision_opt": float(prec_opt),
        "recall_opt": float(rec_opt),
        "roc_auc": float(auc),
        "confusion_matrix_05": cm_05.tolist(),
        "confusion_matrix_opt": cm_opt.tolist(),
        "final_loss": float(loss_history[-1])
    }

    print("\nEvaluation Results on Dev Set (Privacy / Confidentiality):")
    print(f"  • Probability Dynamic Range: [{metrics['prob_min']:.4f}, {metrics['prob_max']:.4f}] (mean={metrics['prob_mean']:.4f})")
    print(f"  • Standard Cutoff (0.50): Acc = {acc_05*100:.2f}% | F1 = {f1_05:.4f} | Prec = {prec_05:.4f} | Rec = {rec_05:.4f} | AUC = {auc:.4f}")
    print(f"  • Tuned Cutoff ({opt_th:.3f}): Acc = {acc_opt*100:.2f}% | F1 = {f1_opt:.4f} | Prec = {prec_opt:.4f} | Rec = {rec_opt:.4f}")
    print(f"  • Confusion Matrix (0.50):\n    [[TN={cm_05[0][0]}, FP={cm_05[0][1]}],\n     [FN={cm_05[1][0]}, TP={cm_05[1][1]}]]")

    # 6. Save Model Checkpoint
    if save_model:
        models_dir = STAGE_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = models_dir / "domain_02_classifier.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "prototype_keys": prototype_keys,
                "num_pos": num_pos,
                "num_neg": num_neg,
                "hidden_dim": hidden_dim,
                "metrics": metrics,
                "optimal_threshold": opt_th
            },
            model_save_path
        )
        print(f"\n✓ Saved model weights & checkpoint to: {model_save_path}")

    return model, metrics


if __name__ == "__main__":
    train_domain_02_model()
