"""
Stage 07 Training Pipeline: Multi-Section Embedding Optimizer for Domain 01.

Trains Stage07DomainClassifier on SemEval training set with:
  - Binary Cross-Entropy with Logits loss
  - AdamW optimizer with Cosine Annealing learning rate schedule
  - Tracking validation Loss, Exact Accuracy, F1, and ROC-AUC
  - Checkpointing the champion model to experiments/stage_07/models/best_domain_01_model.pt
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.stage_07.model import Stage07DomainClassifier

STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
MODELS_DIR = STAGE_DIR / "models"
OUTPUT_DIR = STAGE_DIR / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOTYPES_PT = ROOT_DIR / "experiments" / "stage_05" / "artifacts" / "domain_01_prototype_keys.pt"


def load_dataset(pt_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(pt_path, weights_only=False)
    sections = data["sections_tensor"]  # (N, 4, 384)
    labels = data["labels"]             # (N,)
    return sections, labels


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            probs, logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy().astype(int))

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(auc)
    }


def train_model(
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cpu"
) -> Dict[str, Any]:
    print("=" * 70)
    print("Stage 07: Training Multi-Section Optimizer + Prototype Classifier")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 1. Load Pre-computed Domain 01 Prototypes
    if not PROTOTYPES_PT.exists():
        raise FileNotFoundError(f"Prototype keys not found at {PROTOTYPES_PT}")
    proto_data = torch.load(PROTOTYPES_PT, weights_only=False)
    proto_keys = proto_data["all_keys"].to(device)  # shape (8, 384)
    num_pos = proto_data.get("num_pos", 4)
    print(f"Loaded {proto_keys.shape[0]} prototype keys ({num_pos} Positive Safe, {proto_keys.shape[0] - num_pos} Negative Harmful)")

    # 2. Load Train and Dev Data
    train_pt = DATA_DIR / "train_sections.pt"
    dev_pt = DATA_DIR / "dev_sections.pt"
    if not train_pt.exists() or not dev_pt.exists():
        raise FileNotFoundError("Run prepare_dataset.py first to extract and cache section tensors.")

    x_train, y_train = load_dataset(train_pt)
    x_dev, y_dev = load_dataset(dev_pt)

    print(f"Train samples: {len(x_train)} (Safe: {int(y_train.sum().item())}, Harmful: {int((1-y_train).sum().item())})")
    print(f"Dev samples:   {len(x_dev)} (Safe: {int(y_dev.sum().item())}, Harmful: {int((1-y_dev).sum().item())})")

    # Positive class weighting for balanced gradient
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=batch_size, shuffle=False)

    # 3. Model Initialization
    model = Stage07DomainClassifier(
        initial_prototype_keys=proto_keys,
        num_pos_prototypes=num_pos,
        embed_dim=384,
        hidden_dim=384,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_dev_f1 = 0.0
    best_checkpoint_path = MODELS_DIR / "best_domain_01_model.pt"
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            _, logits, _ = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_b)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Evaluate on Train and Dev
        train_metrics = evaluate(model, train_loader, criterion, device)
        dev_metrics = evaluate(model, dev_loader, criterion, device)

        is_best = dev_metrics["f1"] > best_dev_f1
        if is_best:
            best_dev_f1 = dev_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "prototype_keys": proto_keys.cpu(),
                "dev_metrics": dev_metrics,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay
                }
            }, best_checkpoint_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "dev_loss": dev_metrics["loss"],
            "dev_acc": dev_metrics["accuracy"],
            "dev_f1": dev_metrics["f1"],
            "dev_auc": dev_metrics["roc_auc"],
            "is_best": is_best
        }
        history.append(epoch_record)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']*100:.1f}% | "
                  f"Dev Loss: {dev_metrics['loss']:.4f} Acc: {dev_metrics['accuracy']*100:.1f}% "
                  f"F1: {dev_metrics['f1']:.4f} AUC: {dev_metrics['roc_auc']:.4f}{star}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best checkpoint saved to {best_checkpoint_path}")

    # Save history
    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_dev_f1": best_dev_f1, "history": history}


def main():
    train_model(epochs=40, batch_size=32, lr=1e-3)


if __name__ == "__main__":
    main()

