"""
Stage 08 Training Pipeline: 5-Channel Cross-Attentive Prototype Classifier.

Trains Stage08OptimizedDomainClassifier on SemEval training set with:
  - Combined Binary Cross-Entropy + Contrastive Prototype Margin Loss
  - AdamW optimizer with Cosine Annealing learning rate schedule
  - Dynamic optimal threshold calibration for maximal F1 score
  - Checkpointing champion model to experiments/stage_08/models/best_domain_01_stage08.pt
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

from experiments.stage_08.model import Stage08OptimizedDomainClassifier

STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
MODELS_DIR = STAGE_DIR / "models"
OUTPUT_DIR = STAGE_DIR / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOTYPES_PT = ROOT_DIR / "experiments" / "stage_05" / "artifacts" / "domain_01_prototype_keys.pt"


def load_dataset(pt_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(pt_path, weights_only=False)
    channels = data["channels_tensor"]  # (N, 5, 384)
    labels = data["labels"]             # (N,)
    return channels, labels


class PrototypeMarginLoss(nn.Module):
    """
    Penalizes samples where the positive prototype similarity is not strictly
    separated from the negative prototype similarity by at least margin gamma.
    """

    def __init__(self, margin: float = 0.20, weight: float = 0.15):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        pos_sim_mean: torch.Tensor,
        neg_sim_mean: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # For targets == 1 (Safe): want pos_sim - neg_sim >= margin
        diff_safe = pos_sim_mean - neg_sim_mean
        loss_safe = torch.relu(self.margin - diff_safe) * targets

        # For targets == 0 (Harmful): want neg_sim - pos_sim >= margin
        diff_harm = neg_sim_mean - pos_sim_mean
        loss_harm = torch.relu(self.margin - diff_harm) * (1.0 - targets)

        return self.weight * (loss_safe.mean() + loss_harm.mean())


def evaluate_with_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    margin_loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.50
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in dataloader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            probs, logits, meta = model(x_b)
            bce = criterion(logits, y_b)
            p_margin = margin_loss_fn(meta["pos_sim_mean"], meta["neg_sim_mean"], y_b)
            loss = bce + p_margin
            total_loss += loss.item() * len(y_b)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy().astype(int))

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
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
        "macro_f1": float(f1_mac),
        "roc_auc": float(auc)
    }


def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.linspace(0.30, 0.70, 41):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh, best_f1


def train_stage08(
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cpu"
) -> Dict[str, Any]:
    print("=" * 70)
    print("Stage 08: Training 5-Channel Cross-Attentive Prototype Classifier")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Device: {device}")

    proto_data = torch.load(PROTOTYPES_PT, weights_only=False)
    proto_keys = proto_data["all_keys"].to(device)
    num_pos = proto_data.get("num_pos", 4)

    train_pt = DATA_DIR / "train_5channels.pt"
    dev_pt = DATA_DIR / "dev_5channels.pt"
    if not train_pt.exists() or not dev_pt.exists():
        raise FileNotFoundError("Run prepare_dataset.py in stage_08 first.")

    x_train, y_train = load_dataset(train_pt)
    x_dev, y_dev = load_dataset(dev_pt)

    print(f"Train: {len(x_train)} samples | Dev: {len(x_dev)} samples")

    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    margin_criterion = PrototypeMarginLoss(margin=0.20, weight=0.15)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=batch_size, shuffle=False)

    model = Stage08OptimizedDomainClassifier(
        initial_prototype_keys=proto_keys,
        num_pos_prototypes=num_pos,
        embed_dim=384,
        hidden_dim=384,
        num_channels=5,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_dev_acc = 0.0
    best_checkpoint_path = MODELS_DIR / "best_domain_01_stage08.pt"
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            _, logits, meta = model(x_b)
            loss_bce = bce_criterion(logits, y_b)
            loss_margin = margin_criterion(meta["pos_sim_mean"], meta["neg_sim_mean"], y_b)
            loss = loss_bce + loss_margin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_b)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        train_metrics = evaluate_with_threshold(model, train_loader, bce_criterion, margin_criterion, device)
        dev_metrics = evaluate_with_threshold(model, dev_loader, bce_criterion, margin_criterion, device)

        is_best = dev_metrics["accuracy"] > best_dev_acc
        if is_best:
            best_dev_acc = dev_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
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
            "dev_loss": dev_metrics["loss"],
            "dev_acc": dev_metrics["accuracy"],
            "dev_f1": dev_metrics["f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_auc": dev_metrics["roc_auc"],
            "is_best": is_best
        }
        history.append(epoch_record)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Acc: {train_metrics['accuracy']*100:.1f}% | "
                  f"Dev Acc: {dev_metrics['accuracy']*100:.1f}% "
                  f"Macro F1: {dev_metrics['macro_f1']:.4f} AUC: {dev_metrics['roc_auc']:.4f}{star}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best checkpoint saved to {best_checkpoint_path}")

    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_dev_acc": best_dev_acc, "history": history}


def main():
    train_stage08(epochs=40, batch_size=32, lr=1e-3)


if __name__ == "__main__":
    main()

