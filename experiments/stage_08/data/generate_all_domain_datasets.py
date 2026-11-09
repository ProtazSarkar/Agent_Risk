"""
Stage 08: Master Multi-Domain Dataset Generator for All 11 SemEval Domains.

Combines the authentic SemEval 5-channel interaction tensors with rich,
SemEval-grounded synthetic interactions to create balanced, high-variance
training sets for all 11 risk domains:
  domain_01: No Material Harm
  domain_02: Privacy / Confidentiality
  domain_03: Security / System Integrity
  domain_04: Financial / Assets
  domain_05: Operational / Service
  domain_06: Physical / Health
  domain_07: Legal / Compliance
  domain_08: Reputational / Social / Psychological
  domain_09: Discrimination / Fairness
  domain_10: Environmental
  domain_11: Information Quality

Caches:
  experiments/stage_08/data/train_domain_XX_5ch.pt
  experiments/stage_08/data/dev_domain_XX_5ch.pt
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.summerizer import get_embedder
from experiments.stage_08.prepare_dataset import batch_extract_5channels
from experiments.stage_08.data.domain_interaction_augmentations import DOMAIN_AUGMENTATIONS

STAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = STAGE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_MAPPING = {
    "domain_01": "No Material Harm",
    "domain_02": "Privacy / Confidentiality",
    "domain_03": "Security / System Integrity",
    "domain_04": "Financial / Assets",
    "domain_05": "Operational / Service",
    "domain_06": "Physical / Health",
    "domain_07": "Legal / Compliance",
    "domain_08": "Reputational / Social / Psychological",
    "domain_09": "Discrimination / Fairness",
    "domain_10": "Environmental",
    "domain_11": "Information Quality"
}


def main():
    print("=" * 70)
    print("Stage 08: Generating Balanced 5-Channel Datasets for All 11 Domains")
    print("=" * 70)

    base_train_pt = DATA_DIR / "train_5channels.pt"
    base_dev_pt = DATA_DIR / "dev_5channels.pt"

    if not base_train_pt.exists() or not base_dev_pt.exists():
        raise FileNotFoundError("Base 5-channel tensors not found. Run prepare_dataset.py first.")

    train_base = torch.load(base_train_pt, weights_only=False)
    dev_base = torch.load(base_dev_pt, weights_only=False)

    x_train_raw = train_base["channels_tensor"]  # (568, 5, 384)
    domains_train = train_base["domains_list"]   # list of lists
    x_dev_raw = dev_base["channels_tensor"]      # (143, 5, 384)
    domains_dev = dev_base["domains_list"]       # list of lists

    embedder = get_embedder()

    # Pre-encode synthetic augmentations for each domain
    print("\nEncoding synthetic augmentations for minority domains...")
    aug_tensors: Dict[str, torch.Tensor] = {}
    for d_id, aug_list in DOMAIN_AUGMENTATIONS.items():
        if aug_list:
            stacked = batch_extract_5channels(aug_list, embedder=embedder, batch_size=32)
            aug_tensors[d_id] = torch.tensor(stacked, dtype=torch.float32)
            print(f"  {d_id}: encoded {len(aug_list)} augmentations -> {stacked.shape}")

    # Process all 11 domains
    summary = []
    for d_id, d_name in DOMAIN_MAPPING.items():
        print(f"\nProcessing {d_id} ({d_name})...")

        # Dev labels: 1 if domain present, else 0
        y_dev = torch.tensor([
            1 if d_name in d_list else 0
            for d_list in domains_dev
        ], dtype=torch.float32)

        # Train labels from raw SemEval:
        raw_pos_indices = [i for i, d_list in enumerate(domains_train) if d_name in d_list]
        raw_neg_indices = [i for i, d_list in enumerate(domains_train) if d_name not in d_list]

        pos_chunks = [x_train_raw[raw_pos_indices]]
        # Include synthetic positive augmentations
        if d_id in aug_tensors and len(aug_tensors[d_id]) > 0:
            # Synthetic instances with this domain are positive
            pos_chunks.append(aug_tensors[d_id])

        x_pos = torch.cat(pos_chunks, dim=0)

        # Negative chunks: sample from raw negative interactions
        # If positives are few, balance negatives to a healthy ratio (e.g. 1.5x to 2x)
        num_pos = len(x_pos)
        target_neg_count = max(len(raw_pos_indices) * 2, min(len(raw_neg_indices), max(100, num_pos * 2)))

        np.random.seed(42)
        sampled_neg_idx = np.random.choice(raw_neg_indices, size=min(target_neg_count, len(raw_neg_indices)), replace=False)
        x_neg = x_train_raw[sampled_neg_idx]

        # Combine
        x_train_domain = torch.cat([x_pos, x_neg], dim=0)
        y_train_domain = torch.cat([
            torch.ones(len(x_pos), dtype=torch.float32),
            torch.zeros(len(x_neg), dtype=torch.float32)
        ], dim=0)

        # Shuffle
        perm = torch.randperm(len(x_train_domain))
        x_train_domain = x_train_domain[perm]
        y_train_domain = y_train_domain[perm]

        # Save domain tensors
        train_out = DATA_DIR / f"train_{d_id}_5ch.pt"
        dev_out = DATA_DIR / f"dev_{d_id}_5ch.pt"

        torch.save({
            "channels_tensor": x_train_domain,
            "labels": y_train_domain,
            "domain_id": d_id,
            "domain_name": d_name,
            "num_pos": len(x_pos),
            "num_neg": len(x_neg)
        }, train_out)

        torch.save({
            "channels_tensor": x_dev_raw,
            "labels": y_dev,
            "domain_id": d_id,
            "domain_name": d_name,
            "num_pos": int(y_dev.sum().item()),
            "num_neg": int((1 - y_dev).sum().item())
        }, dev_out)

        rec = {
            "domain_id": d_id,
            "domain_name": d_name,
            "train_total": len(x_train_domain),
            "train_pos": len(x_pos),
            "train_neg": len(x_neg),
            "dev_total": len(x_dev_raw),
            "dev_pos": int(y_dev.sum().item()),
            "dev_neg": int((1 - y_dev).sum().item())
        }
        summary.append(rec)
        print(f"  Train: {len(x_train_domain)} (Pos: {len(x_pos)}, Neg: {len(x_neg)}) | Dev: {len(x_dev_raw)} (Pos: {int(y_dev.sum().item())}, Neg: {int((1-y_dev).sum().item())})")

    print("\n" + "=" * 70)
    print("All 11 Domain Datasets Generated Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
