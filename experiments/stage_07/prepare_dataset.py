"""
Stage 07: Dataset Preparation for Domain 01 (No Material Harm).

Extracts structured interaction sections ([query, action, thought, observation]),
chunks long sections, encodes them in bulk with SentenceTransformer ('all-MiniLM-L6-v2'),
and caches stacked section embeddings:
  - experiments/stage_07/data/train_sections.pt: shape (568, 4, 384)
  - experiments/stage_07/data/dev_sections.pt: shape (143, 4, 384)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.summerizer import (
    get_embedder,
    batch_extract_and_embed_sections
)

STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = ROOT_DIR / "data" / "agentrisk-data-2027" / "data"
TRAIN_JSONL = RAW_DATA_DIR / "train.jsonl"
DEV_JSONL = RAW_DATA_DIR / "dev.jsonl"


def process_split(
    jsonl_path: Path,
    out_pt_path: Path,
    split_name: str,
    embedder=None
) -> Dict[str, Any]:
    print(f"\nProcessing {split_name} split from {jsonl_path}...")
    if embedder is None:
        embedder = get_embedder()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    start_time = time.time()

    # Extract labels, metadata, and example IDs
    example_ids = [r.get("example_id", f"{split_name}_{i:04d}") for i, r in enumerate(records)]
    labels = [1 if "No Material Harm" in r.get("risk_domains", []) else 0 for r in records]
    severities = [r.get("risk_severity", 1) for r in records]
    domains_list = [r.get("risk_domains", []) for r in records]

    # Vectorized bulk chunk extraction and encoding
    print(f"Extracting chunks and bulk encoding {total} {split_name} interactions...")
    stacked_sections, fused_summaries = batch_extract_and_embed_sections(
        records=records,
        embedder=embedder,
        batch_size=64,
        max_chunk_words=120,
        overlap_words=20,
        show_progress=True
    )

    elapsed = time.time() - start_time
    print(f"Encoded {total} samples in {elapsed:.2f}s ({total / elapsed:.1f} samples/s)")

    sections_tensor = torch.tensor(stacked_sections, dtype=torch.float32)
    fused_tensor = torch.tensor(fused_summaries, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    data_payload = {
        "example_ids": example_ids,
        "sections_tensor": sections_tensor,  # shape: (N, 4, 384)
        "fused_tensor": fused_tensor,        # shape: (N, 384)
        "labels": labels_tensor,            # shape: (N,)
        "severities": severities,
        "domains_list": domains_list,
        "section_names": ["query", "action", "thought", "observation"],
        "num_samples": total,
        "num_safe": sum(labels),
        "num_harmful": total - sum(labels)
    }

    torch.save(data_payload, out_pt_path)
    print(f"Saved {split_name} cached tensor to {out_pt_path}")
    print(f"  Sections shape: {sections_tensor.shape}")
    print(f"  Class balance: Safe (1) = {sum(labels)}, Harmful (0) = {total - sum(labels)}")
    return data_payload


def main():
    print("=" * 70)
    print("Stage 07: Multi-Section Chunked Dataset Preparation (Vectorized)")
    print("=" * 70)

    train_out = DATA_DIR / "train_sections.pt"
    dev_out = DATA_DIR / "dev_sections.pt"

    embedder = get_embedder()
    process_split(TRAIN_JSONL, train_out, "train", embedder=embedder)
    process_split(DEV_JSONL, dev_out, "dev", embedder=embedder)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
