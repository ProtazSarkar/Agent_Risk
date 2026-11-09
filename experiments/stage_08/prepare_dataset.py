"""
Stage 08: 5-Channel Interaction Dataset Preparation for Domain 01.

Disentangles agent interactions into 5 dedicated semantic channels:
  Channel 0: query            - User prompt & instructions [USER]
  Channel 1: thought          - Chain-of-thought internal reasoning [THOUGHT]
  Channel 2: context_actions  - Historical action execution sequence [ACTION]
  Channel 3: observations     - Environment & tool outputs [OBSERVATION]
  Channel 4: proposed_action  - Pending next action about to be executed

Chunks long sections, applies SentenceTransformer ('all-MiniLM-L6-v2') in bulk batches,
and caches:
  - experiments/stage_08/data/train_5channels.pt: shape (568, 5, 384)
  - experiments/stage_08/data/dev_5channels.pt: shape (143, 5, 384)
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.summerizer import get_embedder, chunk_text

STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = ROOT_DIR / "data" / "agentrisk-data-2027" / "data"
TRAIN_JSONL = RAW_DATA_DIR / "train.jsonl"
DEV_JSONL = RAW_DATA_DIR / "dev.jsonl"

CHANNELS = ["query", "thought", "context_actions", "observations", "proposed_action"]


def extract_5_channels(formatted_interaction: str, proposed_action: str) -> Dict[str, str]:
    text = formatted_interaction or ""
    sections: Dict[str, List[str]] = {
        "query": [],
        "thought": [],
        "context_actions": [],
        "observations": []
    }
    tag_map = {
        "USER": "query",
        "THOUGHT": "thought",
        "ACTION": "context_actions",
        "OBSERVATION": "observations"
    }

    parts = re.split(r'Step \d+\.\s*\[(USER|THOUGHT|ACTION|OBSERVATION)\]:\s*', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            tag = parts[i].upper()
            content = parts[i + 1].strip()
            if tag in tag_map and content:
                sections[tag_map[tag]].append(content)
    else:
        cur_tag = "query"
        for line in text.split("\n"):
            m = re.match(r'\[(USER|THOUGHT|ACTION|OBSERVATION)\]:\s*(.*)', line, re.IGNORECASE)
            if m:
                cur_tag = tag_map.get(m.group(1).upper(), "query")
                rem = m.group(2).strip()
                if rem:
                    sections[cur_tag].append(rem)
            elif line.strip():
                sections[cur_tag].append(line.strip())

    return {
        "query": " \n".join(sections["query"]).strip(),
        "thought": " \n".join(sections["thought"]).strip(),
        "context_actions": " \n".join(sections["context_actions"]).strip(),
        "observations": " \n".join(sections["observations"]).strip(),
        "proposed_action": (proposed_action or "").strip()
    }


def batch_extract_5channels(
    records: List[Dict[str, Any]],
    embedder=None,
    batch_size: int = 64,
    max_chunk_words: int = 120,
    overlap_words: int = 20
) -> np.ndarray:
    if embedder is None:
        embedder = get_embedder()

    total = len(records)
    all_chunks: List[str] = []
    mapping: List[Tuple[int, int, int, int]] = []

    for rec_idx, rec in enumerate(records):
        fi = rec.get("formatted_interaction", "")
        pa = rec.get("proposed_action", "")
        ch_dict = extract_5_channels(fi, pa)
        for ch_idx, ch_name in enumerate(CHANNELS):
            txt = ch_dict[ch_name]
            chunks = chunk_text(txt, max_chunk_words=max_chunk_words, overlap_words=overlap_words)
            c_start = len(all_chunks)
            all_chunks.extend(chunks)
            c_end = len(all_chunks)
            mapping.append((rec_idx, ch_idx, c_start, c_end))

    print(f"  Encoding {len(all_chunks)} chunks across {total} samples in batches...")
    if all_chunks:
        chunk_embs = embedder.encode(
            all_chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    else:
        chunk_embs = np.zeros((0, 384), dtype=np.float32)

    stacked_channels = np.zeros((total, 5, 384), dtype=np.float32)
    for rec_idx, ch_idx, c_start, c_end in mapping:
        if c_end > c_start:
            ch_mean = np.mean(chunk_embs[c_start:c_end], axis=0)
            norm = np.linalg.norm(ch_mean)
            if norm > 1e-9:
                ch_mean = ch_mean / norm
            stacked_channels[rec_idx, ch_idx] = ch_mean

    return stacked_channels


def process_split(jsonl_path: Path, out_path: Path, split_name: str, embedder=None):
    print(f"\nProcessing {split_name} split from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    start_time = time.time()

    example_ids = [r.get("example_id", f"{split_name}_{i:04d}") for i, r in enumerate(records)]
    labels = [1 if "No Material Harm" in r.get("risk_domains", []) else 0 for r in records]
    severities = [r.get("risk_severity", 1) for r in records]
    domains_list = [r.get("risk_domains", []) for r in records]

    stacked_channels = batch_extract_5channels(
        records=records,
        embedder=embedder,
        batch_size=64
    )

    elapsed = time.time() - start_time
    print(f"Encoded {total} samples into 5 channels in {elapsed:.2f}s ({total/elapsed:.1f} samples/s)")

    channels_tensor = torch.tensor(stacked_channels, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    data_payload = {
        "example_ids": example_ids,
        "channels_tensor": channels_tensor,  # (N, 5, 384)
        "labels": labels_tensor,            # (N,)
        "severities": severities,
        "domains_list": domains_list,
        "channel_names": CHANNELS,
        "num_samples": total,
        "num_safe": sum(labels),
        "num_harmful": total - sum(labels)
    }

    torch.save(data_payload, out_path)
    print(f"Saved {split_name} 5-channel tensor to {out_path}")
    print(f"  Shape: {channels_tensor.shape} (Channels: {CHANNELS})")
    print(f"  Safe: {sum(labels)}, Harmful: {total - sum(labels)}")


def main():
    print("=" * 70)
    print("Stage 08: 5-Channel Disentangled Dataset Preparation")
    print("=" * 70)

    train_out = DATA_DIR / "train_5channels.pt"
    dev_out = DATA_DIR / "dev_5channels.pt"

    embedder = get_embedder()
    process_split(TRAIN_JSONL, train_out, "train", embedder=embedder)
    process_split(DEV_JSONL, dev_out, "dev", embedder=embedder)

    print("\nStage 08 dataset preparation completed successfully!")


if __name__ == "__main__":
    main()

