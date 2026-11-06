"""
Stage 05: Domain 01 ('No Material Harm') Prototype Key Generator.

Provides pre-computed 384-dimensional semantic prototype key vectors
initialized directly from SentenceTransformer ('all-MiniLM-L6-v2').
Zero random noise is used.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import torch

STAGE_DIR = Path(__file__).resolve().parent.parent
PROTOTYPES_DIR = STAGE_DIR / "artifacts"
PROTOTYPES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = PROTOTYPES_DIR / "domain_01_prototype_keys.pt"


def generate_domain_01_prototype_keys() -> Dict[str, Any]:
    """Loads cached multi-prototype keys directly without importing sentence_transformers."""
    if CACHE_PATH.exists():
        return torch.load(CACHE_PATH, weights_only=False)

    # Fallback to stage_04 artifact if not yet copied
    fallback_path = STAGE_DIR.parent / "stage_04" / "artifacts" / "domain_01_prototype_keys.pt"
    if fallback_path.exists():
        data = torch.load(fallback_path, weights_only=False)
        torch.save(data, CACHE_PATH)
        return data

    raise FileNotFoundError(f"Prototype keys not found at {CACHE_PATH} or {fallback_path}")


if __name__ == "__main__":
    payload = generate_domain_01_prototype_keys()
    keys = payload["all_keys"]
    print(f"Loaded prototype keys: shape {keys.shape} ({payload['num_harm']} Harm, {payload['num_safe']} Safe)")

