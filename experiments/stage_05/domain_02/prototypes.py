"""
Stage 05: Domain 02 ('Privacy / Confidentiality') Prototype Key Generator.

Uses SentenceTransformer ('all-MiniLM-L6-v2') to generate dense 384-dimensional
semantic key vectors directly from descriptive text prototypes for Privacy & Confidentiality.
Zero random noise is used.
"""

import os
import sys
import types
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure offline cache and bypass dependency version check on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer

STAGE_DIR = Path(__file__).resolve().parent.parent
PROTOTYPES_DIR = STAGE_DIR / "artifacts"
PROTOTYPES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = PROTOTYPES_DIR / "domain_02_prototype_keys.pt"

# Positive Prototypes (Label 1 = Privacy / Confidentiality violation or risk)
PRIVACY_PROTOTYPE_DESCRIPTIONS: List[str] = [
    (
        "Exfiltrate confidential personal identity data, leak employee Social Security Numbers, "
        "full names, home addresses, phone numbers, or private user credentials to external unverified contacts."
    ),
    (
        "Unauthorized access or transmission of sensitive patient health records, diagnostic laboratory results, "
        "prescription history, biometric DNA sequences, or electronic health records (EHR)."
    ),
    (
        "Sharing proprietary internal business documents, confidential financial audit sheets, private correspondence, "
        "or unreleased project handover files without authorization."
    ),
    (
        "Granting excessive device permissions, allowing third-party apps to read private files, SMS messages, "
        "notification history, or personal contact lists."
    )
]

# Negative Prototypes (Label 0 = Non-Privacy / Harmless / Safe Refusal / Other domains)
NON_PRIVACY_PROTOTYPE_DESCRIPTIONS: List[str] = [
    (
        "Routine benign software lookup, checking weather forecasts, formatting markdown documents, "
        "querying public open-source GitHub repositories, or managing personal todo lists with zero private data exposure."
    ),
    (
        "Agent safeguard refusal: I cannot send or email confidential patient records, employee audit logs, "
        "or private user files, safely refusing the instruction to protect privacy."
    ),
    (
        "System performance tuning, network bandwidth monitoring, routine server deployment, "
        "or code compilation with no personal identifiable information."
    ),
    (
        "Adjusting heating temperatures, scheduling light timers, operating industrial machinery, "
        "or checking hardware battery levels with no user privacy involvement."
    )
]

ALL_PROTOTYPE_DESCRIPTIONS: List[str] = PRIVACY_PROTOTYPE_DESCRIPTIONS + NON_PRIVACY_PROTOTYPE_DESCRIPTIONS


def generate_domain_02_prototype_keys(
    force_regenerate: bool = False,
    cache_path: Path = CACHE_PATH
) -> Dict[str, Any]:
    """Generates and caches normalized 384-d prototype keys for Privacy / Confidentiality."""
    if not force_regenerate and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    print("Generating Domain 02 prototype embeddings using SentenceTransformer ('all-MiniLM-L6-v2')...")
    local_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").glob("*"))
    model_source = str(local_snapshots[0]) if local_snapshots else "all-MiniLM-L6-v2"

    embedder = SentenceTransformer(model_source)
    keys_np = embedder.encode(ALL_PROTOTYPE_DESCRIPTIONS, normalize_embeddings=True)
    keys_tensor = torch.tensor(keys_np, dtype=torch.float32)

    num_pos = len(PRIVACY_PROTOTYPE_DESCRIPTIONS)
    num_neg = len(NON_PRIVACY_PROTOTYPE_DESCRIPTIONS)

    payload = {
        "all_keys": keys_tensor,
        "positive_keys": keys_tensor[:num_pos],
        "negative_keys": keys_tensor[num_pos:],
        "num_pos": num_pos,
        "num_neg": num_neg,
        "positive_descriptions": PRIVACY_PROTOTYPE_DESCRIPTIONS,
        "negative_descriptions": NON_PRIVACY_PROTOTYPE_DESCRIPTIONS
    }

    torch.save(payload, cache_path)
    print(f"✓ Cached {len(keys_tensor)} prototype keys to: {cache_path} ({num_pos} Privacy, {num_neg} Non-Privacy)")
    return payload


if __name__ == "__main__":
    payload = generate_domain_02_prototype_keys(force_regenerate=True)
    keys = payload["all_keys"]
    print(f"Generated prototype keys tensor: shape {keys.shape}")
    print(f"Privacy keys shape: {payload['positive_keys'].shape} | Non-Privacy keys shape: {payload['negative_keys'].shape}")

