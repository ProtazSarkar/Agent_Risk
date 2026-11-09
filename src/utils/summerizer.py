"""
Multi-Section Chunked Interaction Embedding & Fusion Utility.

Extracts structured interaction sections:
  - query: User query / instructions [USER]
  - action: Agent tools and commands [ACTION] + pending proposed action
  - thought: Agent internal reasoning [THOUGHT]
  - observation: Tool outputs and environment observations [OBSERVATION]

Chunks large text windows to prevent truncation, encodes chunks using
SentenceTransformer ('all-MiniLM-L6-v2'), mean-pools chunk representations,
and provides a learnable PyTorch SectionEmbeddingOptimizer for end-to-end fusion.
"""

import os
import sys
import re
import types
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure offline cache and bypass dependency version check on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer

# Cached global embedder instance
_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDER = SentenceTransformer(model_name, device=device)
    return _EMBEDDER


def extract_interaction_sections(
    formatted_interaction: str,
    proposed_action: str = ""
) -> Dict[str, str]:
    """
    Extracts structured text sections from agent interaction logs:
      - 'query': [USER] requests
      - 'action': [ACTION] steps + proposed_action
      - 'thought': [THOUGHT] reasoning steps
      - 'observation': [OBSERVATION] tool outputs

    Handles multiple steps by grouping contents under each section tag.
    """
    text = formatted_interaction or ""
    sections: Dict[str, List[str]] = {
        "query": [],
        "thought": [],
        "action": [],
        "observation": []
    }
    tag_map = {
        "USER": "query",
        "THOUGHT": "thought",
        "ACTION": "action",
        "OBSERVATION": "observation"
    }

    # Split on standard step format: Step <N>. [<TAG>]:
    parts = re.split(r'Step \d+\.\s*\[(USER|THOUGHT|ACTION|OBSERVATION)\]:\s*', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            tag = parts[i].upper()
            content = parts[i + 1].strip()
            if tag in tag_map and content:
                sections[tag_map[tag]].append(content)
    else:
        # Fallback for alternative tag formatting without 'Step N.'
        cur_tag = "query"
        for line in text.split("\n"):
            match = re.match(r'\[(USER|THOUGHT|ACTION|OBSERVATION)\]:\s*(.*)', line, re.IGNORECASE)
            if match:
                cur_tag = tag_map.get(match.group(1).upper(), "query")
                remaining = match.group(2).strip()
                if remaining:
                    sections[cur_tag].append(remaining)
            elif line.strip():
                sections[cur_tag].append(line.strip())

    # Incorporate the critical proposed next action into the action section
    if proposed_action and proposed_action.strip():
        sections["action"].append(f"Proposed Pending Action: {proposed_action.strip()}")

    # Join multiple steps with newline separator; default to neutral fallback if empty
    result = {
        k: " \n".join(v).strip() if v else ""
        for k, v in sections.items()
    }
    return result


def chunk_text(
    text: str,
    max_chunk_words: int = 120,
    overlap_words: int = 20
) -> List[str]:
    """
    Splits text into overlapping word chunks when text exceeds max_chunk_words.
    Prevents token truncation in dense sentence encoders (e.g. 256-token limit).
    """
    if not text or not text.strip():
        return []

    words = text.strip().split()
    if len(words) <= max_chunk_words:
        return [text.strip()]

    chunks = []
    step = max(1, max_chunk_words - overlap_words)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_chunk_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + max_chunk_words >= len(words):
            break

    return chunks


def embed_chunked_text(
    text: str,
    embedder: Optional[SentenceTransformer] = None,
    max_chunk_words: int = 120,
    overlap_words: int = 20
) -> np.ndarray:
    """
    Chunks a text section, encodes each chunk using SentenceTransformer,
    and returns the normalized mean-pooled embedding of shape (384,).
    """
    if embedder is None:
        embedder = get_embedder()

    chunks = chunk_text(text, max_chunk_words=max_chunk_words, overlap_words=overlap_words)
    if not chunks:
        # Zero-vector fallback for empty sections
        return np.zeros(384, dtype=np.float32)

    embeddings = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    mean_vec = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 1e-9:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32)


def extract_and_embed_sections(
    formatted_interaction: str,
    proposed_action: str = "",
    embedder: Optional[SentenceTransformer] = None,
    max_chunk_words: int = 120,
    overlap_words: int = 20
) -> Dict[str, np.ndarray]:
    """
    Extracts the 4 interaction sections (query, action, thought, observation),
    chunks long sections, encodes each chunk, and returns a dictionary of 4
    normalized embeddings, each of shape (384,).
    """
    if embedder is None:
        embedder = get_embedder()

    sections = extract_interaction_sections(formatted_interaction, proposed_action)
    section_embs = {}
    for sec_name in ["query", "action", "thought", "observation"]:
        sec_text = sections.get(sec_name, "")
        section_embs[sec_name] = embed_chunked_text(
            sec_text,
            embedder=embedder,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words
        )
    return section_embs


def extract_and_embed_sections_stacked(
    formatted_interaction: str,
    proposed_action: str = "",
    embedder: Optional[SentenceTransformer] = None,
    max_chunk_words: int = 120,
    overlap_words: int = 20
) -> np.ndarray:
    """
    Returns a stacked (4, 384) numpy array for [query, action, thought, observation].
    """
    embs_dict = extract_and_embed_sections(
        formatted_interaction,
        proposed_action=proposed_action,
        embedder=embedder,
        max_chunk_words=max_chunk_words,
        overlap_words=overlap_words
    )
    return np.stack([
        embs_dict["query"],
        embs_dict["action"],
        embs_dict["thought"],
        embs_dict["observation"]
    ], axis=0).astype(np.float32)


def summerize_embedding(
    text: str,
    proposed_action: str = "",
    weights: Optional[Dict[str, float]] = None,
    embedder: Optional[SentenceTransformer] = None
) -> np.ndarray:
    """
    Generates a single unified 384-dimensional summarized embedding by combining
    the chunked section embeddings with weighted normalization.

    Default weighting emphasizes action and observation where safety violations occur:
      - query: 0.20
      - action: 0.40
      - thought: 0.15
      - observation: 0.25
    """
    if weights is None:
        weights = {
            "query": 0.20,
            "action": 0.40,
            "thought": 0.15,
            "observation": 0.25
        }

    section_embs = extract_and_embed_sections(
        text,
        proposed_action=proposed_action,
        embedder=embedder
    )

    fused = np.zeros(384, dtype=np.float32)
    total_w = sum(weights.values()) or 1.0
    for name, weight in weights.items():
        if name in section_embs:
            fused += (weight / total_w) * section_embs[name]

    norm = np.linalg.norm(fused)
    if norm > 1e-9:
        fused = fused / norm
    return fused.astype(np.float32)


def batch_extract_and_embed_sections(
    records: List[Dict[str, Any]],
    embedder: Optional[SentenceTransformer] = None,
    batch_size: int = 64,
    max_chunk_words: int = 120,
    overlap_words: int = 20,
    weights: Optional[Dict[str, float]] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched extraction and chunked encoding across a collection of interaction records.
    Significantly accelerates processing via bulk vectorized sentence encoding.

    Returns:
      stacked_sections: np.ndarray of shape (N, 4, 384) -> [query, action, thought, observation]
      fused_summaries: np.ndarray of shape (N, 384) -> weighted normalized summary embedding
    """
    if embedder is None:
        embedder = get_embedder()

    if weights is None:
        weights = {"query": 0.20, "action": 0.40, "thought": 0.15, "observation": 0.25}

    total_records = len(records)
    all_chunks: List[str] = []
    mapping: List[Tuple[int, int, int, int]] = []
    sec_names = ["query", "action", "thought", "observation"]

    for rec_idx, rec in enumerate(records):
        fi = rec.get("formatted_interaction", "")
        pa = rec.get("proposed_action", "")
        sec_dict = extract_interaction_sections(fi, pa)
        for sec_idx, sec_name in enumerate(sec_names):
            txt = sec_dict[sec_name]
            chunks = chunk_text(txt, max_chunk_words=max_chunk_words, overlap_words=overlap_words)
            c_start = len(all_chunks)
            all_chunks.extend(chunks)
            c_end = len(all_chunks)
            mapping.append((rec_idx, sec_idx, c_start, c_end))

    if all_chunks:
        chunk_embeddings = embedder.encode(
            all_chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
    else:
        chunk_embeddings = np.zeros((0, 384), dtype=np.float32)

    stacked_sections = np.zeros((total_records, 4, 384), dtype=np.float32)
    for rec_idx, sec_idx, c_start, c_end in mapping:
        if c_end > c_start:
            sec_mean = np.mean(chunk_embeddings[c_start:c_end], axis=0)
            norm = np.linalg.norm(sec_mean)
            if norm > 1e-9:
                sec_mean = sec_mean / norm
            stacked_sections[rec_idx, sec_idx] = sec_mean

    fused_summaries = np.zeros((total_records, 384), dtype=np.float32)
    tot_w = sum(weights.values()) or 1.0
    w_vec = np.array([weights[s] / tot_w for s in sec_names], dtype=np.float32)

    for i in range(total_records):
        f = np.sum(stacked_sections[i] * w_vec[:, None], axis=0)
        norm = np.linalg.norm(f)
        if norm > 1e-9:
            f = f / norm
        fused_summaries[i] = f

    return stacked_sections, fused_summaries


# ==============================================================================
# PyTorch Section Embedding Optimizer / Learnable Fusion Module
# ==============================================================================

class SectionEmbeddingOptimizer(nn.Module):
    """
    Learnable neural fusion module for multi-section interaction embeddings.
    Accepts stacked section embeddings of shape (batch_size, num_sections, embed_dim)
    where num_sections = 4 ([query, action, thought, observation]).

    Learns dynamic section-attention weights conditioned on interaction context,
    projects through a LayerNorm non-linear layer, and applies a residual shortcut.
    Outputs an optimized unified representation of shape (batch_size, embed_dim).
    """

    def __init__(self, embed_dim: int = 384, num_sections: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sections = num_sections

        # Dynamic Section Attention Network
        self.section_attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Non-linear section projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim)
        )

        # Residual gating parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, num_sections, embed_dim) stacked section embeddings.

        Returns:
            fused_embedding: (batch_size, embed_dim) normalized final embedding.
            attention_weights: (batch_size, num_sections) section weights.
        """
        # Compute unnormalized attention logits: (batch_size, num_sections, 1)
        attn_logits = self.section_attention(x)
        attn_weights = F.softmax(attn_logits, dim=1)  # (batch_size, num_sections, 1)

        # Weighted pooling across sections
        pooled = torch.sum(attn_weights * x, dim=1)  # (batch_size, embed_dim)

        # Non-linear projection with residual connection
        projected = self.fusion_proj(pooled)
        out = pooled + self.alpha * projected

        # Final L2 normalization
        out_norm = F.normalize(out, p=2, dim=-1)
        return out_norm, attn_weights.squeeze(-1)
