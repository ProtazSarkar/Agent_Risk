"""
Stage 03: Domain-Aware Attention and Hybrid Feature Extraction Mechanism.

Combines the fine-grained 384-dimensional dense semantic embeddings Q (emb_0 to emb_383)
with the high-level semantic domain similarity scores S (5-d) and softmax attention weights (5-d),
yielding a comprehensive 394-dimensional hybrid representation (384 + 5 + 5 = 394).
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from experiments.stage_03.label import DomainLabelRegistry
except (ImportError, ModuleNotFoundError):
    from label import DomainLabelRegistry


def map_val_to_severity(
    val: Union[float, np.ndarray, torch.Tensor],
    thresholds: Tuple[float, float, float, float] = (1.5, 2.5, 3.5, 4.5)
) -> Union[int, np.ndarray]:
    """
    Maps continuous linear score `val` to discrete risk severity (1 to 5):
        - val in (0, 1.5)   -> Severity 1
        - val in [1.5, 2.5) -> Severity 2
        - val in [2.5, 3.5) -> Severity 3
        - val in [3.5, 4.5) -> Severity 4
        - val >= 4.5       -> Severity 5
    """
    if isinstance(val, torch.Tensor):
        val_arr = val.detach().cpu().numpy()
    else:
        val_arr = np.asarray(val)

    is_scalar = (val_arr.ndim == 0)
    flat = val_arr.reshape(-1)

    t1, t2, t3, t4 = thresholds
    discrete = np.ones(flat.shape, dtype=int)
    discrete[(flat >= t1) & (flat < t2)] = 2
    discrete[(flat >= t2) & (flat < t3)] = 3
    discrete[(flat >= t3) & (flat < t4)] = 4
    discrete[flat >= t4] = 5

    if is_scalar:
        return int(discrete[0])
    return discrete.reshape(val_arr.shape)


class DomainAttention:
    """
    Computes cosine similarity scores and scaled attention values between
    query embedding vectors Q (384-d) and domain prototype key vectors K (5, 384).
    """

    def __init__(
        self,
        key_matrix: Optional[np.ndarray] = None,
        registry: Optional[DomainLabelRegistry] = None
    ):
        if key_matrix is not None:
            self.key_matrix = key_matrix
            self.label_keys = [f"domain_{i+1}" for i in range(len(key_matrix))]
            self.label_names = self.label_keys
        else:
            self.registry = registry or DomainLabelRegistry()
            self.key_matrix = self.registry.get_key_matrix()
            self.label_keys = self.registry.label_keys
            self.label_names = self.registry.label_names

        # Normalize keys: ||K_i||_2 = 1
        k_norms = np.linalg.norm(self.key_matrix, axis=1, keepdims=True)
        k_norms[k_norms == 0] = 1.0
        self.K_norm = self.key_matrix / k_norms
        self.num_labels = len(self.K_norm)
        self.embedding_dim = self.K_norm.shape[1]

        self.similarity_feature_names = [f"sim_{k}" for k in self.label_keys]
        self.attention_feature_names = [f"attn_{k}" for k in self.label_keys]

    def compute_similarity(self, Q: np.ndarray) -> np.ndarray:
        """Computes cosine similarity: S = Q_norm @ K_norm.T"""
        Q_arr = np.asarray(Q, dtype=np.float32)
        is_single = (Q_arr.ndim == 1)
        if is_single:
            Q_arr = Q_arr.reshape(1, -1)

        q_norms = np.linalg.norm(Q_arr, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        Q_norm = Q_arr / q_norms

        similarities = np.dot(Q_norm, self.K_norm.T)
        if is_single:
            return similarities[0]
        return similarities

    def compute_attention_weights(self, Q: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        """Computes scaled dot-product softmax attention distribution across domain labels."""
        sims = self.compute_similarity(Q)
        is_single = (sims.ndim == 1)
        if is_single:
            sims = sims.reshape(1, -1)

        scale = temperature if temperature is not None else np.sqrt(self.embedding_dim)
        scaled_logits = sims / scale
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if is_single:
            return weights[0]
        return weights

    def build_hybrid_features(
        self,
        Q: np.ndarray,
        include_attention: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenates:
            Q (384-d) + S (5-d) + [attn (5-d)] -> 389-d or 394-d
        """
        Q_arr = np.asarray(Q, dtype=np.float32)
        if Q_arr.ndim == 1:
            Q_arr = Q_arr.reshape(1, -1)

        sims = self.compute_similarity(Q_arr)
        raw_names = [f"emb_{i}" for i in range(self.embedding_dim)]
        blocks = [Q_arr, sims]
        names = raw_names + self.similarity_feature_names

        if include_attention:
            attn = self.compute_attention_weights(Q_arr)
            blocks.append(attn)
            names += self.attention_feature_names

        hybrid = np.hstack(blocks)
        return hybrid, names

