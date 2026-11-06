"""
Stage 04: Domain 01 ('No Material Harm') Model Architecture.

Implements the SemanticPrototypeClassifier:
- Initializes prototype keys directly from pre-computed SentenceTransformer text embeddings (no random noise).
- Supports Multi-Prototype Domain Representations (covering Benign, Safe Refusals, and Multi-Domain Harms).
- Trains both the input query projection (query_proj) and the domain prototype key projection (key_proj).
- Uses calibrated temperature scaling to prevent logits from sticking in mid-values.
- Attentively aggregates multi-prototype similarities with latent query features to prevent the 1D bottleneck.
- Completely separated from training loops and evaluation runner.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPrototypeClassifier(nn.Module):
    """
    Semantic Prototype Classifier:
    Projects 384-d input query embeddings and pre-computed prototype key embeddings
    into a shared latent metric space, computing temperature-scaled dot-product similarity
    across multiple specialized domain prototypes (benign queries, safe refusals, and harm categories).
    """

    def __init__(
        self,
        initial_prototype_embeddings: torch.Tensor,
        num_harm: int = 4,
        num_safe: int = 4,
        hidden_dim: int = 64,
        initial_scale: float = 10.0,
        dropout_rate: float = 0.20
    ):
        super().__init__()
        self.embedding_dim = initial_prototype_embeddings.shape[-1]
        self.num_prototypes = initial_prototype_embeddings.shape[0]
        self.num_harm = num_harm
        self.num_safe = num_safe
        self.hidden_dim = hidden_dim

        # 1. Registered Buffer: Pre-computed text prototype embeddings (zero random noise)
        self.register_buffer("prototype_keys", initial_prototype_embeddings.clone().detach())

        # 2. Trainable Query Projection Network: Q -> Q_h
        self.query_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 3. Trainable Key Projection Network: K -> K_h
        # Trains the domain key vectors in the shared metric space alongside queries
        self.key_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 4. Calibrated Scale & Bias
        self.scale = nn.Parameter(torch.tensor(initial_scale))
        self.bias = nn.Parameter(torch.tensor(0.0))

        # 5. Attentive Fusion Head
        # Combines prototype affinity similarities (num_prototypes) with latent query features (hidden_dim)
        # to ensure both semantic grounding and direct discriminative power are utilized.
        self.fusion_head = nn.Sequential(
            nn.Linear(self.num_prototypes + hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def compute_projected_embeddings(
        self,
        q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns projected query latent, normalized query vector, and normalized key vectors."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q_h = self.query_proj(q)
        q_norm = F.normalize(q_h, p=2, dim=-1)

        k_h = self.key_proj(self.prototype_keys)
        k_norm = F.normalize(k_h, p=2, dim=-1)
        return q_h, q_norm, k_norm

    def compute_prototype_similarities(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cosine similarity matrix in [-1, 1] scaled by temperature:
            sim = (Q_norm @ K_norm.T) * scale
        Returns:
            sim: (batch_size, num_prototypes)
            q_h: (batch_size, hidden_dim)
        """
        q_h, q_norm, k_norm = self.compute_projected_embeddings(q)
        sim = torch.matmul(q_norm, k_norm.t()) * self.scale
        return sim, q_h

    def forward_logits(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes fused classification logit for class 1 ('No Material Harm').
        Returns:
            Tensor of shape (batch_size,)
        """
        sim, q_h = self.compute_prototype_similarities(q)
        fused = torch.cat([sim, q_h], dim=-1)
        logits = self.fusion_head(fused).squeeze(-1) + self.bias
        return logits

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes calibrated probability for class 1 ('No Material Harm') via sigmoid.
        Returns:
            Tensor of shape (batch_size,) in range (0.0, 1.0)
        """
        logits = self.forward_logits(q)
        return torch.sigmoid(logits)

    def predict_proba(self, q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Returns 1D numpy array of probabilities for 'No Material Harm' (class 1)."""
        self.eval()
        with torch.no_grad():
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q, dtype=torch.float32)
            probs = self.forward(q)
            return probs.cpu().numpy()

    def predict(
        self,
        q: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.50
    ) -> np.ndarray:
        """Returns discrete 0 or 1 predictions based on decision threshold."""
        probs = self.predict_proba(q)
        return (probs >= threshold).astype(int)
