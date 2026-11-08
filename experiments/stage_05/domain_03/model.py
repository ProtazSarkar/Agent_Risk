"""
Stage 05: domain_03 ('Security / System Integrity') Model Architecture.

Implements the SemanticPrototypeClassifier trained on summarized embeddings:
- Multi-prototype cross-attention with calibrated temperature scaling (scale = 10.0)
- Trainable Query Projection Network (query_proj) and Key Projection Network (key_proj)
- Attentive Fusion Head combining prototype similarity signals with query latent features
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
    Semantic Prototype Classifier for domain_03: 'Security / System Integrity'.
    Projects 384-d summarized input query embeddings and pre-computed prototype key embeddings
    into a shared latent metric space, computing temperature-scaled dot-product similarity.
    """

    def __init__(
        self,
        initial_prototype_embeddings: torch.Tensor,
        num_pos: int = 4,
        num_neg: int = 4,
        hidden_dim: int = 64,
        initial_scale: float = 10.0,
        dropout_rate: float = 0.20
    ):
        super().__init__()
        self.embedding_dim = initial_prototype_embeddings.shape[-1]
        self.num_prototypes = initial_prototype_embeddings.shape[0]
        self.num_pos = num_pos
        self.num_neg = num_neg
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
        self.key_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 4. Calibrated Scale & Bias
        self.scale = nn.Parameter(torch.tensor(initial_scale))
        self.bias = nn.Parameter(torch.tensor(0.0))

        # 5. Attentive Fusion Head
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
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q_h = self.query_proj(q)
        q_norm = F.normalize(q_h, p=2, dim=-1)

        k_h = self.key_proj(self.prototype_keys)
        k_norm = F.normalize(k_h, p=2, dim=-1)
        return q_h, q_norm, k_norm

    def compute_prototype_similarities(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_h, q_norm, k_norm = self.compute_projected_embeddings(q)
        sim = torch.matmul(q_norm, k_norm.t()) * self.scale
        return sim, q_h

    def forward_logits(self, q: torch.Tensor) -> torch.Tensor:
        sim, q_h = self.compute_prototype_similarities(q)
        fused = torch.cat([sim, q_h], dim=-1)
        logits = self.fusion_head(fused).squeeze(-1) + self.bias
        return logits

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(q)
        return torch.sigmoid(logits)

    def predict_proba(self, q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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
        probs = self.predict_proba(q)
        return (probs >= threshold).astype(int)
