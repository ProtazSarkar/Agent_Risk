"""
Stage 07 Model: Multi-Section Embedding Optimizer + Semantic Prototype Classifier.

Architecture:
  1. SectionEmbeddingOptimizer:
     Accepts (batch_size, 4, 384) representing [query, action, thought, observation].
     Computes dynamic section-attention weights, pools sections, and projects with residual LayerNorm.
     Outputs optimized fused query vector (batch_size, 384).

  2. SemanticPrototypeClassifier:
     Contains pre-computed Domain 01 prototype keys buffer (8, 384):
       - 4 Positive Safe keys ("No Material Harm")
       - 4 Negative Harm keys (malicious, exfiltration, destruction, etc.)
     Projects both query and keys into hidden space, computes scaled cosine similarity,
     and combines positive and negative prototype affinities to produce calibrated P(Safe = 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SectionEmbeddingOptimizer(nn.Module):
    """
    Learns dynamic section-importance attention across the 4 interaction sections:
      Index 0: query ([USER])
      Index 1: action ([ACTION] + proposed action)
      Index 2: thought ([THOUGHT])
      Index 3: observation ([OBSERVATION])
    """

    def __init__(self, embed_dim: int = 384, num_sections: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sections = num_sections

        # Attention scoring network
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Context-aware non-linear refinement
        self.refinement_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim)
        )

        # Learnable residual mixing coefficient
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, num_sections, embed_dim) stacked section embeddings.

        Returns:
            fused: (batch_size, embed_dim) normalized representation.
            attn_weights: (batch_size, num_sections) attention distribution.
        """
        # Compute dynamic section attention logits
        logits = self.attn_net(x)  # (batch_size, num_sections, 1)
        attn_weights = F.softmax(logits, dim=1)  # (batch_size, num_sections, 1)

        # Attention-weighted pooling
        pooled = torch.sum(attn_weights * x, dim=1)  # (batch_size, embed_dim)

        # Residual refinement
        refined = self.refinement_proj(pooled)
        out = pooled + self.gate * refined

        # Unit L2 normalization
        out_norm = F.normalize(out, p=2, dim=-1)
        return out_norm, attn_weights.squeeze(-1)


class Stage07DomainClassifier(nn.Module):
    """
    Unified Stage 07 Network for Domain 01 (No Material Harm).
    Integrates the SectionEmbeddingOptimizer with pre-computed Semantic Prototype Keys.
    """

    def __init__(
        self,
        initial_prototype_keys: torch.Tensor,
        num_pos_prototypes: int = 4,
        embed_dim: int = 384,
        hidden_dim: int = 384,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_pos = num_pos_prototypes
        self.num_prototypes = initial_prototype_keys.shape[0]

        # 1. Section Fusion Optimizer
        self.section_optimizer = SectionEmbeddingOptimizer(
            embed_dim=embed_dim,
            num_sections=4,
            hidden_dim=256
        )

        # 2. Prototype Keys Buffer (zero random noise)
        self.register_buffer("prototype_keys", initial_prototype_keys.clone().detach())

        # 3. Query & Key Projections
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.scale = nn.Parameter(torch.tensor(1.0 / (hidden_dim ** 0.5)))

        # 4. Prototype Affinity Classifier Head
        # Takes similarity vector of size num_prototypes (e.g. 8)
        self.head = nn.Sequential(
            nn.Linear(self.num_prototypes, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x_sections: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x_sections: (batch_size, 4, 384) section embeddings.

        Returns:
            prob: (batch_size,) calibrated probability that sample is Safe (1).
            logits: (batch_size,) raw logits for BCEWithLogitsLoss.
            meta: dictionary with section weights and prototype similarities.
        """
        # Step 1: Optimize and fuse sections
        fused_query, section_weights = self.section_optimizer(x_sections)

        # Step 2: Project query and prototype keys
        q_proj = self.query_proj(fused_query)
        q_norm = F.normalize(q_proj, p=2, dim=-1)

        k_proj = self.key_proj(self.prototype_keys)
        k_norm = F.normalize(k_proj, p=2, dim=-1)

        # Step 3: Scaled dot-product prototype similarities
        sim = torch.matmul(q_norm, k_norm.t()) * self.scale  # (batch_size, num_prototypes)

        # Step 4: Classification head
        logits = self.head(sim).squeeze(-1)  # (batch_size,)
        prob = torch.sigmoid(logits)

        meta = {
            "section_weights": section_weights,
            "prototype_similarities": sim,
            "pos_sim_mean": sim[:, :self.num_pos].mean(dim=-1),
            "neg_sim_mean": sim[:, self.num_pos:].mean(dim=-1)
        }
        return prob, logits, meta

