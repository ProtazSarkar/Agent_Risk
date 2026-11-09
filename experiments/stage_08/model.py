"""
Stage 08 Optimized Model: 5-Channel Cross-Attentive Prototype Classifier.

Key Innovations:
  1. 5 Semantic Channels:
     Isolates [query, thought, context_actions, observations, proposed_action].
     Distinctly isolates pending execution from historical traces.
  2. Channel Type Embeddings:
     Adds learnable semantic positional indicators for each interaction role.
  3. Domain-Prototype Cross-Attention:
     Positive ("Safe") and Negative ("Harmful") prototype keys directly cross-attend
     into the 5 interaction channels, measuring exact alignment against safety guidelines.
  4. Multi-Head Self-Attention Context Bridge:
     Enables cross-talk between pending proposed actions and tool observations.
  5. Calibrated Dual-Path Head:
     Combines direct cosine prototype similarity and cross-attention representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ChannelEncoder(nn.Module):
    """Encodes and refines 5 interaction channels with channel type embeddings."""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 384, num_channels: int = 5, dropout: float = 0.15):
        super().__init__()
        self.channel_type_emb = nn.Embedding(num_channels, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x_channels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_channels: (batch_size, 5, embed_dim)
        Returns:
            refined: (batch_size, 5, hidden_dim)
        """
        B, C, D = x_channels.shape
        proj = self.input_proj(x_channels)

        # Add channel type embeddings
        channel_indices = torch.arange(C, device=x_channels.device).unsqueeze(0).expand(B, -1)
        type_embs = self.channel_type_emb(channel_indices)
        h = proj + type_embs

        # Self-attention across the 5 channels
        attn_out, _ = self.self_attn(h, h, h)
        h = self.norm1(h + attn_out)

        # FFN
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        return h


class Stage08OptimizedDomainClassifier(nn.Module):
    """
    Optimized 5-Channel Prototype Cross-Attention Network for Domain 01.
    """

    def __init__(
        self,
        initial_prototype_keys: torch.Tensor,
        num_pos_prototypes: int = 4,
        embed_dim: int = 384,
        hidden_dim: int = 384,
        num_channels: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_pos = num_pos_prototypes
        self.num_prototypes = initial_prototype_keys.shape[0]

        # 1. Prototype keys buffer
        self.register_buffer("prototype_keys", initial_prototype_keys.clone().detach())

        # 2. 5-Channel Encoder with Self-Attention
        self.channel_encoder = ChannelEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            dropout=dropout
        )

        # 3. Prototype Key Projection
        self.key_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 4. Prototype-to-Channel Cross-Attention:
        # Prototypes (queries) attend to interaction channels (keys/values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # 5. Channel-wise Attention Pooling for Global Query
        self.channel_pooling = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 6. Dual-Path Classification Head
        # Inputs: pooled prototype similarity (num_prototypes) + cross-attended summary (num_prototypes)
        self.head = nn.Sequential(
            nn.Linear(self.num_prototypes * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(hidden_dim)))

    def forward(
        self,
        x_channels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x_channels: (batch_size, 5, embed_dim)

        Returns:
            prob: (batch_size,) calibrated P(Safe = 1)
            logits: (batch_size,)
            meta: dictionary of cross-attention weights and prototype similarities
        """
        B = x_channels.shape[0]

        # 1. Encode 5 channels with self-attention
        ch_tokens = self.channel_encoder(x_channels)  # (B, 5, hidden_dim)

        # 2. Global Query vector via attention pooling
        pool_weights = F.softmax(self.channel_pooling(ch_tokens), dim=1)  # (B, 5, 1)
        global_query = torch.sum(pool_weights * ch_tokens, dim=1)         # (B, hidden_dim)
        q_norm = F.normalize(global_query, p=2, dim=-1)

        # 3. Prototype Projection
        k_proj = self.key_proj(self.prototype_keys)                       # (8, hidden_dim)
        k_norm = F.normalize(k_proj, p=2, dim=-1)

        # Direct Prototype Cosine Similarities
        proto_sim = torch.matmul(q_norm, k_norm.t()) * self.scale        # (B, 8)

        # 4. Prototype-Guided Cross-Attention:
        # Replicate prototype keys across batch: (B, 8, hidden_dim)
        proto_queries = k_proj.unsqueeze(0).expand(B, -1, -1)
        cross_out, cross_weights = self.cross_attn(
            query=proto_queries,
            key=ch_tokens,
            value=ch_tokens
        )
        cross_out = self.cross_norm(proto_queries + cross_out)            # (B, 8, hidden_dim)

        # Measure alignment of cross-attended representation with original prototypes
        cross_sim = F.cosine_similarity(cross_out, proto_queries, dim=-1) # (B, 8)

        # 5. Dual-Path Head Combination
        combined_feats = torch.cat([proto_sim, cross_sim], dim=-1)       # (B, 16)
        logits = self.head(combined_feats).squeeze(-1)                    # (B,)
        prob = torch.sigmoid(logits)

        meta = {
            "channel_pool_weights": pool_weights.squeeze(-1),             # (B, 5)
            "cross_attn_weights": cross_weights,                          # (B, 8, 5)
            "proto_sim": proto_sim,                                       # (B, 8)
            "pos_sim_mean": proto_sim[:, :self.num_pos].mean(dim=-1),
            "neg_sim_mean": proto_sim[:, self.num_pos:].mean(dim=-1)
        }
        return prob, logits, meta

