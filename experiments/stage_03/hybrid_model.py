"""
Stage 03: Deep Hybrid Neural Network Architecture for Agent Risk Prediction.

Combines fine-grained semantic representations (384-d embeddings) with explicit
risk domain alignments (5-d similarity + 5-d attention weights) using dual-branch
deep neural networks with dynamic range calibration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from experiments.stage_03.attention import map_val_to_severity
except (ImportError, ModuleNotFoundError):
    from attention import map_val_to_severity


class DeepHybridRiskRegressor(nn.Module):
    """
    Dual-branch Deep Neural Regressor:
      Branch A: 384-d Semantic Dense Embeddings -> Linear(384, 64) -> LayerNorm -> ReLU -> Dropout
      Branch B: 10-d Domain Projections (5 sim + 5 attn) -> Linear(10, 16) -> ReLU
      Combined Head: Linear(80, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(
        self,
        emb_dim: int = 384,
        domain_dim: int = 10,
        dropout_rate: float = 0.15,
        thresholds: Tuple[float, float, float, float] = (1.5, 2.5, 3.5, 4.5)
    ):
        super().__init__()
        self.thresholds = thresholds
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.domain_branch = nn.Sequential(
            nn.Linear(domain_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Learnable / calibrated affine scaling layer: val_cal = alpha * val + beta
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward_raw(self, x_emb: torch.Tensor, x_domain: torch.Tensor) -> torch.Tensor:
        h_emb = self.emb_branch(x_emb)
        h_dom = self.domain_branch(x_domain)
        h = torch.cat([h_emb, h_dom], dim=-1)
        return self.head(h)

    def forward(self, x_emb: torch.Tensor, x_domain: torch.Tensor) -> torch.Tensor:
        raw_val = self.forward_raw(x_emb, x_domain)
        return self.scale * raw_val + self.shift

    def predict_continuous(
        self,
        x_emb: Union[np.ndarray, torch.Tensor],
        x_domain: Union[np.ndarray, torch.Tensor],
        apply_calibration: bool = True
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if not isinstance(x_emb, torch.Tensor):
                x_emb = torch.tensor(x_emb, dtype=torch.float32)
            if not isinstance(x_domain, torch.Tensor):
                x_domain = torch.tensor(x_domain, dtype=torch.float32)

            if apply_calibration:
                preds = self.forward(x_emb, x_domain)
            else:
                preds = self.forward_raw(x_emb, x_domain)
            return preds.squeeze().numpy()

    def predict_discrete(
        self,
        x_emb: Union[np.ndarray, torch.Tensor],
        x_domain: Union[np.ndarray, torch.Tensor],
        thresholds: Optional[Tuple[float, float, float, float]] = None,
        apply_calibration: bool = True
    ) -> np.ndarray:
        th = thresholds or self.thresholds
        cont_preds = self.predict_continuous(x_emb, x_domain, apply_calibration=apply_calibration)
        return map_val_to_severity(cont_preds, thresholds=th)

    def fit_calibration_bounds(
        self,
        raw_preds: np.ndarray,
        target_min: float = 0.8,
        target_max: float = 4.8
    ):
        """Calibrates scale and shift so outputs dynamically span [target_min, target_max]."""
        p_min = float(np.percentile(raw_preds, 2))
        p_max = float(np.percentile(raw_preds, 98))
        if p_max <= p_min:
            p_max = p_min + 1.0

        alpha = (target_max - target_min) / (p_max - p_min)
        beta = target_min - alpha * p_min

        with torch.no_grad():
            self.scale.copy_(torch.tensor(alpha, dtype=torch.float32))
            self.shift.copy_(torch.tensor(beta, dtype=torch.float32))


class DeepHybridRiskClassifier(nn.Module):
    """
    Dual-branch Multi-Class Classifier with softmax probability distribution.
    Outputs 5 class logits for Risk Severity Levels 1 through 5.
    """

    def __init__(
        self,
        emb_dim: int = 384,
        domain_dim: int = 10,
        num_classes: int = 5,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.domain_branch = nn.Sequential(
            nn.Linear(domain_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_emb: torch.Tensor, x_domain: torch.Tensor) -> torch.Tensor:
        h_emb = self.emb_branch(x_emb)
        h_dom = self.domain_branch(x_domain)
        return self.head(torch.cat([h_emb, h_dom], dim=-1))

    def predict(
        self,
        x_emb: Union[np.ndarray, torch.Tensor],
        x_domain: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if not isinstance(x_emb, torch.Tensor):
                x_emb = torch.tensor(x_emb, dtype=torch.float32)
            if not isinstance(x_domain, torch.Tensor):
                x_domain = torch.tensor(x_domain, dtype=torch.float32)
            logits = self.forward(x_emb, x_domain)
            preds = torch.argmax(logits, dim=-1).numpy() + 1
            return preds

