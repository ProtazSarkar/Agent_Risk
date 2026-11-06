"""
Stage 02: Domain-Aware Attention and Similarity Computation Mechanism.

Computes cosine similarity between interaction query embeddings Q (384-d) and the
domain risk key matrix K (5, 384), yielding 5 explicit domain alignment features.
Concatenates Q (384) + S (5) into a 389-dimensional hybrid feature vector.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import numpy as np

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from experiments.stage_02.label import DomainLabelRegistry, DOMAIN_DEFINITIONS
except (ImportError, ModuleNotFoundError):
    from label import DomainLabelRegistry, DOMAIN_DEFINITIONS


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

        # Ensure K is L2-normalized: ||K_i||_2 = 1
        k_norms = np.linalg.norm(self.key_matrix, axis=1, keepdims=True)
        k_norms[k_norms == 0] = 1.0
        self.K_norm = self.key_matrix / k_norms  # Shape: (num_labels, 384)
        self.num_labels = len(self.K_norm)
        self.embedding_dim = self.K_norm.shape[1]

        self.similarity_feature_names = [f"sim_{k}" for k in self.label_keys]

    def compute_similarity(self, Q: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity: S = Q_norm @ K_norm.T
        Args:
            Q: Query embeddings array of shape (N, 384) or (384,)
        Returns:
            Cosine similarity scores of shape (N, num_labels)
        """
        Q_arr = np.asarray(Q, dtype=np.float32)
        is_single = (Q_arr.ndim == 1)
        if is_single:
            Q_arr = Q_arr.reshape(1, -1)

        # L2-normalize queries
        q_norms = np.linalg.norm(Q_arr, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        Q_norm = Q_arr / q_norms

        # S = Q_norm @ K_norm.T  -> shape (N, num_labels)
        similarities = np.dot(Q_norm, self.K_norm.T)

        if is_single:
            return similarities[0]
        return similarities

    def compute_attention_weights(self, Q: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        """
        Computes scaled dot-product softmax attention distribution across domain labels:
        alpha = softmax(S / sqrt(d))
        """
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

    def augment_features(
        self,
        Q: np.ndarray,
        include_attention_weights: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenates raw dense embedding Q (384-d) with domain similarity scores S (5-d).
        Optionally appends softmax attention weights (5-d).

        Returns:
            Hybrid feature array: shape (N, 389) [or (N, 394)]
            Feature column names list
        """
        Q_arr = np.asarray(Q, dtype=np.float32)
        if Q_arr.ndim == 1:
            Q_arr = Q_arr.reshape(1, -1)

        sims = self.compute_similarity(Q_arr)

        feature_blocks = [Q_arr, sims]
        raw_names = [f"emb_{i}" for i in range(self.embedding_dim)]
        feature_names = raw_names + self.similarity_feature_names

        if include_attention_weights:
            attn_weights = self.compute_attention_weights(Q_arr)
            feature_blocks.append(attn_weights)
            feature_names += [f"attn_{k}" for k in self.label_keys]

        hybrid = np.hstack(feature_blocks)
        return hybrid, feature_names


def map_val_to_severity(
    val: Union[float, np.ndarray, torch.Tensor],
    thresholds: Tuple[float, float, float, float] = (1.5, 2.5, 3.5, 4.5)
) -> Union[int, np.ndarray]:
    """
    Maps continuous linear score `val` to discrete risk severity (1 to 5)
    based on the exact range specification:
        - If val < 1.5      (i.e. (0, 1.5)) -> Severity 1
        - If 1.5 <= val < 2.5               -> Severity 2
        - If 2.5 <= val < 3.5               -> Severity 3
        - If 3.5 <= val < 4.5               -> Severity 4
        - If val >= 4.5                     -> Severity 5
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


class DomainLinearDeepModel(nn.Module):
    """
    Deep Learning Linear Neuron that computes:
        val = w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4 + w5 * l5 + b
    and maps the continuous value to discrete risk severity (1 to 5) using:
        val in (0, 1.5)   -> 1
        val in [1.5, 2.5) -> 2
        val in [2.5, 3.5) -> 3
        val in [3.5, 4.5) -> 4
        val >= 4.5       -> 5
    """

    def __init__(
        self,
        num_features: int = 5,
        feature_names: Optional[List[str]] = None,
        thresholds: Tuple[float, float, float, float] = (1.5, 2.5, 3.5, 4.5)
    ):
        super().__init__()
        self.num_features = num_features
        self.feature_names = feature_names or [
            "sim_emergency_risk",
            "sim_authorization_access_risk",
            "sim_data_privacy_risk",
            "sim_operational_policy_deviation",
            "sim_routine_safe_query"
        ]
        self.thresholds = thresholds
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing continuous value:
            val = sum_i(w_i * l_i) + b
        Args:
            x: Tensor of shape (batch_size, num_features) or (num_features,)
        Returns:
            Tensor of shape (batch_size, 1) or (1,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.linear(x)

    def predict_continuous(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Returns continuous score val as a numpy array."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            preds = self.forward(x)
            return preds.squeeze().numpy()

    def predict_discrete(
        self,
        x: Union[np.ndarray, torch.Tensor],
        thresholds: Optional[Tuple[float, float, float, float]] = None
    ) -> np.ndarray:
        """Computes val and maps to discrete severity level 1-5."""
        th = thresholds or self.thresholds
        cont_preds = self.predict_continuous(x)
        return map_val_to_severity(cont_preds, thresholds=th)

    def get_weights_dict(self) -> Dict[str, float]:
        """Returns learned weights mapped to feature names."""
        w = self.linear.weight.detach().cpu().numpy().squeeze()
        return {name: float(weight) for name, weight in zip(self.feature_names, w)}

    def get_bias(self) -> float:
        """Returns learned bias term b."""
        return float(self.linear.bias.detach().cpu().item())

    def get_equation_string(self) -> str:
        """Returns human-readable algebraic equation string."""
        terms = [f"({w:+.4f} * {name})" for name, w in self.get_weights_dict().items()]
        bias_str = f"{self.get_bias():+.4f}"
        return f"val = {' + '.join(terms)} + {bias_str}"

    def explain_sample(
        self,
        l_values: Union[List[float], np.ndarray],
        thresholds: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, Any]:
        """Returns a breakdown of term contributions and discrete mapping for a sample."""
        th = thresholds or self.thresholds
        w_dict = self.get_weights_dict()
        b = self.get_bias()
        l_arr = np.asarray(l_values, dtype=np.float32)

        contributions = {}
        total_val = b
        for i, (name, w) in enumerate(w_dict.items()):
            val_i = float(l_arr[i])
            contrib = w * val_i
            contributions[name] = {
                "label_value": val_i,
                "weight": w,
                "term_contribution": contrib
            }
            total_val += contrib

        severity = map_val_to_severity(total_val, thresholds=th)

        if total_val < th[0]:
            interval = f"< {th[0]}"
        elif total_val < th[1]:
            interval = f"[{th[0]}, {th[1]})"
        elif total_val < th[2]:
            interval = f"[{th[1]}, {th[2]})"
        elif total_val < th[3]:
            interval = f"[{th[2]}, {th[3]})"
        else:
            interval = f">= {th[3]}"

        return {
            "formula_value": float(total_val),
            "bias": b,
            "term_contributions": contributions,
            "matching_interval": interval,
            "predicted_severity": int(severity)
        }


class DomainDeepMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for capturing non-linear cross-domain risk interactions:
        Input(5) -> Linear(16) -> ReLU -> Linear(8) -> ReLU -> Linear(1)
    """

    def __init__(
        self,
        num_features: int = 5,
        thresholds: Tuple[float, float, float, float] = (1.5, 2.5, 3.5, 4.5)
    ):
        super().__init__()
        self.thresholds = thresholds
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def predict_continuous(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            preds = self.forward(x)
            return preds.squeeze().numpy()

    def predict_discrete(
        self,
        x: Union[np.ndarray, torch.Tensor],
        thresholds: Optional[Tuple[float, float, float, float]] = None
    ) -> np.ndarray:
        th = thresholds or self.thresholds
        cont = self.predict_continuous(x)
        return map_val_to_severity(cont, thresholds=th)


def main():
    print("=" * 80)
    print("Domain-Aware Attention & Similarity Module Verification")
    print("=" * 80)

    attention = DomainAttention()
    print(f"Domain labels loaded: {attention.label_names}")
    print(f"Key Matrix Shape    : {attention.key_matrix.shape}")

    # Test with synthetic test queries
    test_queries = {
        "Privilege Escalation": "User requests sudo chmod 777 /etc/shadow and extract administrator root passwords.",
        "Routine Query": "What is the capital of France and what time is the next train to Paris?",
        "Medical Data Leak": "Download full electronic health records of patient John Doe and email laboratory results.",
        "System Emergency": "Critical reactor overheating warning: emergency shutdown command failed, physical safety hazard!"
    }

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for desc, text in test_queries.items():
        q_vec = embedder.encode(text)
        sims = attention.compute_similarity(q_vec)
        attn = attention.compute_attention_weights(q_vec)
        hybrid, names = attention.augment_features(q_vec)

        print(f"\n[Test Prompt: '{desc}']")
        print(f"Text: \"{text}\"")
        print(f"Hybrid Feature Dimension: {hybrid.shape[1]} (384 raw + 5 similarity)")
        print("Domain Similarity Scores:")
        for name, s, a in zip(attention.label_names, sims, attn):
            print(f"  • {name:<30}: Cosine Sim = {s:+.4f} | Softmax Weight = {a:.4f}")


if __name__ == "__main__":
    main()
