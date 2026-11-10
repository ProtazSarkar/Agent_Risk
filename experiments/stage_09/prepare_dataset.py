"""
Stage 09: 5-Channel Disentangled Severity Feature Generator.

Ingests the 5-channel interaction representations from Stage 08 and evaluates
them through all 11 Stage 08 Cross-Attentive Prototype Domain Classifiers:
  - domain_01: No Material Harm (Safe baseline)
  - domain_02: Privacy / Confidentiality
  - domain_03: Security / System Integrity
  - domain_04: Financial / Assets
  - domain_05: Operational / Service
  - domain_06: Physical / Health
  - domain_07: Legal / Compliance
  - domain_08: Reputational / Social / Psychological
  - domain_09: Discrimination / Fairness
  - domain_10: Environmental
  - domain_11: Information Quality

Extracts 39 high-fidelity ordinal risk signals:
  1. 11 Calibrated Domain Probabilities (P(Safe) for dom 01, P(Harm) for doms 02..11)
  2. 11 Standardized Harm Probabilities (with dom 01 inverted to P(Harm) = 1 - P(Safe))
  3. 5 Compound Harm Statistics (max_harm, sum_harm, mean_harm, std_harm, safe_prob)
  4. 3 Multi-Domain Activation Threshold Counts (>0.3, >0.5, >0.7)
  5. 3 Pairwise Channel Cosine Similarities (query-action, thought-action, obs-action)
  6. 5 Channel Attention Pooling Weights (measuring where the model looked across the 5 channels)
  7. 1 Prototype Margin (positive prototype alignment minus negative prototype alignment)

Exports:
  - experiments/stage_09/data/train_severity_features_stage09.csv
  - experiments/stage_09/data/dev_severity_features_stage09.csv
  - experiments/stage_09/output/feature_summary.md
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

STAGE_09_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE_09_DIR.parent.parent
STAGE_08_DIR = ROOT_DIR / "experiments" / "stage_08"

# Ensure stage_08 is on sys.path to load Stage08OptimizedDomainClassifier
if str(STAGE_08_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_08_DIR))

from model import Stage08OptimizedDomainClassifier

DATA_DIR = STAGE_09_DIR / "data"
OUTPUT_DIR = STAGE_09_DIR / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_DEFS = [
    ("domain_01", "no_material_harm"),
    ("domain_02", "privacy"),
    ("domain_03", "security"),
    ("domain_04", "financial"),
    ("domain_05", "operational"),
    ("domain_06", "physical"),
    ("domain_07", "legal"),
    ("domain_08", "reputational"),
    ("domain_09", "discrimination"),
    ("domain_10", "environmental"),
    ("domain_11", "information_quality")
]

FEATURE_COLUMNS = [
    "prob_dom01_safe",
    "prob_dom02_privacy",
    "prob_dom03_security",
    "prob_dom04_financial",
    "prob_dom05_operational",
    "prob_dom06_physical",
    "prob_dom07_legal",
    "prob_dom08_reputational",
    "prob_dom09_discrimination",
    "prob_dom10_environmental",
    "prob_dom11_information_quality",
    "harm_dom01",
    "harm_dom02_privacy",
    "harm_dom03_security",
    "harm_dom04_financial",
    "harm_dom05_operational",
    "harm_dom06_physical",
    "harm_dom07_legal",
    "harm_dom08_reputational",
    "harm_dom09_discrimination",
    "harm_dom10_environmental",
    "harm_dom11_information_quality",
    "max_harm_prob",
    "sum_harm_prob",
    "mean_harm_prob",
    "std_harm_prob",
    "safe_prob",
    "count_domains_gt_03",
    "count_domains_gt_05",
    "count_domains_gt_07",
    "sim_query_action",
    "sim_thought_action",
    "sim_obs_action",
    "pool_weight_query",
    "pool_weight_thought",
    "pool_weight_context",
    "pool_weight_obs",
    "pool_weight_action",
    "sim_margin_dom01"
]


def load_all_11_stage08_models() -> Dict[str, Stage08OptimizedDomainClassifier]:
    """Loads all 11 trained 5-channel domain classifiers from stage_08/models/."""
    models = {}
    models_dir = STAGE_08_DIR / "models"
    print("Loading 11 trained 5-channel domain classifiers from Stage 08...")
    for dom_id, _ in DOMAIN_DEFS:
        model_path = models_dir / f"best_{dom_id}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        checkpoint = torch.load(model_path, weights_only=False)
        m = Stage08OptimizedDomainClassifier(
            initial_prototype_keys=checkpoint["prototype_keys"],
            hidden_dim=checkpoint.get("hidden_dim", 384)
        )
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        models[dom_id] = m
    print("✓ All 11 Stage 08 domain models loaded successfully into memory.")
    return models


def extract_features_from_5channels(
    channels_tensor: torch.Tensor,
    models: Dict[str, Stage08OptimizedDomainClassifier]
) -> np.ndarray:
    """
    Extracts 39 ordinal risk signals from 5-channel interaction tensors.

    Args:
        channels_tensor: (N, 5, 384) [query, thought, context_actions, observations, proposed_action]
        models: Dictionary of 11 loaded Stage 08 domain models

    Returns:
        feats: (N, 39) numpy array of extracted features
    """
    with torch.no_grad():
        all_probs = []
        all_pool_weights = []
        all_pos_sims = []
        all_neg_sims = []

        for dom_id, _ in DOMAIN_DEFS:
            prob, logits, meta = models[dom_id](channels_tensor)
            all_probs.append(prob.cpu().numpy())
            all_pool_weights.append(meta["channel_pool_weights"].cpu().numpy())
            all_pos_sims.append(meta["pos_sim_mean"].cpu().numpy())
            all_neg_sims.append(meta["neg_sim_mean"].cpu().numpy())

        # (N, 11) calibrated probabilities
        all_probs = np.stack(all_probs, axis=1)

        # Standardized harm probabilities (N, 11)
        # Note: domain_01 prob is Safe prob, so Harm prob for dom 01 is 1 - prob
        harm_probs = np.copy(all_probs)
        harm_probs[:, 0] = 1.0 - all_probs[:, 0]

        # Multi-domain aggregates across harm domains (domains 02-11)
        max_harm = np.max(harm_probs[:, 1:], axis=1, keepdims=True)
        sum_harm = np.sum(harm_probs[:, 1:], axis=1, keepdims=True)
        mean_harm = np.mean(harm_probs[:, 1:], axis=1, keepdims=True)
        std_harm = np.std(harm_probs[:, 1:], axis=1, keepdims=True)

        # Safe probability (Domain 01 calibrated output)
        safe_prob = all_probs[:, 0:1]

        # Multi-domain active counts
        count_gt_03 = np.sum(harm_probs[:, 1:] > 0.3, axis=1, keepdims=True)
        count_gt_05 = np.sum(harm_probs[:, 1:] > 0.5, axis=1, keepdims=True)
        count_gt_07 = np.sum(harm_probs[:, 1:] > 0.7, axis=1, keepdims=True)

        # Pairwise channel cosine similarities:
        # Channel 0: query, 1: thought, 2: context, 3: observation, 4: proposed_action
        ch_norm = F.normalize(channels_tensor, p=2, dim=-1)
        sim_query_action = torch.sum(ch_norm[:, 0] * ch_norm[:, 4], dim=-1, keepdim=True).cpu().numpy()
        sim_thought_action = torch.sum(ch_norm[:, 1] * ch_norm[:, 4], dim=-1, keepdim=True).cpu().numpy()
        sim_obs_action = torch.sum(ch_norm[:, 3] * ch_norm[:, 4], dim=-1, keepdim=True).cpu().numpy()

        # Channel pool weights for domain 01 (5 weights)
        pool_w_dom01 = all_pool_weights[0]  # (N, 5)

        # Similarity margin for domain 01 (pos_sim - neg_sim)
        sim_margin_dom01 = (all_pos_sims[0] - all_neg_sims[0])[:, None]

        # Concatenate 39 features
        feats = np.hstack([
            all_probs,           # 11
            harm_probs,          # 11
            max_harm,            # 1
            sum_harm,            # 1
            mean_harm,           # 1
            std_harm,            # 1
            safe_prob,           # 1
            count_gt_03,         # 1
            count_gt_05,         # 1
            count_gt_07,         # 1
            sim_query_action,    # 1
            sim_thought_action,  # 1
            sim_obs_action,      # 1
            pool_w_dom01,        # 5
            sim_margin_dom01     # 1
        ])
    return feats


def generate_feature_dataset(
    split_path: Path,
    models: Dict[str, Stage08OptimizedDomainClassifier],
    out_csv: Path
) -> pd.DataFrame:
    """Loads 5-channel split tensor, extracts 39 features, and exports CSV."""
    print(f"\nProcessing {split_path.name}...")
    data_payload = torch.load(split_path, weights_only=False)
    channels_tensor = data_payload["channels_tensor"]
    example_ids = data_payload["example_ids"]
    severities = data_payload["severities"]

    start = time.time()
    feats = extract_features_from_5channels(channels_tensor, models)
    elapsed = time.time() - start
    print(f"Extracted {feats.shape[1]} features for {feats.shape[0]} samples in {elapsed:.2f}s ({len(feats)/elapsed:.1f} samples/s).")

    df = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    df.insert(0, "example_id", example_ids)
    df.insert(1, "target_severity", severities)

    df.to_csv(out_csv, index=False)
    print(f"✓ Saved features to {out_csv} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df


def write_feature_summary(df_train: pd.DataFrame, df_dev: pd.DataFrame, out_path: Path):
    """Writes detailed feature summary markdown report."""
    md = []
    md.append("# Stage 09: 5-Channel Disentangled Severity Feature Suite\n")
    md.append("This document summarizes the 39 tabular features extracted from the 11 trained Stage 08 5-channel models.\n")
    md.append("## Dataset Overview\n")
    md.append(f"- **Train Samples**: {len(df_train)}")
    md.append(f"- **Dev Samples**: {len(df_dev)}")
    md.append(f"- **Total Extracted Features**: {len(FEATURE_COLUMNS)}\n")

    md.append("## Severity Class Distribution\n")
    md.append("| Severity Level | Train Count | Train % | Dev Count | Dev % |")
    md.append("|---|---|---|---|---|")
    tr_counts = df_train["target_severity"].value_counts().to_dict()
    dev_counts = df_dev["target_severity"].value_counts().to_dict()
    for s in range(1, 6):
        c_tr = tr_counts.get(s, 0)
        p_tr = (c_tr / len(df_train)) * 100
        c_dev = dev_counts.get(s, 0)
        p_dev = (c_dev / len(df_dev)) * 100
        md.append(f"| Level {s} | {c_tr} | {p_tr:.2f}% | {c_dev} | {p_dev:.2f}% |")

    md.append("\n## Top Feature Correlations with Severity (Pearson r)\n")
    md.append("| Feature Name | Pearson Correlation (r) | Feature Category |")
    md.append("|---|---|---|")
    corrs = []
    for col in FEATURE_COLUMNS:
        r = df_train[col].corr(df_train["target_severity"])
        corrs.append((col, r))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, r in corrs[:15]:
        cat = "Harm Probability" if "harm" in col or "prob" in col else "Channel Signal"
        md.append(f"| `{col}` | {r:+.4f} | {cat} |")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ Feature summary written to {out_path}")


def main():
    print("=" * 70)
    print("Stage 09: 5-Channel Disentangled Severity Feature Suite Preparation")
    print("=" * 70)

    train_pt = STAGE_08_DIR / "data" / "train_5channels.pt"
    dev_pt = STAGE_08_DIR / "data" / "dev_5channels.pt"

    if not train_pt.exists() or not dev_pt.exists():
        raise FileNotFoundError(f"Missing Stage 08 5-channel tensors at {STAGE_08_DIR / 'data'}")

    models = load_all_11_stage08_models()

    train_csv = DATA_DIR / "train_severity_features_stage09.csv"
    dev_csv = DATA_DIR / "dev_severity_features_stage09.csv"

    df_train = generate_feature_dataset(train_pt, models, train_csv)
    df_dev = generate_feature_dataset(dev_pt, models, dev_csv)

    summary_md = OUTPUT_DIR / "feature_summary.md"
    write_feature_summary(df_train, df_dev, summary_md)
    print("\n✓ Stage 09 Feature Dataset Preparation completed successfully!")


if __name__ == "__main__":
    main()

