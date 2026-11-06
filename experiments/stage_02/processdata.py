"""
Stage 02: Dataset Preprocessing - Domain Label Values Only (No Raw Embeddings).

Loads raw SemEval 2027 AgentRisk dataset, computes domain cosine similarities Q @ K^T
and softmax attention weights using the 5 static key vectors from label.py, and creates
a clean, lightweight tabular dataset containing ONLY the domain label features (no emb_0..emb_383).

Saves to:
  - experiments/stage_02/data/processed/train_domain_values.csv / .parquet
  - experiments/stage_02/data/processed/dev_domain_values.csv / .parquet
  (and updates train_domain_augmented.csv to mirror the label-only format).
"""

import os
import sys
import types
from pathlib import Path
from typing import Tuple, Optional, List
import json
import numpy as np
import pandas as pd

# Offline SentenceTransformer setup
os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer

STAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE_DIR.parent.parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from experiments.stage_02.label import DomainLabelRegistry
    from experiments.stage_02.attention import DomainAttention
except (ImportError, ModuleNotFoundError):
    from label import DomainLabelRegistry
    from attention import DomainAttention


def locate_root_raw_data() -> Tuple[Path, Path]:
    """Finds train.jsonl and dev.jsonl starting from root path."""
    candidates = [
        ROOT_DIR / "data" / "agentrisk-data-2027" / "data",
        ROOT_DIR / "experiments" / "stage_01" / "data" / "raw_data",
        ROOT_DIR / "data" / "raw"
    ]
    for c in candidates:
        tr = c / "train.jsonl"
        dv = c / "dev.jsonl"
        if tr.exists() and dv.exists():
            return tr, dv
    raise FileNotFoundError("Could not find raw train.jsonl and dev.jsonl under root data/ paths.")


def locate_cached_embeddings() -> Tuple[Optional[Path], Optional[Path]]:
    """Loads base embeddings if cached in stage_01 to avoid re-encoding text."""
    candidates = [
        ROOT_DIR / "experiments" / "stage_01" / "data" / "processed_data"
    ]
    for c in candidates:
        tr = c / "train_embedded.csv"
        dv = c / "dev_embedded.csv"
        if tr.exists() and dv.exists():
            return tr, dv
    return None, None


def build_combined_text(row: pd.Series) -> str:
    interaction = str(row.get("formatted_interaction", "") or "").strip()
    action = str(row.get("proposed_action", "") or "").strip()
    if action:
        return f"{interaction} [PROPOSED_ACTION]: {action}"
    return interaction


def process_domain_values_dataset(
    output_dir: Path = STAGE_DIR / "data" / "processed",
    include_attention_weights: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs dataset containing ONLY domain label values (similarities & attention weights).
    Completely removes all 384 raw embedding columns.
    """
    print("=" * 85)
    print("Stage 02: Building Domain-Labels-Only Dataset (Embeddings Removed)")
    print("=" * 85)

    # 1. Initialize Domain Registry and Attention
    registry = DomainLabelRegistry()
    registry.generate_key_vectors()
    attention = DomainAttention(registry=registry)
    print(f"✓ DomainAttention initialized with {attention.num_labels} prototype risk domains.")

    # 2. Obtain base embeddings for query vectors Q (384-d)
    cached_train, cached_dev = locate_cached_embeddings()
    raw_train, raw_dev = locate_root_raw_data()

    if cached_train is not None and cached_dev is not None:
        print(f"✓ Using cached embeddings to compute domain projections from: {cached_train.parent}")
        df_train_base = pd.read_csv(cached_train)
        df_dev_base = pd.read_csv(cached_dev)

        emb_cols = [c for c in df_train_base.columns if c.startswith("emb_")]
        X_train_emb = df_train_base[emb_cols].values
        X_dev_emb = df_dev_base[emb_cols].values

        meta_train = df_train_base[["example_id", "risk_severity", "risk_domains", "risk_drivers"]]
        meta_dev = df_dev_base[["example_id", "risk_severity", "risk_domains", "risk_drivers"]]
    else:
        print("Computing embeddings from raw JSONL using all-MiniLM-L6-v2...")
        df_raw_train = pd.read_json(raw_train, lines=True)
        df_raw_dev = pd.read_json(raw_dev, lines=True)

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts_train = df_raw_train.apply(build_combined_text, axis=1).tolist()
        texts_dev = df_raw_dev.apply(build_combined_text, axis=1).tolist()

        X_train_emb = embedder.encode(texts_train, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        X_dev_emb = embedder.encode(texts_dev, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

        meta_train = pd.DataFrame({
            "example_id": df_raw_train["example_id"].values,
            "risk_severity": df_raw_train["risk_severity"].astype(int).values,
            "risk_domains": df_raw_train["risk_domains"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x or "")),
            "risk_drivers": df_raw_train["risk_drivers"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x or ""))
        })
        meta_dev = pd.DataFrame({
            "example_id": df_raw_dev["example_id"].values,
            "risk_severity": df_raw_dev["risk_severity"].astype(int).values,
            "risk_domains": df_raw_dev["risk_domains"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x or "")),
            "risk_drivers": df_raw_dev["risk_drivers"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x or ""))
        })

    # 3. Compute Domain Cosine Similarities: Q_norm @ K_norm.T  (Shape: N x 5)
    print("Computing domain label values (cosine similarities)...")
    sims_train = attention.compute_similarity(X_train_emb)
    sims_dev = attention.compute_similarity(X_dev_emb)

    sim_cols = [f"sim_{k}" for k in attention.label_keys]
    sim_df_train = pd.DataFrame(sims_train, columns=sim_cols)
    sim_df_dev = pd.DataFrame(sims_dev, columns=sim_cols)

    feature_blocks_train = [sim_df_train]
    feature_blocks_dev = [sim_df_dev]

    # Optionally add softmax attention weights over the labels
    if include_attention_weights:
        attn_train = attention.compute_attention_weights(X_train_emb)
        attn_dev = attention.compute_attention_weights(X_dev_emb)
        attn_cols = [f"attn_{k}" for k in attention.label_keys]
        feature_blocks_train.append(pd.DataFrame(attn_train, columns=attn_cols))
        feature_blocks_dev.append(pd.DataFrame(attn_dev, columns=attn_cols))

    # 4. Assemble: ONLY Metadata + Domain Label Features (No emb_0..emb_383!)
    df_train_domain_only = pd.concat([meta_train.reset_index(drop=True)] + feature_blocks_train, axis=1)
    df_dev_domain_only = pd.concat([meta_dev.reset_index(drop=True)] + feature_blocks_dev, axis=1)

    # 5. Export clean datasets
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to train_domain_values and also overwrite train_domain_augmented so all scripts use it
    for name in ["train_domain_values", "train_domain_augmented"]:
        df_train_domain_only.to_csv(output_dir / f"{name}.csv", index=False)
        df_train_domain_only.to_parquet(output_dir / f"{name}.parquet", index=False)

    for name in ["dev_domain_values", "dev_domain_augmented"]:
        df_dev_domain_only.to_csv(output_dir / f"{name}.csv", index=False)
        df_dev_domain_only.to_parquet(output_dir / f"{name}.parquet", index=False)

    total_feature_cols = [c for c in df_train_domain_only.columns if c.startswith("sim_") or c.startswith("attn_")]
    print(f"\n✓ Saved Domain-Labels-Only Dataset to: {output_dir}")
    print(f"  • Total Columns   : {df_train_domain_only.shape[1]} (4 Metadata + {len(total_feature_cols)} Domain Features)")
    print(f"  • Raw Embeddings  : REMOVED (0 embedding columns)")
    print(f"  • Domain Features : {total_feature_cols}")

    print("\n" + "=" * 90)
    print("DOMAIN-ONLY DATASET PREVIEW (First 5 Records):")
    print("=" * 90)
    preview_cols = ["example_id", "risk_severity"] + sim_cols
    print(df_train_domain_only[preview_cols].head(5).to_string(index=False))
    print("=" * 90)

    return df_train_domain_only, df_dev_domain_only


# Backwards compatibility alias
process_and_augment_dataset = process_domain_values_dataset


def main():
    process_domain_values_dataset()


if __name__ == "__main__":
    main()
