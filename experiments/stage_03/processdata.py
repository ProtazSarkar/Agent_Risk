"""
Stage 03: Dataset Preprocessing - 394-Dimensional Hybrid Feature Representation.

Loads interaction embeddings Q (384-d), projects onto enhanced domain prototypes
yielding domain similarity scores S (5-d) and softmax attention weights (5-d),
and creates structured hybrid datasets:
  - train_hybrid.csv / .parquet: 568 rows x 398 columns (4 meta + 384 emb + 5 sim + 5 attn)
  - dev_hybrid.csv / .parquet  : 143 rows x 398 columns
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import json
import numpy as np
import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE_DIR.parent.parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

try:
    from experiments.stage_03.label import DomainLabelRegistry
    from experiments.stage_03.attention import DomainAttention
except (ImportError, ModuleNotFoundError):
    from label import DomainLabelRegistry
    from attention import DomainAttention


def locate_cached_embeddings() -> Tuple[Path, Path]:
    candidates = [
        ROOT_DIR / "experiments" / "stage_01" / "data" / "processed_data"
    ]
    for c in candidates:
        tr = c / "train_embedded.csv"
        dv = c / "dev_embedded.csv"
        if tr.exists() and dv.exists():
            return tr, dv
    raise FileNotFoundError("Could not find base embedded datasets in experiments/stage_01/data/processed_data.")


def process_hybrid_dataset(
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = output_dir or (STAGE_DIR / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("Stage 03: Generating 394-Dimensional Hybrid Tabular Datasets")
    print("=" * 85)

    # 1. Initialize Registry and Domain Attention
    registry = DomainLabelRegistry()
    registry.generate_key_vectors()
    attention = DomainAttention(registry=registry)
    print(f"DomainAttention initialized with {attention.num_labels} enhanced prototypes.")

    # 2. Load cached base embeddings
    tr_emb_path, dv_emb_path = locate_cached_embeddings()
    print(f"Loading base embeddings from:\n  • Train: {tr_emb_path}\n  • Dev  : {dv_emb_path}")

    df_tr_base = pd.read_csv(tr_emb_path)
    df_dv_base = pd.read_csv(dv_emb_path)

    emb_cols = [c for c in df_tr_base.columns if c.startswith("emb_")]
    meta_cols = ["example_id", "risk_severity", "risk_domains", "risk_drivers"]

    Q_tr = df_tr_base[emb_cols].values.astype(np.float32)
    Q_dv = df_dv_base[emb_cols].values.astype(np.float32)

    # 3. Compute domain projections
    print(f"Projecting {len(Q_tr)} training interactions onto domain risk prototypes...")
    sims_tr = attention.compute_similarity(Q_tr)
    attn_tr = attention.compute_attention_weights(Q_tr)

    print(f"Projecting {len(Q_dv)} dev interactions onto domain risk prototypes...")
    sims_dv = attention.compute_similarity(Q_dv)
    attn_dv = attention.compute_attention_weights(Q_dv)

    sim_cols = attention.similarity_feature_names
    attn_cols = attention.attention_feature_names

    # 4. Construct DataFrames
    df_sim_tr = pd.DataFrame(sims_tr, columns=sim_cols)
    df_attn_tr = pd.DataFrame(attn_tr, columns=attn_cols)
    df_train_hybrid = pd.concat([df_tr_base[meta_cols], df_tr_base[emb_cols], df_sim_tr, df_attn_tr], axis=1)

    df_sim_dv = pd.DataFrame(sims_dv, columns=sim_cols)
    df_attn_dv = pd.DataFrame(attn_dv, columns=attn_cols)
    df_dev_hybrid = pd.concat([df_dv_base[meta_cols], df_dv_base[emb_cols], df_sim_dv, df_attn_dv], axis=1)

    # 5. Save outputs
    tr_csv = out_dir / "train_hybrid.csv"
    tr_pq = out_dir / "train_hybrid.parquet"
    dv_csv = out_dir / "dev_hybrid.csv"
    dv_pq = out_dir / "dev_hybrid.parquet"

    df_train_hybrid.to_csv(tr_csv, index=False)
    df_train_hybrid.to_parquet(tr_pq, index=False)
    df_dev_hybrid.to_csv(dv_csv, index=False)
    df_dev_hybrid.to_parquet(dv_pq, index=False)

    print(f"\n[Saved Hybrid Datasets]")
    print(f"  • Train: {tr_csv} | Shape: {df_train_hybrid.shape}")
    print(f"  • Dev  : {dv_csv} | Shape: {df_dev_hybrid.shape}")
    print(f"  • Feature breakdown: 384 raw embeddings + 5 similarities + 5 attention weights = 394 features")

    return df_train_hybrid, df_dev_hybrid


if __name__ == "__main__":
    process_hybrid_dataset()

