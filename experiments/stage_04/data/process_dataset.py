"""
Stage 04: Processed Dataset Generator for Domain 01 (No Material Harm).

Loads the 384-dimensional dense semantic embeddings (emb_0 to emb_383)
and adds the binary target label for 'No Material Harm':
  - label = 1 if 'No Material Harm' is in risk_domains
  - label = 0 otherwise

Saves:
  - experiments/stage_04/data/train_no_material_harm.csv / .parquet
  - experiments/stage_04/data/dev_no_material_harm.csv / .parquet
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

STAGE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = STAGE_DIR.parent.parent


def locate_base_embeddings() -> Tuple[Path, Path]:
    candidates = [
        ROOT_DIR / "experiments" / "stage_01" / "data" / "processed_data",
        STAGE_DIR.parent / "stage_01" / "data" / "processed_data"
    ]
    for c in candidates:
        tr = c / "train_embedded.csv"
        dv = c / "dev_embedded.csv"
        if tr.exists() and dv.exists():
            return tr, dv
    raise FileNotFoundError("Could not locate base embedded CSV files in stage_01.")


def extract_no_material_harm_label(series: pd.Series) -> pd.Series:
    """Returns 1 if 'No Material Harm' is in the risk_domains entry, else 0."""
    return series.astype(str).str.contains("No Material Harm", case=False).astype(int)


def build_no_material_harm_dataset(
    output_dir: Path = STAGE_DIR / "data"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tr_path, dv_path = locate_base_embeddings()
    print("=" * 80)
    print("Stage 04: Generating Embeddings-Only Dataset for 'No Material Harm'")
    print("=" * 80)
    print(f"Loading base embeddings from:\n  • Train: {tr_path}\n  • Dev  : {dv_path}")

    df_tr_raw = pd.read_csv(tr_path)
    df_dv_raw = pd.read_csv(dv_path)

    emb_cols = [c for c in df_tr_raw.columns if c.startswith("emb_")]
    if len(emb_cols) != 384:
        raise ValueError(f"Expected 384 embedding columns, found {len(emb_cols)}.")

    # Target label: 1 if No Material Harm else 0
    y_tr = extract_no_material_harm_label(df_tr_raw["risk_domains"])
    y_dv = extract_no_material_harm_label(df_dv_raw["risk_domains"])

    # Construct clean dataset with ONLY example_id, emb_0..emb_383, and label
    df_train = pd.concat([
        df_tr_raw[["example_id"]],
        df_tr_raw[emb_cols],
        pd.Series(y_tr, name="label")
    ], axis=1)

    df_dev = pd.concat([
        df_dv_raw[["example_id"]],
        df_dv_raw[emb_cols],
        pd.Series(y_dv, name="label")
    ], axis=1)

    # Save to CSV and Parquet
    tr_csv = output_dir / "train_no_material_harm.csv"
    tr_pq = output_dir / "train_no_material_harm.parquet"
    dv_csv = output_dir / "dev_no_material_harm.csv"
    dv_pq = output_dir / "dev_no_material_harm.parquet"

    df_train.to_csv(tr_csv, index=False)
    df_train.to_parquet(tr_pq, index=False)
    df_dev.to_csv(dv_csv, index=False)
    df_dev.to_parquet(dv_pq, index=False)

    print(f"\n[Generated Datasets]")
    print(f"  • Train: {tr_csv} | Shape: {df_train.shape}")
    print(f"    - Positive ('No Material Harm'): {(y_tr == 1).sum()} ({(y_tr == 1).mean()*100:.1f}%)")
    print(f"    - Negative (Other domains)    : {(y_tr == 0).sum()} ({(y_tr == 0).mean()*100:.1f}%)")
    print(f"  • Dev  : {dv_csv} | Shape: {df_dev.shape}")
    print(f"    - Positive ('No Material Harm'): {(y_dv == 1).sum()} ({(y_dv == 1).mean()*100:.1f}%)")
    print(f"    - Negative (Other domains)    : {(y_dv == 0).sum()} ({(y_dv == 0).mean()*100:.1f}%)")
    print(f"  • Feature Layout: 384 dense embeddings (emb_0 to emb_383) + 1 binary classification label")

    return df_train, df_dev


if __name__ == "__main__":
    build_no_material_harm_dataset()

