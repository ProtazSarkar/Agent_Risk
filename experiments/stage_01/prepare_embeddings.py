"""
Data Embedding and Preprocessing Pipeline for SemEval 2027 AgentRisk.

Transforms raw interaction logs (train.jsonl, dev.jsonl) into structured tabular datasets
where one side contains the 384-dimensional dense semantic embeddings (emb_0 to emb_383)
and the other side contains the target risk severity level (1 to 5).
"""

import os
import sys
import types
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Workaround for huggingface offline mode and dependency version checks on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer


def build_combined_text(row: pd.Series) -> str:
    """Combines formatted interaction history and proposed action into a single prompt."""
    interaction = str(row.get("formatted_interaction", "") or "").strip()
    action = str(row.get("proposed_action", "") or "").strip()
    if action:
        return f"{interaction} [PROPOSED_ACTION]: {action}"
    return interaction


def encode_dataset(
    df: pd.DataFrame,
    embedder: SentenceTransformer,
    dataset_name: str = "dataset"
) -> pd.DataFrame:
    """
    Encodes text features into dense 384-d vectors and builds a structured DataFrame.
    Layout:
      - Left side: Metadata & Target (example_id, risk_severity, risk_domains, risk_drivers)
      - Right side: Dense Embeddings (emb_0 through emb_383)
    """
    print(f"\n[{dataset_name.upper()}] Preparing and embedding {len(df)} records...")
    texts = df.apply(build_combined_text, axis=1).tolist()

    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    num_dims = embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(num_dims)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)

    # Format risk_domains and risk_drivers as JSON strings if they are lists
    domains_col = df["risk_domains"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x or ""))
    drivers_col = df["risk_drivers"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x or ""))

    meta_df = pd.DataFrame({
        "example_id": df["example_id"].values,
        "risk_severity": df["risk_severity"].astype(int).values,
        "risk_domains": domains_col.values,
        "risk_drivers": drivers_col.values,
    }, index=df.index)

    # Merge side-by-side: metadata & target on left, embeddings on right
    processed_df = pd.concat([meta_df, emb_df], axis=1)
    return processed_df


def prepare_data(
    raw_train_path: Path,
    raw_dev_path: Path,
    output_dirs: list[Path],
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw JSONL datasets, extracts embeddings, and exports CSV & Parquet files.
    """
    print("=" * 80)
    print("AgentRisk SemEval 2027: Embedding & Dataset Preprocessing Pipeline")
    print("=" * 80)
    print(f"Loading SentenceTransformer: {model_name}...")
    embedder = SentenceTransformer(model_name)
    print("SentenceTransformer loaded successfully.")

    # 1. Process Train Set
    print(f"\nReading raw train data from: {raw_train_path}")
    df_raw_train = pd.read_json(raw_train_path, lines=True)
    df_train_proc = encode_dataset(df_raw_train, embedder, "train")

    # 2. Process Dev Set
    print(f"\nReading raw dev data from: {raw_dev_path}")
    df_raw_dev = pd.read_json(raw_dev_path, lines=True)
    df_dev_proc = encode_dataset(df_raw_dev, embedder, "dev")

    # 3. Save to specified output directories
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        train_csv = out_dir / "train_embedded.csv"
        dev_csv = out_dir / "dev_embedded.csv"
        train_parquet = out_dir / "train_embedded.parquet"
        dev_parquet = out_dir / "dev_embedded.parquet"

        df_train_proc.to_csv(train_csv, index=False)
        df_dev_proc.to_csv(dev_csv, index=False)
        df_train_proc.to_parquet(train_parquet, index=False)
        df_dev_proc.to_parquet(dev_parquet, index=False)
        print(f" Saved CSV and Parquet files to: {out_dir}")

    # Display dataset preview
    print("\n" + "=" * 80)
    print("PROCESSED DATASET STRUCTURE PREVIEW (Train)")
    print("=" * 80)
    preview_cols = ["example_id", "risk_severity", "emb_0", "emb_1", "emb_2", "emb_3", "emb_382", "emb_383"]
    available_preview = [c for c in preview_cols if c in df_train_proc.columns]
    print(df_train_proc[available_preview].head(5).to_string())
    print("\nDataset Summary:")
    print(f"  • Train Set : {df_train_proc.shape[0]} samples x {df_train_proc.shape[1]} columns")
    print(f"  • Dev Set   : {df_dev_proc.shape[0]} samples x {df_dev_proc.shape[1]} columns")
    print(f"  • Feature Columns : emb_0 to emb_383 (384 dimensions)")
    print(f"  • Target Column   : risk_severity (Levels 1 to 5)")
    print("  • Risk Severity Class Breakdown (Train):")
    for sev, cnt in df_train_proc["risk_severity"].value_counts().sort_index().items():
        pct = (cnt / len(df_train_proc)) * 100
        print(f"      Level {sev}: {cnt:4d} samples ({pct:5.1f}%)")

    return df_train_proc, df_dev_proc


def main():
    parser = argparse.ArgumentParser(description="Embed AgentRisk interactions and export to CSV/Parquet")
    parser.add_argument(
        "--train-raw",
        type=str,
        default="experiments/stage_01/data/raw_data/train.jsonl",
        help="Path to raw train.jsonl"
    )
    parser.add_argument(
        "--dev-raw",
        type=str,
        default="experiments/stage_01/data/raw_data/dev.jsonl",
        help="Path to raw dev.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/stage_01/data/processed_data",
        help="Primary output directory for processed files"
    )
    args = parser.parse_args()

    train_path = Path(args.train_raw)
    dev_path = Path(args.dev_raw)
    output_dir = Path(args.output_dir)
    # Also support user's misspelled directory if it exists
    alt_dir = Path("experiments/stage_01/data/prcessed_data")
    dest_dirs = [output_dir]
    if alt_dir.exists():
        dest_dirs.append(alt_dir)

    prepare_data(train_path, dev_path, dest_dirs)


if __name__ == "__main__":
    main()

