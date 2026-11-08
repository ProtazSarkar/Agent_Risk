"""
Stage 05: Multi-Domain Dataset Generator & Augmentation Suite.

Processes and balances datasets for all 11 SemEval AgentRisk domains in exact order:
  01. No Material Harm
  02. Privacy / Confidentiality
  03. Security / System Integrity
  04. Financial / Assets
  05. Operational / Service
  06. Physical / Health
  07. Legal / Compliance
  08. Reputational / Social / Psychological
  09. Discrimination / Fairness
  10. Environmental
  11. Information Quality

Key Capabilities:
- Extracts real SemEval summarized samples from train_summarized_embedded.csv & dev_summarized_embedded.csv.
- Augments under-represented severity levels across all domains using SemEval-grounded synthetic examples.
- Encodes all summaries with SentenceTransformer ('all-MiniLM-L6-v2') into 384-dimensional dense vectors.
- Exports isolated, domain-specific CSVs:
    experiments/stage_05/data/train_domain_XX.csv
    experiments/stage_05/data/dev_domain_XX.csv
- Generates a complete severity distribution matrix report across all domains.
"""

import os
import sys
import types
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Workaround for huggingface offline mode and dependency version checks on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer

STAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = STAGE_DIR / "data"

try:
    from experiments.stage_05.data.domain_augmentations import AUGMENTATION_REGISTRY
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, str(DATA_DIR))
    from domain_augmentations import AUGMENTATION_REGISTRY

DOMAIN_ORDER = [
    ("domain_01", "No Material Harm"),
    ("domain_02", "Privacy / Confidentiality"),
    ("domain_03", "Security / System Integrity"),
    ("domain_04", "Financial / Assets"),
    ("domain_05", "Operational / Service"),
    ("domain_06", "Physical / Health"),
    ("domain_07", "Legal / Compliance"),
    ("domain_08", "Reputational / Social / Psychological"),
    ("domain_09", "Discrimination / Fairness"),
    ("domain_10", "Environmental"),
    ("domain_11", "Information Quality")
]


def load_base_summarized_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_path = DATA_DIR / "train_summarized_embedded.csv"
    dv_path = DATA_DIR / "dev_summarized_embedded.csv"

    if not tr_path.exists() or not dv_path.exists():
        from process_summarized_dataset import build_stage_05_dataset
        return build_stage_05_dataset(output_dir=DATA_DIR)

    return pd.read_csv(tr_path), pd.read_csv(dv_path)


def embed_augmentation_records(
    embedder: SentenceTransformer,
    aug_records: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Encodes augmented summaries with SentenceTransformer into 384-d vectors."""
    if not aug_records:
        return pd.DataFrame()
    summaries = [r["summary_text"] for r in aug_records]
    embs = embedder.encode(summaries, normalize_embeddings=True, show_progress_bar=False)

    df_meta = pd.DataFrame(aug_records)
    emb_cols = [f"emb_{i}" for i in range(embs.shape[1])]
    df_emb = pd.DataFrame(embs, columns=emb_cols)

    return pd.concat([df_meta, df_emb], axis=1)


def generate_all_domain_datasets():
    print("=" * 90)
    print("STAGE 05: GENERATING BALANCED, ISOLATED DATASETS FOR ALL 11 DOMAINS")
    print("=" * 90)

    # 1. Load Base Summarized Datasets
    df_tr_base, df_dv_base = load_base_summarized_data()
    emb_cols = [c for c in df_tr_base.columns if c.startswith("emb_")]

    print(f"Base Datasets Loaded: Train={len(df_tr_base)}, Dev={len(df_dv_base)}")

    # 2. Load Embedder for Augmentations
    print("Loading SentenceTransformer ('all-MiniLM-L6-v2') for synthetic embeddings...")
    local_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").glob("*"))
    model_source = str(local_snapshots[0]) if local_snapshots else "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_source)

    summary_stats = []

    # 3. Build & Save Each Domain's Dataset
    for dom_folder, dom_name in DOMAIN_ORDER:
        print(f"\nProcessing [{dom_folder}]: '{dom_name}'...")

        # Check if domain has augmentations
        aug_entry = AUGMENTATION_REGISTRY.get(dom_folder, None)
        df_tr_aug = embed_augmentation_records(embedder, aug_entry["train"]) if aug_entry else None
        df_dv_aug = embed_augmentation_records(embedder, aug_entry["dev"]) if aug_entry else None

        # Binary label extraction: 1 if domain in risk_domains else 0
        def prepare_split(df_base: pd.DataFrame, df_aug: pd.DataFrame = None):
            cols = ["example_id", "summary_text", "risk_severity", "risk_domains"] + emb_cols
            if df_aug is not None and not df_aug.empty:
                df_merged = pd.concat([df_base[cols], df_aug[cols]], ignore_index=True)
            else:
                df_merged = df_base[cols].copy()

            # Binary label: 1 if this domain is matched, 0 otherwise
            mask = df_merged["risk_domains"].astype(str).str.contains(dom_name, case=False, regex=False)
            df_merged["label"] = mask.astype(int)
            return df_merged

        df_tr_dom = prepare_split(df_tr_base, df_tr_aug)
        df_dv_dom = prepare_split(df_dv_base, df_dv_aug)

        # Save to CSV
        tr_out = DATA_DIR / f"train_{dom_folder}.csv"
        dv_out = DATA_DIR / f"dev_{dom_folder}.csv"
        df_tr_dom.to_csv(tr_out, index=False)
        df_dv_dom.to_csv(dv_out, index=False)

        # Statistics
        n_pos_tr = (df_tr_dom["label"] == 1).sum()
        n_neg_tr = (df_tr_dom["label"] == 0).sum()
        n_pos_dv = (df_dv_dom["label"] == 1).sum()
        n_neg_dv = (df_dv_dom["label"] == 0).sum()

        pos_sub_tr = df_tr_dom[df_tr_dom["label"] == 1]
        sev_counts_tr = [int((pos_sub_tr["risk_severity"] == s).sum()) for s in [1, 2, 3, 4, 5]]

        pos_sub_dv = df_dv_dom[df_dv_dom["label"] == 1]
        sev_counts_dv = [int((pos_sub_dv["risk_severity"] == s).sum()) for s in [1, 2, 3, 4, 5]]

        print(f"  • Train: {len(df_tr_dom)} samples | Positives={n_pos_tr} | Negatives={n_neg_tr} | Sev Counts: {sev_counts_tr}")
        print(f"  • Dev  : {len(df_dv_dom)} samples | Positives={n_pos_dv} | Negatives={n_neg_dv} | Sev Counts: {sev_counts_dv}")
        print(f"  • Saved -> {tr_out.name} & {dv_out.name}")

        summary_stats.append({
            "folder": dom_folder,
            "domain_name": dom_name,
            "train_total": len(df_tr_dom),
            "train_positives": int(n_pos_tr),
            "train_sev_1": sev_counts_tr[0],
            "train_sev_2": sev_counts_tr[1],
            "train_sev_3": sev_counts_tr[2],
            "train_sev_4": sev_counts_tr[3],
            "train_sev_5": sev_counts_tr[4],
            "dev_total": len(df_dv_dom),
            "dev_positives": int(n_pos_dv),
            "dev_sev_1": sev_counts_dv[0],
            "dev_sev_2": sev_counts_dv[1],
            "dev_sev_3": sev_counts_dv[2],
            "dev_sev_4": sev_counts_dv[3],
            "dev_sev_5": sev_counts_dv[4]
        })

    # 4. Save Global Summary Table
    df_summary = pd.DataFrame(summary_stats)
    summary_csv = DATA_DIR / "dataset_domains_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved global domain dataset summary matrix to: {summary_csv}")

    # Display clean ASCII matrix
    print("\n" + "=" * 105)
    print(f"{'Folder':<10} | {'Domain Name':<35} | {'Train (Total/Pos)':<18} | {'Dev (Total/Pos)':<16} | {'Dev Sev 1-5':<15}")
    print("-" * 105)
    for r in summary_stats:
        tr_info = f"{r['train_total']} / {r['train_positives']}"
        dv_info = f"{r['dev_total']} / {r['dev_positives']}"
        sev_info = f"[{r['dev_sev_1']}, {r['dev_sev_2']}, {r['dev_sev_3']}, {r['dev_sev_4']}, {r['dev_sev_5']}]"
        print(f"{r['folder']:<10} | {r['domain_name']:<35} | {tr_info:<18} | {dv_info:<16} | {sev_info:<15}")
    print("=" * 105)


if __name__ == "__main__":
    generate_all_domain_datasets()
