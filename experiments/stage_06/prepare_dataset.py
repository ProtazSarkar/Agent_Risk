"""
Stage 06: Multi-Domain Probability Ensemble & Severity Dataset Generator.

Takes the 11 trained domain prototype models from Stage 05 (domain_01 to domain_11)
and applies them across all interaction embeddings to extract:
  1. The 11 calibrated domain risk probabilities:
     - prob_no_material_harm (domain_01)
     - prob_privacy (domain_02)
     - prob_security (domain_03)
     - prob_financial (domain_04)
     - prob_operational (domain_05)
     - prob_physical (domain_06)
     - prob_legal (domain_07)
     - prob_reputational (domain_08)
     - prob_discrimination (domain_09)
     - prob_environmental (domain_10)
     - prob_information_quality (domain_11)
  2. Multi-domain statistical aggregates:
     - max_harm_prob, sum_harm_prob, mean_harm_prob, std_harm_prob
     - harm_to_safe_ratio, domain_entropy
     - num_domains_ge_05, num_domains_ge_02
     - top1_domain_id, top1_domain_prob, top2_domain_prob, top_domain_margin
     - high-consequence severity flags (physical_critical, security_critical)
  3. Latent semantic features:
     - 8-dimensional PCA projections from 384-d embeddings
  4. Text statistical metadata:
     - summary_word_count, summary_char_count, refusal_word_count, critical_threat_count

Exports:
  experiments/stage_06/data/train_severity_features.csv
  experiments/stage_06/data/dev_severity_features.csv
  experiments/stage_06/output/feature_summary.md
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
from sklearn.decomposition import PCA

STAGE_06_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE_06_DIR.parent.parent
STAGE_05_DIR = ROOT_DIR / "experiments" / "stage_05"

# Ensure stage_05 is on sys.path to load SemanticPrototypeClassifier
sys.path.insert(0, str(STAGE_05_DIR / "domain_02"))
from model import SemanticPrototypeClassifier

DATA_DIR = STAGE_06_DIR / "data"
OUTPUT_DIR = STAGE_06_DIR / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_NAMES = [
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

REFUSAL_KEYWORDS = ["cannot", "safeguard", "refuse", "refusal", "violat", "protect", "denied", "prohibit"]
CRITICAL_KEYWORDS = ["lethal", "nuclear", "overdose", "destroy", "kill", "rm -rf", "wipe", "poison", "sanction", "50,000,000", "fatal"]


def load_all_11_models() -> Dict[str, SemanticPrototypeClassifier]:
    """Loads all 11 trained domain classifiers from stage_05/models/."""
    models = {}
    models_dir = STAGE_05_DIR / "models"
    print("Loading 11 trained domain classifiers from Stage 05...")
    for dom_id, col_name in DOMAIN_NAMES:
        model_path = models_dir / f"{dom_id}_classifier.pt"
        if not model_path.exists() and dom_id == "domain_01":
            model_path = models_dir / "semantic_prototype_classifier.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        checkpoint = torch.load(model_path, weights_only=False)
        m = SemanticPrototypeClassifier(
            initial_prototype_embeddings=checkpoint["prototype_keys"],
            hidden_dim=checkpoint.get("hidden_dim", 64)
        )
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        models[dom_id] = m
    print("✓ All 11 domain models loaded successfully into memory.")
    return models


def load_input_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads base summarized embedded datasets plus balanced domain augmentations."""
    data_dir = STAGE_05_DIR / "data"
    tr_path = data_dir / "train_summarized_embedded.csv"
    dv_path = data_dir / "dev_summarized_embedded.csv"

    if not tr_path.exists() or not dv_path.exists():
        raise FileNotFoundError(f"Base summarized datasets not found in {data_dir}")

    df_tr_base = pd.read_csv(tr_path)
    df_dv_base = pd.read_csv(dv_path)

    # Gather synthetic severity augmentations generated in Stage 05 (to ensure rich supervision for Sev 2 and 5)
    # Check all train_domain_XX.csv for synthetic examples
    synth_train_records = []
    synth_dev_records = []
    for dom_id, _ in DOMAIN_NAMES:
        tr_dom_file = data_dir / f"train_{dom_id}.csv"
        dv_dom_file = data_dir / f"dev_{dom_id}.csv"
        if tr_dom_file.exists():
            df_t = pd.read_csv(tr_dom_file)
            synth_t = df_t[df_t["example_id"].str.startswith("SYN-")].copy()
            if not synth_t.empty:
                synth_train_records.append(synth_t)
        if dv_dom_file.exists():
            df_d = pd.read_csv(dv_dom_file)
            synth_d = df_d[df_d["example_id"].str.startswith("SYN-")].copy()
            if not synth_d.empty:
                synth_dev_records.append(synth_d)

    # Deduplicate synthetic records by example_id
    emb_cols = [c for c in df_tr_base.columns if c.startswith("emb_")]
    meta_cols = ["example_id", "summary_text", "risk_severity", "risk_domains"] + emb_cols

    if synth_train_records:
        df_synth_tr = pd.concat(synth_train_records, ignore_index=True).drop_duplicates(subset=["example_id"])[meta_cols]
        df_tr_all = pd.concat([df_tr_base[meta_cols], df_synth_tr], ignore_index=True)
    else:
        df_tr_all = df_tr_base[meta_cols].copy()

    if synth_dev_records:
        df_synth_dv = pd.concat(synth_dev_records, ignore_index=True).drop_duplicates(subset=["example_id"])[meta_cols]
        df_dv_all = pd.concat([df_dv_base[meta_cols], df_synth_dv], ignore_index=True)
    else:
        df_dv_all = df_dv_base[meta_cols].copy()

    print(f"Loaded Datasets:")
    print(f"  • Base Train: {len(df_tr_base)} rows | Augmented Train: {len(df_tr_all)} rows")
    print(f"  • Base Dev  : {len(df_dv_base)} rows | Augmented Dev  : {len(df_dv_all)} rows")
    return df_tr_all, df_dv_all


def compute_domain_probabilities(
    models: Dict[str, SemanticPrototypeClassifier],
    X_tensor: torch.Tensor
) -> Dict[str, np.ndarray]:
    """Runs all 11 models over input embeddings to obtain calibrated risk probabilities."""
    probs = {}
    with torch.no_grad():
        for dom_id, short_name in DOMAIN_NAMES:
            model = models[dom_id]
            p = model.predict_proba(X_tensor)
            probs[f"prob_{short_name}"] = p
    return probs


def extract_features(
    df: pd.DataFrame,
    models: Dict[str, SemanticPrototypeClassifier],
    pca_model: PCA = None,
    fit_pca: bool = False
) -> Tuple[pd.DataFrame, PCA]:
    """Extracts the full tabular feature matrix for severity prediction."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X_emb = torch.tensor(df[emb_cols].values, dtype=torch.float32)

    # 1. Domain Probabilities
    domain_probs_dict = compute_domain_probabilities(models, X_emb)
    df_features = pd.DataFrame(domain_probs_dict, index=df.index)

    # 2. Harm Domains Aggregate Statistics (Domains 2 through 11)
    harm_cols = [f"prob_{name}" for dom_id, name in DOMAIN_NAMES if dom_id != "domain_01"]
    harm_mat = df_features[harm_cols].values  # shape (N, 10)

    p_no_harm = df_features["prob_no_material_harm"].values

    df_features["max_harm_prob"] = np.max(harm_mat, axis=1)
    df_features["sum_harm_prob"] = np.sum(harm_mat, axis=1)
    df_features["mean_harm_prob"] = np.mean(harm_mat, axis=1)
    df_features["std_harm_prob"] = np.std(harm_mat, axis=1)
    df_features["harm_to_safe_ratio"] = df_features["sum_harm_prob"] / (p_no_harm + 1e-4)

    # Shannon Entropy across all 11 domains (measures dispersion of risk)
    all_prob_cols = [f"prob_{name}" for _, name in DOMAIN_NAMES]
    all_mat = df_features[all_prob_cols].values
    all_norm = all_mat / (np.sum(all_mat, axis=1, keepdims=True) + 1e-8)
    entropy = -np.sum(all_norm * np.log(all_norm + 1e-8), axis=1)
    df_features["domain_entropy"] = entropy

    # Count of domains triggered
    df_features["num_domains_ge_05"] = np.sum(harm_mat >= 0.50, axis=1)
    df_features["num_domains_ge_02"] = np.sum(harm_mat >= 0.20, axis=1)

    # Top-1 and Top-2 harm domain features
    sorted_harm = np.sort(harm_mat, axis=1)[:, ::-1]
    top1_probs = sorted_harm[:, 0]
    top2_probs = sorted_harm[:, 1]
    df_features["top1_harm_prob"] = top1_probs
    df_features["top2_harm_prob"] = top2_probs
    df_features["top_harm_margin"] = top1_probs - top2_probs

    # High-consequence severity domain flags (Physical = prob_physical, Security = prob_security)
    df_features["is_physical_high"] = (df_features["prob_physical"] >= 0.40).astype(float)
    df_features["is_security_high"] = (df_features["prob_security"] >= 0.40).astype(float)
    df_features["is_financial_high"] = (df_features["prob_financial"] >= 0.40).astype(float)

    # 3. PCA Latent Features from 384-d Embeddings
    emb_values = df[emb_cols].values
    if fit_pca:
        pca_model = PCA(n_components=8, random_state=42)
        pca_feats = pca_model.fit_transform(emb_values)
    else:
        pca_feats = pca_model.transform(emb_values)

    for i in range(8):
        df_features[f"pca_{i}"] = pca_feats[:, i]

    # 4. Text Statistical Metadata
    summaries = df["summary_text"].fillna("").astype(str).tolist()
    char_lens = [len(s) for s in summaries]
    word_lens = [len(s.split()) for s in summaries]

    refusal_counts = []
    critical_counts = []
    for s in summaries:
        s_lower = s.lower()
        refusal_counts.append(sum(1 for kw in REFUSAL_KEYWORDS if kw in s_lower))
        critical_counts.append(sum(1 for kw in CRITICAL_KEYWORDS if kw in s_lower))

    df_features["text_word_count"] = word_lens
    df_features["text_char_count"] = char_lens
    df_features["refusal_keyword_count"] = refusal_counts
    df_features["critical_keyword_count"] = critical_counts

    # 5. Attach Meta & Target Columns
    df_features["example_id"] = df["example_id"].values
    df_features["summary_text"] = df["summary_text"].values
    df_features["risk_severity"] = df["risk_severity"].astype(int).values
    df_features["risk_domains"] = df["risk_domains"].fillna("No Material Harm").values

    # Reorder columns: meta & target first, then engineered features
    front_cols = ["example_id", "risk_severity", "risk_domains", "summary_text"]
    other_cols = [c for c in df_features.columns if c not in front_cols]
    df_features = df_features[front_cols + other_cols]

    return df_features, pca_model


def prepare_stage_06_severity_datasets():
    print("=" * 100)
    print("STAGE 06: MULTI-DOMAIN PROBABILITY ENSEMBLE & SEVERITY DATASET PREPARATION")
    print("=" * 100)

    start_time = time.time()

    # 1. Load Models & Base Data
    models = load_all_11_models()
    df_tr_raw, df_dv_raw = load_input_datasets()

    # 2. Extract Tabular Features
    print("\nExtracting multi-domain probabilities and engineered features for Train set...")
    df_tr_feat, pca = extract_features(df_tr_raw, models, fit_pca=True)
    print(f"✓ Train features extracted: shape {df_tr_feat.shape}")

    print("\nExtracting multi-domain probabilities and engineered features for Dev set...")
    df_dv_feat, _ = extract_features(df_dv_raw, models, pca_model=pca, fit_pca=False)
    print(f"✓ Dev features extracted: shape {df_dv_feat.shape}")

    # 3. Export to CSV
    tr_out = DATA_DIR / "train_severity_features.csv"
    dv_out = DATA_DIR / "dev_severity_features.csv"
    df_tr_feat.to_csv(tr_out, index=False)
    df_dv_feat.to_csv(dv_out, index=False)
    print(f"\n✓ Saved Train dataset -> {tr_out}")
    print(f"✓ Saved Dev dataset   -> {dv_out}")

    # 4. Generate Severity Distribution & Feature Summary Report
    elapsed = time.time() - start_time
    summary_report_path = OUTPUT_DIR / "feature_summary.md"
    generate_feature_summary_report(df_tr_feat, df_dv_feat, elapsed, summary_report_path)
    print(f"✓ Saved Feature Summary Report -> {summary_report_path}")

    # Display clean table
    print("\n" + "=" * 80)
    print("DATASET SEVERITY BREAKDOWN (RISK LEVEL 1 TO 5)")
    print("=" * 80)
    tr_vc = df_tr_feat["risk_severity"].value_counts().sort_index()
    dv_vc = df_dv_feat["risk_severity"].value_counts().sort_index()
    print(f"{'Severity':<10} | {'Train Count':<15} | {'Dev Count':<15}")
    print("-" * 50)
    for s in [1, 2, 3, 4, 5]:
        print(f"Severity {s:<2} | {tr_vc.get(s, 0):<15} | {dv_vc.get(s, 0):<15}")
    print("=" * 80)
    print(f"Pipeline executed in {elapsed:.2f}s")


def generate_feature_summary_report(df_tr: pd.DataFrame, df_dv: pd.DataFrame, elapsed: float, out_path: Path):
    feature_cols = [c for c in df_tr.columns if c not in ["example_id", "risk_severity", "risk_domains", "summary_text"]]

    # Pearson correlations with risk_severity
    corrs = {}
    for c in feature_cols:
        corrs[c] = float(df_tr[c].corr(df_tr["risk_severity"]))
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    corr_rows = []
    for f_name, r_val in sorted_corrs:
        corr_rows.append(f"| `{f_name}` | {r_val:+.4f} |")

    corr_table = "\n".join(corr_rows[:15])

    content = f"""# Stage 06: Multi-Domain Probability Ensemble & Feature Summary Report

## 1. Executive Summary

This report documents the **Stage 06 feature preparation pipeline** that feeds directly into **Stage 07** risk level (1–5) severity models.

- **Pipeline Purpose**: Transforms raw interactions into an information-dense feature space using all 11 calibrated domain probability outputs from Stage 05.
- **Total Features**: {len(feature_cols)} engineered tabular features.
- **Dataset Dimensions**:
  - Train: `{len(df_tr)}` rows $\\times$ `{df_tr.shape[1]}` columns
  - Dev  : `{len(df_dv)}` rows $\\times$ `{df_dv.shape[1]}` columns
- **Execution Time**: `{elapsed:.2f}s`

---

## 2. Severity Target Distribution (Risk Levels 1 to 5)

| Risk Severity | Train Samples | Dev Samples | Description |
| :---: | :---: | :---: | :--- |
| **Severity 1** | `{int((df_tr['risk_severity']==1).sum())}` | `{int((df_dv['risk_severity']==1).sum())}` | Negligible / Public Info / Safe Refusal |
| **Severity 2** | `{int((df_tr['risk_severity']==2).sum())}` | `{int((df_dv['risk_severity']==2).sum())}` | Minor / Local Config / Non-Destructive |
| **Severity 3** | `{int((df_tr['risk_severity']==3).sum())}` | `{int((df_dv['risk_severity']==3).sum())}` | Moderate / Reversible Service Disruption |
| **Severity 4** | `{int((df_tr['risk_severity']==4).sum())}` | `{int((df_dv['risk_severity']==4).sum())}` | Serious / Data Exfiltration / Financial Loss |
| **Severity 5** | `{int((df_tr['risk_severity']==5).sum())}` | `{int((df_dv['risk_severity']==5).sum())}` | Critical / Catastrophic / Bodily Danger |
| **TOTAL** | **`{len(df_tr)}`** | **`{len(df_dv)}`** | Held-out validation split |

---

## 3. Top Predictive Features (Correlations with `risk_severity`)

| Feature Name | Pearson Correlation ($r$) |
| :--- | :---: |
{corr_table}

---

## 4. Feature Groups Breakdown
1. **11 Domain Probabilities**: Calibrated outputs from `domain_01` to `domain_11`.
2. **Harm Aggregate Statistics**: `max_harm_prob`, `sum_harm_prob`, `harm_to_safe_ratio`, `domain_entropy`.
3. **Multi-Domain Counters**: `num_domains_ge_05`, `num_domains_ge_02`.
4. **Severity Flags**: `is_physical_high`, `is_security_high`, `is_financial_high`.
5. **PCA Latents**: 8 principal components capturing dense semantic context.
6. **Text & Refusal Keywords**: `refusal_keyword_count`, `critical_keyword_count`, `text_word_count`.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    prepare_stage_06_severity_datasets()

