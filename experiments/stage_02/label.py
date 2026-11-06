"""
Stage 02: Domain Risk Labels and Static Key Vector Generation.

Defines prototype risk domains, generates semantic key embeddings (all-MiniLM-L6-v2),
and normalizes them into a static Key Matrix K of shape (num_labels, 384).
"""

import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np

# Ensure offline cache and bypass dependency version check on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer

STAGE_DIR = Path(__file__).resolve().parent

# Explicit Domain Risk Prototypes
DOMAIN_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "emergency_risk": {
        "id": 1,
        "name": "Emergency Risk",
        "description": "Critical system failure, physical danger, severe hazard, immediate emergency situation.",
        "severity_level_affinity": 5
    },
    "authorization_access_risk": {
        "id": 2,
        "name": "Authorization & Access Risk",
        "description": "Unauthorized privilege escalation, security breach, credential compromise, policy violation.",
        "severity_level_affinity": 4
    },
    "data_privacy_risk": {
        "id": 3,
        "name": "Data Privacy Risk",
        "description": "Sensitive personal data leak, confidential data exposure, compliance breach.",
        "severity_level_affinity": 3
    },
    "operational_policy_deviation": {
        "id": 4,
        "name": "Operational Policy Deviation",
        "description": "Minor workflow error, standard warning, non-critical guideline deviation.",
        "severity_level_affinity": 2
    },
    "routine_safe_query": {
        "id": 5,
        "name": "Routine Safe Query",
        "description": "Standard user prompt, normal operational request, zero safety risk.",
        "severity_level_affinity": 1
    }
}


class DomainLabelRegistry:
    """
    Manages semantic domain risk definitions and encodes them into static
    normalized key vectors K of shape (num_labels, 384).
    """

    def __init__(
        self,
        definitions: Dict[str, Dict[str, Any]] = DOMAIN_DEFINITIONS,
        model_name: str = "all-MiniLM-L6-v2",
        artifact_dir: Path = STAGE_DIR / "artifacts"
    ):
        self.definitions = definitions
        self.model_name = model_name
        self.artifact_dir = Path(artifact_dir)
        self.keys_path = self.artifact_dir / "domain_keys.npy"
        self.metadata_path = self.artifact_dir / "domain_metadata.json"

        self.label_keys = list(self.definitions.keys())
        self.label_names = [self.definitions[k]["name"] for k in self.label_keys]
        self.descriptions = [self.definitions[k]["description"] for k in self.label_keys]
        self.key_matrix: np.ndarray = np.empty((0, 384))

    def generate_key_vectors(self, force_recompute: bool = False) -> np.ndarray:
        """Encodes domain descriptions into L2-normalized 384-dimensional key vectors."""
        if not force_recompute and self.keys_path.exists() and self.metadata_path.exists():
            print(f"✓ Loading existing domain key matrix from: {self.keys_path}")
            self.key_matrix = np.load(self.keys_path)
            return self.key_matrix

        print(f"Encoding {len(self.descriptions)} domain prototype descriptions with {self.model_name}...")
        embedder = SentenceTransformer(self.model_name)
        raw_keys = embedder.encode(self.descriptions, convert_to_numpy=True, show_progress_bar=False)

        # L2-normalize each key vector: K_norm = K / ||K||_2
        norms = np.linalg.norm(raw_keys, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.key_matrix = raw_keys / norms

        # Save artifacts
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.keys_path, self.key_matrix)

        metadata = {
            "model_name": self.model_name,
            "num_domains": len(self.label_keys),
            "embedding_dim": int(self.key_matrix.shape[1]),
            "domains": self.definitions
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Successfully generated and saved static key matrix K: {self.key_matrix.shape} to {self.keys_path}")
        return self.key_matrix

    def get_key_matrix(self) -> np.ndarray:
        """Returns the static normalized Key Matrix (5, 384)."""
        if self.key_matrix.size == 0:
            return self.generate_key_vectors()
        return self.key_matrix

    def display_domain_keys_info(self) -> None:
        """Prints diagnostic overview and cross-domain similarity matrix."""
        keys = self.get_key_matrix()
        print("\n" + "=" * 90)
        print("DOMAIN RISK PROTOTYPES & STATIC KEY VECTORS (K):")
        print("=" * 90)
        for i, key in enumerate(self.label_keys):
            d = self.definitions[key]
            norm = np.linalg.norm(keys[i])
            print(f"[{i+1}] {d['name']:<30} (L{d['severity_level_affinity']}) | Norm: {norm:.4f}")
            print(f"    Description: \"{d['description']}\"")
            print(f"    First 5 key dimensions: {np.round(keys[i][:5], 4)}")

        # Pairwise cosine similarity matrix between prototype keys (K @ K.T)
        sim_matrix = np.dot(keys, keys.T)
        print("\n" + "=" * 90)
        print("PAIRWISE COSINE SIMILARITY MATRIX BETWEEN DOMAIN KEY VECTORS (K @ K^T):")
        print("=" * 90)
        header = f"{'Domain':<30}" + "".join([f"{f'D{j+1}':<8}" for j in range(len(self.label_keys))])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(sim_matrix):
            row_str = f"{self.definitions[self.label_keys[i]]['name']:<30}" + "".join([f"{val:<8.3f}" for val in row])
            print(row_str)
        print("=" * 90)


def main():
    registry = DomainLabelRegistry()
    registry.generate_key_vectors(force_recompute=True)
    registry.display_domain_keys_info()


if __name__ == "__main__":
    main()

