"""
Stage 03: Enhanced Domain Risk Label Definitions and Prototype Key Generation.

Addresses the feature overlap and low-separability bottleneck identified in Stage 02
by expanding domain definitions into multi-perspective, realistic operational risk prototypes:
1. Emergency & Physical Hazard Risk (critical infrastructure, physical danger, disaster dispatch)
2. Cyber Security & Privilege Escalation Risk (root access, credential leakage, authentication bypass)
3. Confidential Data & Privacy Leakage Risk (PII, electronic health records, financial transactions)
4. Operational & Policy Safety Deviation (unauthorized workflow actions, bypassing confirmation gates)
5. Routine Safe Operational Query (benign retrieval, harmless queries, standard instructions)
"""

import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
if "transformers.dependency_versions_check" not in sys.modules:
    fake_dep = types.ModuleType("transformers.dependency_versions_check")
    fake_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = fake_dep

from sentence_transformers import SentenceTransformer

STAGE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = STAGE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "emergency_risk": {
        "name": "Emergency & Physical Hazard Risk",
        "description": (
            "Critical emergency situation, acute physical safety hazard, immediate threat to human life. "
            "Dispatching emergency services, fire rescue, ambulances, or industrial plant shutdown failure. "
            "Severe physical disaster, chemical spill, reactor overheating, structural building collapse."
        )
    },
    "authorization_access_risk": {
        "name": "Cyber Security & Privilege Escalation Risk",
        "description": (
            "Unauthorized privilege escalation, root administrator credential compromise, and security policy breach. "
            "Executing unauthorized sudo bash shell commands, modifying shadow passwords, opening backdoors. "
            "Arbitrary remote code execution, token hijacking, and security perimeter bypass."
        )
    },
    "data_privacy_risk": {
        "name": "Confidential Data & Privacy Leakage Risk",
        "description": (
            "Exfiltration and leakage of sensitive personal identifiable information (PII), confidential records, "
            "or proprietary organizational data. Unauthorized access to electronic health patient medical records, "
            "credit card numbers, private banking transfers, or internal secret keys sent to external mailboxes."
        )
    },
    "operational_policy_deviation": {
        "name": "Operational & Policy Safety Deviation",
        "description": (
            "Autonomous agent policy violation, executing high-impact workflows without required human confirmation. "
            "Bypassing organization guidelines, unapproved bulk deletions, unauthorized automated calendar changes, "
            "or initiating financial fund transfers without human-in-the-loop authorization."
        )
    },
    "routine_safe_query": {
        "name": "Routine Safe Operational Query",
        "description": (
            "Standard harmless user query, routine information lookup, benign developer programming inquiry. "
            "Querying public open-source GitHub repositories, formatting markdown documents, checking train schedules, "
            "calculating mathematical formulas, or reading documentation with zero safety risk."
        )
    }
}


class DomainLabelRegistry:
    """Registry managing domain prototypes and generation of static key matrix K."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        keys_cache_path: Optional[Path] = None
    ):
        self.model_name = model_name
        self.keys_cache_path = keys_cache_path or (ARTIFACTS_DIR / "domain_keys.npy")
        self.definitions = DOMAIN_DEFINITIONS
        self.label_keys = list(self.definitions.keys())
        self.label_names = [self.definitions[k]["name"] for k in self.label_keys]
        self.descriptions = [self.definitions[k]["description"] for k in self.label_keys]
        self.key_matrix: Optional[np.ndarray] = None

    def generate_key_vectors(self, force_regenerate: bool = False) -> np.ndarray:
        if not force_regenerate and self.keys_cache_path.exists():
            self.key_matrix = np.load(self.keys_cache_path)
            return self.key_matrix

        embedder = SentenceTransformer(self.model_name)
        keys = embedder.encode(self.descriptions, convert_to_numpy=True, normalize_embeddings=True)
        self.key_matrix = keys.astype(np.float32)
        np.save(self.keys_cache_path, self.key_matrix)
        return self.key_matrix

    def get_key_matrix(self) -> np.ndarray:
        if self.key_matrix is None:
            return self.generate_key_vectors()
        return self.key_matrix


def main():
    registry = DomainLabelRegistry()
    K = registry.generate_key_vectors(force_regenerate=True)
    print(f"Generated {len(K)} enhanced domain key vectors of shape: {K.shape}")
    for k, name in zip(registry.label_keys, registry.label_names):
        print(f"  • {name} -> key '{k}'")


if __name__ == "__main__":
    main()

