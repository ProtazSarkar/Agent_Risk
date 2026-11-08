"""
Stage 05: domain_06 ('Physical / Health') Prototype Key Loader & Generator.

Loads pre-computed 384-dimensional dense semantic key vectors from:
  artifacts/domain_06_prototype_keys.pt
Zero random noise is used.
"""

from pathlib import Path
from typing import Dict, Any
import torch

STAGE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = STAGE_DIR / "artifacts" / "domain_06_prototype_keys.pt"


def generate_domain_06_prototype_keys(
    force_regenerate: bool = False,
    cache_path: Path = CACHE_PATH
) -> Dict[str, Any]:
    if not force_regenerate and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    # If cache does not exist, trigger generator
    from experiments.stage_05.artifacts.generate_all_prototypes import generate_all_prototype_keys
    generate_all_prototype_keys(force_regenerate=True)
    return torch.load(cache_path, weights_only=False)


if __name__ == "__main__":
    payload = generate_domain_06_prototype_keys()
    keys = payload["all_keys"]
    print(f"Loaded prototype keys for domain_06 ('Physical / Health'): shape {keys.shape} ({payload['num_pos']} Pos, {payload['num_neg']} Neg)")
