"""
matchers/__init__.py
====================
Factory per la creazione dei matcher.

La mappa method → matcher rispecchia quella degli estrattori:
  sift     → FLANNMatcher     (descrittori float)
  orb      → BruteForceMatcher (descrittori binari Hamming)
  akaze    → BruteForceMatcher (descrittori binari Hamming)
  minutiae → MinutiaeMatcher  (matching per coordinate + RANSAC affine)

Per aggiungere un nuovo matcher:
  1. Crea matchers/<nome>_matcher.py
  2. Importalo qui e aggiungilo a REGISTRY
"""

from .base               import BaseMatcher, MatchScore
from .flann_matcher      import FLANNMatcher
from .bruteforce_matcher import BruteForceMatcher
from .minutiae_matcher   import MinutiaeMatcher

REGISTRY = {
    "sift":     FLANNMatcher,
    "orb":      BruteForceMatcher,
    "akaze":    BruteForceMatcher,
    "minutiae":      MinutiaeMatcher,
    "enhanced_sift": FLANNMatcher,   # usa FLANN come il SIFT normale
}


def get_matcher(method: str, config) -> BaseMatcher:
    """Crea e restituisce il matcher per il metodo indicato."""
    cls = REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Matcher '{method}' non trovato. "
            f"Disponibili: {list(REGISTRY)}"
        )
    return cls(config)


__all__ = [
    "BaseMatcher", "MatchScore",
    "FLANNMatcher", "BruteForceMatcher", "MinutiaeMatcher",
    "get_matcher", "REGISTRY",
]
