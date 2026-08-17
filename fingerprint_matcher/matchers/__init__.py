"""Factory per la creazione dei matcher."""
from .base import BaseMatcher, MatchScore
from .flann_matcher import FLANNMatcher
from .bruteforce_matcher import BruteForceMatcher
from .minutiae_matcher import MinutiaeMatcher
from .nbis_matcher import NBISMatcher

REGISTRY = {
    "sift": FLANNMatcher,
    "orb": BruteForceMatcher,
    "akaze": BruteForceMatcher,
    "minutiae": MinutiaeMatcher,
    "enhanced_sift": FLANNMatcher,
    "nbis": NBISMatcher,
}


def get_matcher(method: str, config) -> BaseMatcher:
    cls = REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"Matcher '{method}' non trovato. Disponibili: {list(REGISTRY)}")
    return cls(config)


__all__ = [
    "BaseMatcher", "MatchScore", "FLANNMatcher", "BruteForceMatcher",
    "MinutiaeMatcher", "NBISMatcher", "get_matcher", "REGISTRY",
]
