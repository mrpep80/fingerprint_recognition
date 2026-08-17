"""Factory per gli estrattori di feature."""
from .base import BaseExtractor, FeatureSet
from .sift_extractor import SIFTExtractor
from .orb_extractor import ORBExtractor
from .akaze_extractor import AKAZEExtractor
from .minutiae_extractor import MinutiaeExtractor
from .enhanced_sift_extractor import EnhancedSIFTExtractor
from .nbis_extractor import NBISExtractor

REGISTRY = {
    "sift": SIFTExtractor,
    "orb": ORBExtractor,
    "akaze": AKAZEExtractor,
    "minutiae": MinutiaeExtractor,
    "enhanced_sift": EnhancedSIFTExtractor,
    "nbis": NBISExtractor,
}


def get_extractor(method: str, config) -> BaseExtractor:
    cls = REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"Estrattore '{method}' non trovato. Disponibili: {list(REGISTRY)}")
    return cls(config)


__all__ = [
    "BaseExtractor", "FeatureSet", "SIFTExtractor", "ORBExtractor",
    "AKAZEExtractor", "MinutiaeExtractor", "EnhancedSIFTExtractor",
    "NBISExtractor", "get_extractor", "REGISTRY",
]
