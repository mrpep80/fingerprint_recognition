"""
extractors/__init__.py
=======================
Factory per la creazione degli estrattori.

Per aggiungere un nuovo estrattore:
  1. Crea extractors/<nome>_extractor.py con la classe che estende BaseExtractor
  2. Importala qui e aggiungila a REGISTRY
"""

from .base import BaseExtractor, FeatureSet
from .sift_extractor     import SIFTExtractor
from .orb_extractor      import ORBExtractor
from .akaze_extractor    import AKAZEExtractor
from .minutiae_extractor    import MinutiaeExtractor
from .enhanced_sift_extractor import EnhancedSIFTExtractor

REGISTRY = {
    "sift":     SIFTExtractor,
    "orb":      ORBExtractor,
    "akaze":    AKAZEExtractor,
    "minutiae":       MinutiaeExtractor,
    "enhanced_sift":  EnhancedSIFTExtractor,
}


def get_extractor(method: str, config) -> BaseExtractor:
    """Crea e restituisce l'estrattore per il metodo indicato."""
    cls = REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Estrattore '{method}' non trovato. "
            f"Disponibili: {list(REGISTRY)}"
        )
    return cls(config)


__all__ = [
    "BaseExtractor", "FeatureSet",
    "SIFTExtractor", "ORBExtractor", "AKAZEExtractor",
    "MinutiaeExtractor", "EnhancedSIFTExtractor",
    "get_extractor", "REGISTRY",
]
