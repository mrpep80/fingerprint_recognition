"""
extractors/base.py
==================
Definisce il contratto comune per tutti gli estrattori di feature.

Aggiungere un nuovo metodo di estrazione significa:
  1. Creare un file <nome>_extractor.py in questa cartella
  2. Estendere BaseExtractor implementando name() ed extract()
  3. Registrarlo in extractors/__init__.py
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeatureSet:
    """
    Contenitore per le feature estratte da un'immagine.

    Campi:
        keypoints   – lista di cv2.KeyPoint (usati anche per le minuzie)
        descriptors – matrice dei descrittori (None per le minuzie)
        metadata    – dati extra (es. lista 'minutiae' per MinutiaeExtractor)
        source      – quale metodo ha prodotto questo FeatureSet
    """
    keypoints:   Optional[List[Any]]  = None
    descriptors: Optional[np.ndarray] = None
    metadata:    Dict[str, Any]       = field(default_factory=dict)
    source:      str                  = ""

    @property
    def n_features(self) -> int:
        return len(self.keypoints) if self.keypoints else 0

    def is_valid(self) -> bool:
        """True se contiene almeno un keypoint."""
        return self.keypoints is not None and len(self.keypoints) > 0


class BaseExtractor(ABC):
    """Classe base astratta per tutti gli estrattori di feature."""

    def __init__(self, config):
        self.cfg = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificatore stringa del metodo (es. 'sift', 'minutiae')."""

    @abstractmethod
    def extract(self, img: np.ndarray) -> FeatureSet:
        """
        Estrae feature dall'immagine pre-elaborata.

        Args:
            img: immagine in scala di grigi già pre-elaborata

        Returns:
            FeatureSet con i risultati (valido o vuoto in caso di errore)
        """

    def is_available(self) -> bool:
        """
        Verifica se le dipendenze necessarie sono disponibili.
        Sovrascrivere nelle sottoclassi che richiedono tool esterni.
        """
        return True

    @property
    def needs_raw_input(self) -> bool:
        """
        Se True, l'engine passa l'immagine GREZZA (non preprocessata) all'extractor.
        Usato dagli estrattori che gestiscono internamente la propria pipeline
        (es. EnhancedSIFTExtractor che fa DFT→HE→Hong-Jain internamente).
        Default: False (riceve immagine già CLAHE+Gabor).
        """
        return False
