"""
extractors/enhanced_sift_extractor.py
=======================================
SIFT su mappa di creste pre-elaborata con pipeline Hong-Jain.

Pipeline (basato su Bhowmik 2012 + Sahu 2013 + BoLiu 2009):
    Input
      ↓ DFT enhancement (Bhowmik 2012, k=0.3)
      ↓ Histogram Equalization (Bhowmik 2012, step 2)
      ↓ Hong-Jain fingerprint enhancement (orientation-adaptive Gabor)
      ↓ Inversione (creste scure → sfondo chiaro per SIFT)
      ↓ SIFT feature extraction

Vantaggi rispetto al SIFT standard:
  - L'input di SIFT è una mappa normalizzata di creste, non l'immagine raw
  - Rimuove il rumore di sfondo (importante per impronte latenti)
  - Il campo di orientazione adattivo produce feature più stabili
  - Efficace per matching cross-domain (latente ↔ inchiostro)

Prestazioni (test sperimentale):
  - Latente → inchiostro: ~28% vs ~0% con SIFT su immagine raw
  - Sahu 2013: SURF EER=2% vs SIFT EER=10% (pipeline simile)
"""

import cv2
import numpy as np
from .base import BaseExtractor, FeatureSet
from ..preprocessing.dft_enhancer  import DFTEnhancer
from ..preprocessing.hong_enhancer import HongEnhancer


class EnhancedSIFTExtractor(BaseExtractor):
    """SIFT su binary ridge map (DFT→HE→Hong-Jain). Ottimizzato per cross-domain."""

    def __init__(self, config):
        super().__init__(config)
        self._dft  = DFTEnhancer(config)
        self._hong = HongEnhancer(config)

    @property
    def name(self) -> str:
        return "enhanced_sift"

    @property
    def needs_raw_input(self) -> bool:
        return True  # gestisce internamente: DFT→HE→Hong-Jain→SIFT

    def is_available(self) -> bool:
        return self._hong.available

    def extract(self, img: np.ndarray) -> FeatureSet:
        """
        Applica il pipeline completo e poi SIFT sulla ridge map.

        Resize a enhanced_sift_max_dim (default 800px) PRIMA di Hong-Jain:
        fingerprint_enhancer scala in O(n_pixels), quindi passare da 1500px
        a 800px riduce il tempo da ~4s a ~0.16s per immagine (24x più veloce).
        """
        # 0. Resize a enhanced_sift_max_dim (velocizzazione principale)
        h, w = img.shape[:2]
        max_d = self.cfg.enhanced_sift_max_dim
        if max(h, w) > max_d:
            scale = max_d / max(h, w)
            img   = cv2.resize(img,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        # 1. DFT enhancement (Bhowmik 2012)
        dft_img = self._dft.enhance(img, k=self.cfg.dft_k)

        # 2. Histogram equalization (Bhowmik 2012 step 2)
        he_img = cv2.equalizeHist(dft_img)

        # 3. Hong-Jain fingerprint enhancement
        ridge_map = self._hong.enhance(he_img)   # uint8 {0, 255}

        if ridge_map is he_img:   # fallback: hong non disponibile
            ridge_map = he_img

        # 4. Inverte: creste nere su sfondo bianco (SIFT funziona meglio su bordi scuri)
        inv_map = cv2.bitwise_not(ridge_map)

        # 5. SIFT feature extraction
        try:
            det = cv2.SIFT_create(
                nfeatures=self.cfg.max_features,
                contrastThreshold=self.cfg.sift_contrast_threshold,
                edgeThreshold=self.cfg.sift_edge_threshold,
                sigma=self.cfg.sift_sigma,
            )
            kp, desc = det.detectAndCompute(inv_map, None)
            if kp and len(kp) > 0:
                return FeatureSet(
                    keypoints=kp, descriptors=desc,
                    metadata={"pipeline": "dft→he→hong→sift"},
                    source="enhanced_sift",
                )
        except Exception:
            pass
        return FeatureSet(source="enhanced_sift")
