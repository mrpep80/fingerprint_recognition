"""
extractors/akaze_extractor.py
==============================
Estrattore AKAZE. Robusto al rumore, buono su impronte latenti parziali.
Usa descrittori binari come ORB.
"""

import cv2
import numpy as np
from .base import BaseExtractor, FeatureSet


class AKAZEExtractor(BaseExtractor):
    """Estrae feature AKAZE dall'immagine preprocessata."""

    @property
    def name(self) -> str:
        return "akaze"

    def is_available(self) -> bool:
        """Verifica a runtime che cv2.AKAZE_create esista (assente in alcune build)."""
        return hasattr(cv2, "AKAZE_create")

    def extract(self, img: np.ndarray) -> FeatureSet:
        try:
            # Ridimensiona se troppo grande: AKAZE scala male con immagini ad alta risoluzione
            img_work, scale = self._maybe_resize(img)

            det = cv2.AKAZE_create(threshold=self.cfg.akaze_threshold)
            kp, desc = det.detectAndCompute(img_work, None)

            if not kp:
                return FeatureSet(source="akaze")

            # Limita al numero massimo di feature per coerenza con gli altri metodi
            if len(kp) > self.cfg.max_features:
                kp_desc = sorted(zip(kp, desc), key=lambda x: -x[0].response)
                kp, desc = zip(*kp_desc[:self.cfg.max_features])
                kp   = list(kp)
                desc = np.array(desc)

            # Riscala le coordinate se l'immagine era stata ridimensionata
            if scale != 1.0:
                for k in kp:
                    k.pt = (k.pt[0] / scale, k.pt[1] / scale)

            return FeatureSet(keypoints=kp, descriptors=desc, source="akaze")
        except Exception:
            pass
        return FeatureSet(source="akaze")

    def _maybe_resize(self, img: np.ndarray):
        """Ridimensiona se superiore a 1000px sul lato maggiore."""
        max_dim = 1000
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return img, 1.0
        scale = max_dim / longest
        resized = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return resized, scale
