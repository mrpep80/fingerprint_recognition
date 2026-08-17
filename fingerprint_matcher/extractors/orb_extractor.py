"""
extractors/orb_extractor.py
============================
Estrattore ORB (Oriented FAST and Rotated BRIEF).
Più veloce di SIFT, usa descrittori binari (Hamming distance).
"""

import cv2
import numpy as np
from .base import BaseExtractor, FeatureSet


class ORBExtractor(BaseExtractor):
    """Estrae feature ORB dall'immagine preprocessata."""

    @property
    def name(self) -> str:
        return "orb"

    def extract(self, img: np.ndarray) -> FeatureSet:
        try:
            det = cv2.ORB_create(
                nfeatures=self.cfg.max_features,
                scaleFactor=self.cfg.orb_scale_factor,
                nlevels=self.cfg.orb_n_levels,
                edgeThreshold=self.cfg.orb_edge_threshold,
            )
            kp, desc = det.detectAndCompute(img, None)
            if kp and len(kp) > 0:
                return FeatureSet(keypoints=kp, descriptors=desc, source="orb")
        except Exception:
            pass
        return FeatureSet(source="orb")
