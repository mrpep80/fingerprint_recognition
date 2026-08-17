"""
extractors/sift_extractor.py
=============================
Estrattore SIFT (Scale-Invariant Feature Transform).
Invariante a scala e rotazione. Standard de facto per il matching di impronte.
"""

import cv2
import numpy as np
from .base import BaseExtractor, FeatureSet


class SIFTExtractor(BaseExtractor):
    """Estrae feature SIFT dall'immagine preprocessata."""

    @property
    def name(self) -> str:
        return "sift"

    def extract(self, img: np.ndarray) -> FeatureSet:
        try:
            det = cv2.SIFT_create(
                nfeatures=self.cfg.max_features,
                contrastThreshold=self.cfg.sift_contrast_threshold,
                edgeThreshold=self.cfg.sift_edge_threshold,
                sigma=self.cfg.sift_sigma,
            )
            kp, desc = det.detectAndCompute(img, None)
            if kp and len(kp) > 0:
                return FeatureSet(keypoints=kp, descriptors=desc, source="sift")
        except Exception:
            pass
        return FeatureSet(source="sift")
