"""
matchers/flann_matcher.py
==========================
FLANN (Fast Library for Approximate Nearest Neighbors) per descrittori float.
Ottimale per SIFT (descrittori float32 a 128 dimensioni).
Usa FLANN_INDEX_KDTREE + test del rapporto di Lowe.
"""

import cv2
import numpy as np
from .base import BaseMatcher, MatchScore
from ..extractors.base import FeatureSet


class FLANNMatcher(BaseMatcher):
    """Matcher FLANN – ideale per descrittori float (SIFT)."""

    @property
    def name(self) -> str:
        return "flann"

    def match(self, query: FeatureSet, reference: FeatureSet) -> MatchScore:
        n_q = query.n_features

        if not query.is_valid() or not reference.is_valid():
            return MatchScore([], n_q)
        if query.descriptors is None or reference.descriptors is None:
            return MatchScore([], n_q)

        flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": self.cfg.flann_trees},
            {"checks": self.cfg.flann_checks},
        )
        try:
            raw = flann.knnMatch(
                query.descriptors.astype(np.float32),
                reference.descriptors.astype(np.float32),
                k=2,
            )
        except cv2.error:
            return MatchScore([], n_q)

        good = [
            m for pair in raw
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.cfg.sift_ratio_threshold * n.distance
        ]
        return MatchScore(good, n_q)
