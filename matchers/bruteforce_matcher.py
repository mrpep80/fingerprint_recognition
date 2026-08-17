"""
matchers/bruteforce_matcher.py
================================
BFMatcher con norma Hamming per descrittori binari (ORB, AKAZE).
Usa il test del rapporto di Lowe adattato alla distanza di Hamming.
"""

import cv2
from .base import BaseMatcher, MatchScore
from ..extractors.base import FeatureSet


class BruteForceMatcher(BaseMatcher):
    """BFMatcher Hamming – per descrittori binari (ORB / AKAZE)."""

    @property
    def name(self) -> str:
        return "bruteforce"

    def match(self, query: FeatureSet, reference: FeatureSet) -> MatchScore:
        n_q = query.n_features

        if not query.is_valid() or not reference.is_valid():
            return MatchScore([], n_q)
        if query.descriptors is None or reference.descriptors is None:
            return MatchScore([], n_q)
        if len(query.descriptors) < 2 or len(reference.descriptors) < 2:
            return MatchScore([], n_q)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw = bf.knnMatch(query.descriptors, reference.descriptors, k=2)
        except cv2.error:
            return MatchScore([], n_q)

        good = [
            m for pair in raw
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.cfg.bin_ratio_threshold * n.distance
        ]
        return MatchScore(good, n_q)
