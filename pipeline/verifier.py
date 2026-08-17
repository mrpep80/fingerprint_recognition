"""
pipeline/verifier.py
=====================
Verifica geometrica dei match tramite RANSAC + omografia.

Usata solo per i matcher a descrittore (SIFT, ORB, AKAZE).
MinutiaeMatcher ha già la verifica RANSAC interna → is_verified=True.

L'omografia H (3×3) modella la relazione geometrica tra le due immagini
ed è invariante a rotazione, scala e prospettiva.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..matchers.base import MatchScore
from ..extractors.base import FeatureSet


@dataclass
class VerificationResult:
    """Risultato della verifica geometrica."""
    n_inliers:   int
    homography:  Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    inlier_matches: List = None          # subset di DMatch, utile al visualizer

    def __post_init__(self):
        if self.inlier_matches is None:
            self.inlier_matches = []


class RANSACVerifier:
    """Verifica geometrica con RANSAC e stima dell'omografia."""

    def __init__(self, config):
        self.cfg = config

    def verify(
        self,
        query_feat:  FeatureSet,
        ref_feat:    FeatureSet,
        match_score: MatchScore,
    ) -> VerificationResult:
        """
        Verifica i match con RANSAC.

        Args:
            query_feat  : FeatureSet del frammento
            ref_feat    : FeatureSet del riferimento
            match_score : MatchScore prodotto dal matcher

        Returns:
            VerificationResult con numero di inlier e omografia
        """
        # Se il matcher ha già verificato internamente (MinutiaeMatcher)
        if match_score.is_verified:
            return VerificationResult(
                n_inliers=match_score.n_good,
                inlier_matches=match_score.raw_matches,
            )

        good = match_score.raw_matches
        if len(good) < self.cfg.min_inliers:
            return VerificationResult(n_inliers=0)

        kp_q = query_feat.keypoints
        kp_r = ref_feat.keypoints

        src = np.float32(
            [kp_q[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp_r[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        try:
            H, mask = cv2.findHomography(
                src, dst,
                cv2.RANSAC,
                self.cfg.ransac_reprojection_error,
            )
        except cv2.error:
            return VerificationResult(n_inliers=0)

        if mask is None:
            return VerificationResult(n_inliers=0)

        n_inliers      = int(mask.sum())
        inlier_matches = [good[i] for i, v in enumerate(mask.ravel()) if v]

        return VerificationResult(
            n_inliers=n_inliers,
            homography=H,
            inlier_mask=mask,
            inlier_matches=inlier_matches,
        )
