from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..matchers.base import MatchScore
from ..extractors.base import FeatureSet


@dataclass
class VerificationResult:
    n_inliers: int
    homography: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    inlier_matches: List = None
    geometry_quality: float = 0.0
    coverage: float = 0.0

    def __post_init__(self):
        if self.inlier_matches is None:
            self.inlier_matches = []


class RANSACVerifier:
    """RANSAC con controlli contro match degenerati o troppo concentrati."""

    def __init__(self, config):
        self.cfg = config

    def verify(self, query_feat: FeatureSet, ref_feat: FeatureSet,
               match_score: MatchScore) -> VerificationResult:
        good = match_score.raw_matches
        if len(good) < self.cfg.min_inliers:
            return VerificationResult(0)

        if match_score.is_verified:
            pairs = good
            if len(pairs) < self.cfg.min_inliers:
                return VerificationResult(0)
            qmins = query_feat.metadata.get("minutiae", [])
            rmins = ref_feat.metadata.get("minutiae", [])
            if not qmins or not rmins:
                return VerificationResult(0)
            q = np.float32([[qmins[a]["x"], qmins[a]["y"]] for a, _ in pairs])
            r = np.float32([[rmins[b]["x"], rmins[b]["y"]] for _, b in pairs])
            return self._quality(len(pairs), q, r, None, pairs)

        kpq, kpr = query_feat.keypoints, ref_feat.keypoints
        src = np.float32([kpq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kpr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            H, mask = cv2.findHomography(
                src, dst, cv2.RANSAC,
                self.cfg.ransac_reprojection_error,
                maxIters=3000, confidence=0.995,
            )
        except cv2.error:
            return VerificationResult(0)
        if mask is None:
            return VerificationResult(0)
        idx = np.flatnonzero(mask.ravel())
        q = src.reshape(-1, 2)[idx]
        r = dst.reshape(-1, 2)[idx]
        pairs = [good[i] for i in idx]
        return self._quality(len(idx), q, r, H, pairs, mask)

    def _quality(self, n, q, r, H, pairs, mask=None):
        if n < self.cfg.min_inliers:
            return VerificationResult(0)

        def spread(pts):
            if len(pts) < 4:
                return 0.0
            area = max(float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])), 1.0)
            return float(np.clip(area / self.cfg.geometry_area_reference, 0, 1))

        coverage = min(spread(q), spread(r))
        count_term = min(1.0, n / self.cfg.geometry_inlier_target)
        quality = (count_term ** 0.55) * (max(0.25, coverage) ** 0.45)

        if H is not None:
            A = H[:2, :2]
            det = abs(float(np.linalg.det(A)))
            scale = np.sqrt(max(det, 1e-9))
            if not (self.cfg.homography_min_scale <= scale <= self.cfg.homography_max_scale):
                quality *= 0.25
            perspective = max(abs(float(H[2, 0])), abs(float(H[2, 1])))
            if perspective > self.cfg.homography_max_perspective:
                quality *= 0.35

        return VerificationResult(
            n_inliers=n,
            homography=H,
            inlier_mask=mask,
            inlier_matches=pairs,
            geometry_quality=float(np.clip(quality, 0, 1)),
            coverage=float(coverage),
        )
