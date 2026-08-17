from __future__ import annotations

import numpy as np


class MatchScorer:
    """Converte l'evidenza dei matcher in score comparabili 0..100."""

    def __init__(self, config):
        self.cfg = config

    @staticmethod
    def _nbis_score(native: float) -> float:
        """Initial calibration for BOZORTH3.

        BOZORTH3 has its own score scale; it is not averaged directly with
        OpenCV scores. The mapping is intentionally conservative and can be
        recalibrated later from the benchmark dataset.
        """
        s = max(float(native), 0.0)
        if s < 10.0:
            return 0.0
        # Smooth saturation: 20≈50, 40≈78, 60≈90, 100≈97.
        return float(np.clip(100.0 * (1.0 - np.exp(-(s - 10.0) / 22.0)), 0, 100))

    def score(self, verification, match_score=None, n_feat_query=0,
              n_feat_ref=0, method="sift") -> float:
        if method == "nbis":
            native = float(getattr(match_score, "native_score", 0.0)) if match_score is not None else 0.0
            return self._nbis_score(native)

        n_inliers = int(getattr(verification, "n_inliers", verification))
        geometry = float(getattr(verification, "geometry_quality", 1.0))
        n_good = int(getattr(match_score, "n_good", n_inliers)) if match_score is not None else n_inliers

        if n_inliers < self.cfg.min_inliers:
            return 0.0

        if method == "minutiae":
            target = self.cfg.minutiae_score_target
            consensus = np.clip((n_inliers / max(n_good, 1) - 0.35) / 0.65, 0, 1)
            strength = np.clip(n_inliers / target, 0, 1)
            score = 100 * (strength ** 0.65) * (consensus ** 0.35) * geometry
            return float(np.clip(score, 0, 100))

        target = self.cfg.descriptor_score_target
        inlier_ratio = n_inliers / max(n_good, 1)
        coverage = n_inliers / max(min(n_feat_query, n_feat_ref or n_feat_query), 1)
        coverage_term = np.clip(np.sqrt(coverage * self.cfg.coverage_scale), 0, 1)
        strength = np.clip(n_inliers / target, 0, 1) ** 0.65
        consensus = np.clip((inlier_ratio - 0.08) / 0.42, 0, 1) ** 0.35
        score = 100 * strength * consensus * max(0.35, coverage_term) * geometry
        return float(np.clip(score, 0, 100))
