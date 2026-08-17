from __future__ import annotations

import numpy as np


class MatchScorer:
    """Score assoluto e conservativo, indipendente dal database corrente."""

    def __init__(self, config):
        self.cfg = config

    def score(self, verification, match_score=None, n_feat_query=0,
              n_feat_ref=0, method="sift") -> float:
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
