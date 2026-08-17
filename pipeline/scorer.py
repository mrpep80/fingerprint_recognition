"""
pipeline/scorer.py – v2.2
===========================
Formula: geometric mean di inlier_strength e coverage.

  inlier_strength = min(1, n_inliers / target_inliers)
      → penalizza match con pochi inlier assoluti

  coverage = min(cov_query, cov_ref)
      → penalizza match in cui il frammento/riferimento è poco coperto

  score = sqrt(strength × coverage) × 100

Separazione attesa:
  Match genuino (>40 inliers)  → 60-100%
  Falso positivo strutturale   → 15-30%
  Falso positivo casuale       →  0-10%
"""

import numpy as np
from ..config import Config


class MatchScorer:

    def __init__(self, config: Config):
        self.cfg = config

    def score(
        self,
        n_inliers:    int,
        n_good:       int,
        n_feat_query: int,
        n_feat_ref:   int = 0,
    ) -> float:
        if n_inliers < self.cfg.min_inliers:
            return 0.0

        # Forza assoluta: quanti inlier abbiamo rispetto al target?
        target   = self.cfg.score_bonus_ref     # riutilizziamo questo campo (default 80)
        strength = min(1.0, n_inliers / target)

        # Copertura bidirezionale
        cov_q = n_inliers / max(n_feat_query, 1)
        if n_feat_ref > 0:
            cov_r    = n_inliers / max(n_feat_ref, 1)
            coverage = min(cov_q, cov_r)
        else:
            coverage = cov_q

        # Media geometrica: penalizza se uno dei due è basso
        combined = (strength * coverage) ** 0.5
        return float(np.clip(combined * 100, 0.0, 100.0))
