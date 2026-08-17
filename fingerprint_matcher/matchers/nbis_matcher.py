from __future__ import annotations

import os
import subprocess

from .base import BaseMatcher, MatchScore
from ..engines.nbis import get_nbis_runtime


class NBISMatcher(BaseMatcher):
    """BOZORTH3 matcher for MINDTCT .xyt templates."""

    name = "nbis"

    def __init__(self, config):
        super().__init__(config)
        runtime = get_nbis_runtime()
        self.bozorth3 = os.environ.get("NBIS_BOZORTH3") or (
            str(runtime.bozorth3) if runtime.available else None
        )

    def match(self, query, reference) -> MatchScore:
        if not self.bozorth3:
            return MatchScore([], query.n_features, False, reference.n_features, 0.0)
        q = query.metadata.get("xyt_path")
        r = reference.metadata.get("xyt_path")
        if not q or not r or not os.path.isfile(q) or not os.path.isfile(r):
            return MatchScore([], query.n_features, False, reference.n_features, 0.0)
        try:
            p = subprocess.run([self.bozorth3, "-m1", q, r],
                               capture_output=True, text=True, timeout=15)
            token = (p.stdout or "").strip().split()
            score = float(token[-1]) if token else 0.0
            if p.returncode != 0:
                score = 0.0
        except (OSError, ValueError, subprocess.SubprocessError):
            score = 0.0
        return MatchScore(raw_matches=[score] if score > 0 else [],
                          n_features_query=query.n_features,
                          is_verified=score > 0,
                          n_features_ref=reference.n_features,
                          native_score=score)
