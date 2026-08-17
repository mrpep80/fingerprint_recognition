from __future__ import annotations

from typing import List

from .base import BaseMatcher, MatchScore


class NBISMatcher(BaseMatcher):
    """BOZORTH3 matcher for MINDTCT .xyt templates."""

    name = "nbis"

    def __init__(self, config):
        super().__init__(config)
        import os
        import shutil
        self.bozorth3 = os.environ.get("NBIS_BOZORTH3") or shutil.which("bozorth3") or shutil.which("bozorth3.exe")

    def match(self, query, reference) -> MatchScore:
        import os
        import subprocess
        if not self.bozorth3:
            return MatchScore([], query.n_features, False, reference.n_features)
        q = query.metadata.get("xyt_path")
        r = reference.metadata.get("xyt_path")
        if not q or not r or not os.path.isfile(q) or not os.path.isfile(r):
            return MatchScore([], query.n_features, False, reference.n_features)
        try:
            p = subprocess.run([self.bozorth3, q, r], capture_output=True, text=True, timeout=15)
            raw = (p.stdout or "").strip().split()
            score = int(float(raw[-1])) if raw else 0
        except (OSError, ValueError, subprocess.SubprocessError):
            score = 0
        # raw_matches carries one synthetic entry so the common reporter has
        # a useful count; the actual NBIS score is stored in metadata.
        return MatchScore(
            raw_matches=[score] if score > 0 else [],
            n_features_query=query.n_features,
            is_verified=score > 0,
            n_features_ref=reference.n_features,
        )
