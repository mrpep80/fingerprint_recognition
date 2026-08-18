from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FusedResult:
    filename: str
    combined_score: float
    per_method: Dict[str, float]
    best_method: str
    n_active: int
    inliers: Dict[str, int]
    ranks: Dict[str, int]
    consensus: float = 0.0
    matched_finger: str = "n/a"

    def __lt__(self, other):
        return self.combined_score < other.combined_score


class ScoreFusion:
    """Evidence fusion for heterogeneous fingerprint matchers.

    Scores are assumed to be converted by each matcher/scorer to 0..100.
    Missing/zero evidence is deliberately kept in the denominator: a candidate
    supported by only one engine must not receive the same score as one with
    independent support from several engines.
    """

    def __init__(self, support_threshold: float = 5.0,
                 support_thresholds: Optional[Dict[str, float]] = None):
        self.support_threshold = float(support_threshold)
        self.support_thresholds = support_thresholds or {}

    def _threshold(self, method: str) -> float:
        return float(self.support_thresholds.get(method, self.support_threshold))

    def fuse(self, results_by_method: Dict[str, List],
             weights: Optional[Dict[str, float]] = None) -> List[FusedResult]:
        if not results_by_method:
            return []
        methods = list(results_by_method)
        weights = weights or {m: 1.0 for m in methods}
        weights = {m: max(float(weights.get(m, 0.0)), 0.0) for m in methods}
        active_methods = [m for m in methods if weights[m] > 0]
        if not active_methods:
            return []

        files = sorted({r.filename for rs in results_by_method.values() for r in rs})
        raw = {f: {m: 0.0 for m in methods} for f in files}
        inl = {f: {m: 0 for m in methods} for f in files}
        fingers = {f: {} for f in files}
        for m, rs in results_by_method.items():
            for r in rs:
                raw[r.filename][m] = float(r.score)
                inl[r.filename][m] = int(r.n_inliers)
                finger = getattr(r, "matched_finger", "n/a")
                if finger != "n/a":
                    fingers[r.filename][m] = finger

        ranks = {
            m: {f: i + 1 for i, f in enumerate(
                sorted(files, key=lambda x: raw[x][m], reverse=True))}
            for m in methods
        }
        n = len(files)
        denom = sum(weights[m] for m in active_methods) or 1.0
        fused = []

        for f in files:
            positive = [m for m in active_methods if raw[f][m] > 0.0]
            if not positive:
                best_raw_method = max(active_methods, key=lambda m: raw[f][m])
                positive = [best_raw_method]

            parts = []
            for m in positive:
                percentile = 1.0 - (ranks[m][f] - 1) / max(n - 1, 1)
                confidence = np.clip(raw[f][m] / 100.0, 0, 1)
                evidence = (confidence ** 0.72) * (percentile ** 0.28)
                parts.append(evidence * weights[m])

            # IMPORTANT: denom includes every active engine, including engines
            # that returned zero evidence for this candidate. This prevents a
            # one-engine false positive from being normalized as if it had
            # multi-engine support.
            base = sum(parts) / denom
            supporters = [m for m in active_methods
                          if raw[f][m] >= self._threshold(m)]
            consensus = len(supporters) / len(active_methods)

            # Keep a modest floor so a genuine single-engine result is not
            # discarded completely, while still rewarding independent support.
            agreement = 0.55 + 0.45 * consensus
            combined = 100.0 * base * agreement

            best = max(active_methods, key=lambda m: raw[f][m])
            # Never borrow a finger label from another engine: otherwise the
            # reported finger can disagree with the engine that supplied the
            # best score.
            finger = fingers[f].get(best, "n/a")
            fused.append(FusedResult(
                filename=f,
                combined_score=float(np.clip(combined, 0, 100)),
                per_method=raw[f],
                best_method=best,
                n_active=len(positive),
                inliers=inl[f],
                ranks={m: ranks[m][f] for m in methods},
                consensus=consensus,
                matched_finger=finger,
            ))

        fused.sort(key=lambda r: (r.combined_score, r.consensus,
                                  max(r.inliers.values())), reverse=True)
        return fused
