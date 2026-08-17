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
    """Fusion multi-metodo senza min-max sul database.

    Usa score assoluti + percentile/rank + consenso tra metodi. In questo
    modo un falso positivo che e' semplicemente il migliore del database non
    diventa automaticamente 100%.
    """

    def __init__(self, threshold: float = 12.0):
        self.threshold = threshold

    def fuse(self, results_by_method: Dict[str, List],
             weights: Optional[Dict[str, float]] = None) -> List[FusedResult]:
        if not results_by_method:
            return []
        methods = list(results_by_method)
        weights = weights or {m: 1.0 for m in methods}
        weights = {m: max(float(weights.get(m, 0.0)), 0.0) for m in methods}

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
            m: {f: i + 1 for i, f in enumerate(sorted(files, key=lambda x: raw[x][m], reverse=True))}
            for m in methods
        }
        n = len(files)
        fused = []

        for f in files:
            active = [m for m in methods if raw[f][m] >= self.threshold]
            if not active:
                active = [max(methods, key=lambda m: raw[f][m])]

            parts = []
            for m in active:
                percentile = 1.0 - (ranks[m][f] - 1) / max(n - 1, 1)
                confidence = np.clip(raw[f][m] / 100.0, 0, 1)
                parts.append((percentile ** 0.55) * (confidence ** 0.45) * weights[m])

            denom = sum(weights[m] for m in active) or 1.0
            base = sum(parts) / denom
            consensus = len(active) / len(methods)
            agreement = 0.70 + 0.30 * consensus
            combined = 100.0 * base * agreement
            best = max(methods, key=lambda m: raw[f][m])
            finger = fingers[f].get(best, next(iter(fingers[f].values()), "n/a"))

            fused.append(FusedResult(
                filename=f,
                combined_score=float(np.clip(combined, 0, 100)),
                per_method=raw[f],
                best_method=best,
                n_active=len(active),
                inliers=inl[f],
                ranks={m: ranks[m][f] for m in methods},
                consensus=consensus,
                matched_finger=finger,
            ))

        fused.sort(key=lambda r: (r.combined_score, r.n_active,
                                  max(r.inliers.values())), reverse=True)
        return fused
