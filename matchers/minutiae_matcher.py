"""
matchers/minutiae_matcher.py
==============================
Matching basato su minuzie con invarianza alla rotazione.

Pipeline v2.2
-------------
1. Multi-angle sweep: prova 6 rotazioni discrete (0°, 30°, 60°, 90°, 120°, 150°).
   Per ogni rotazione, ruota le coordinate delle minuzie query e trova i NN.
   Questo rende il matching robusto a qualsiasi orientazione del frammento.
2. NN pass con filtro angolare (distanza + orientazione).
3. estimateAffinePartial2D + RANSAC (soglia 12px).
4. Restituisce il miglior set di inlier tra tutte le rotazioni.
"""

from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

from .base import BaseMatcher, MatchScore
from ..extractors.base import FeatureSet


class MinutiaeMatcher(BaseMatcher):

    # Angoli da provare nel sweep (gradi)
    ROTATION_SWEEP = [0, 30, 60, 90, 120, 150]

    @property
    def name(self) -> str:
        return "minutiae"

    def match(self, query: FeatureSet, reference: FeatureSet) -> MatchScore:
        q_min = query.metadata.get("minutiae", [])
        r_min = reference.metadata.get("minutiae", [])
        n_q   = len(q_min)
        n_r   = len(r_min)

        if n_q < 4 or n_r < 4:
            return MatchScore([], n_q, is_verified=True, n_features_ref=n_r)

        q_pts = np.float32([[m["x"], m["y"]] for m in q_min])
        r_pts = np.float32([[m["x"], m["y"]] for m in r_min])

        # Centro dell'immagine query (usato per la rotazione)
        cx = float(q_pts[:, 0].mean())
        cy = float(q_pts[:, 1].mean())

        best_pairs: List[Tuple[int, int]] = []

        for angle in self.ROTATION_SWEEP:
            # Ruota le coordinate dei keypoints query
            q_pts_rot = self._rotate_points(q_pts, cx, cy, angle)

            # Angoli delle minuzie query aggiornati per la rotazione
            q_min_rot = [{**m, "theta": (m["theta"] + angle) % 180}
                         for m in q_min]

            cands_q, cands_r = self._nn_with_angle_filter(
                q_pts_rot, r_pts, q_min_rot, r_min
            )
            if len(cands_q) < 4:
                continue

            pairs = self._ransac_affine(q_pts_rot, r_pts, cands_q, cands_r)
            if len(pairs) > len(best_pairs):
                best_pairs = pairs

        return MatchScore(best_pairs, n_q, is_verified=True, n_features_ref=n_r)

    # ── Rotazione coordinate ──────────────────────────────────────────

    @staticmethod
    def _rotate_points(pts: np.ndarray, cx: float, cy: float,
                       angle_deg: float) -> np.ndarray:
        """Ruota i punti di angle_deg gradi intorno al centroide."""
        if angle_deg == 0:
            return pts
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        p  = pts - np.array([cx, cy])
        rx = p[:, 0] * cos_a - p[:, 1] * sin_a
        ry = p[:, 0] * sin_a + p[:, 1] * cos_a
        return np.column_stack([rx + cx, ry + cy]).astype(np.float32)

    # ── NN con filtro angolare ────────────────────────────────────────

    def _nn_with_angle_filter(
        self,
        q_pts: np.ndarray, r_pts: np.ndarray,
        q_min: list, r_min: list,
    ) -> Tuple[List[int], List[int]]:
        threshold = self.cfg.minutiae_dist_tolerance * self.cfg.minutiae_nn_loose_ratio
        angle_tol = self.cfg.minutiae_angle_tolerance

        cands_q: List[int] = []
        cands_r: List[int] = []

        for qi, qp in enumerate(q_pts):
            diffs = r_pts - qp
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            order = np.argsort(dists)

            for ri in order:
                if dists[ri] >= threshold:
                    break
                qa   = q_min[qi]["theta"]
                ra   = r_min[ri]["theta"]
                diff = abs(qa - ra) % 180
                if diff <= angle_tol or (180 - diff) <= angle_tol:
                    cands_q.append(qi)
                    cands_r.append(int(ri))
                    break

        return cands_q, cands_r

    # ── RANSAC affine ─────────────────────────────────────────────────

    def _ransac_affine(
        self,
        q_pts: np.ndarray, r_pts: np.ndarray,
        cands_q: List[int], cands_r: List[int],
    ) -> List[Tuple[int, int]]:
        src = q_pts[cands_q].reshape(-1, 1, 2)
        dst = r_pts[cands_r].reshape(-1, 1, 2)
        try:
            _, mask = cv2.estimateAffinePartial2D(
                src, dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.minutiae_dist_tolerance,
                maxIters=2000,
                confidence=0.99,
            )
        except cv2.error:
            return []
        if mask is None:
            return []
        return [(cands_q[i], cands_r[i]) for i, v in enumerate(mask.ravel()) if v]
