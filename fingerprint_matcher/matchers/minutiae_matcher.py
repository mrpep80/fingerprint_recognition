from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np
from .base import BaseMatcher, MatchScore
from ..extractors.base import FeatureSet


class MinutiaeMatcher(BaseMatcher):
    """Matching one-to-one delle minuzie + RANSAC affine."""

    ROTATION_SWEEP = [0, 30, 60, 90, 120, 150]

    @property
    def name(self) -> str:
        return "minutiae"

    def match(self, query: FeatureSet, reference: FeatureSet) -> MatchScore:
        q_min = query.metadata.get("minutiae", [])
        r_min = reference.metadata.get("minutiae", [])
        n_q, n_r = len(q_min), len(r_min)
        if n_q < 4 or n_r < 4:
            return MatchScore([], n_q, is_verified=True, n_features_ref=n_r)

        q_pts = np.float32([[m["x"], m["y"]] for m in q_min])
        r_pts = np.float32([[m["x"], m["y"]] for m in r_min])
        cx, cy = float(q_pts[:, 0].mean()), float(q_pts[:, 1].mean())
        best_pairs: List[Tuple[int, int]] = []

        for angle in self.ROTATION_SWEEP:
            q_rot = self._rotate_points(q_pts, cx, cy, angle)
            q_rot_meta = [{**m, "theta": (m["theta"] + angle) % 180} for m in q_min]
            cq, cr = self._nn_one_to_one(q_rot, r_pts, q_rot_meta, r_min)
            if len(cq) < 4:
                continue
            pairs = self._ransac_affine(q_rot, r_pts, cq, cr)
            if len(pairs) > len(best_pairs):
                best_pairs = pairs

        return MatchScore(best_pairs, n_q, is_verified=True, n_features_ref=n_r)

    @staticmethod
    def _rotate_points(pts, cx, cy, angle_deg):
        if angle_deg == 0:
            return pts
        rad = np.radians(angle_deg)
        ca, sa = np.cos(rad), np.sin(rad)
        p = pts - np.array([cx, cy])
        return np.column_stack((p[:, 0] * ca - p[:, 1] * sa + cx,
                                p[:, 0] * sa + p[:, 1] * ca + cy)).astype(np.float32)

    def _nn_one_to_one(self, q_pts, r_pts, q_min, r_min):
        threshold = self.cfg.minutiae_dist_tolerance * self.cfg.minutiae_nn_loose_ratio
        angle_tol = self.cfg.minutiae_angle_tolerance
        candidates = []
        for qi, qp in enumerate(q_pts):
            d = np.hypot(r_pts[:, 0] - qp[0], r_pts[:, 1] - qp[1])
            for ri in np.argsort(d):
                if d[ri] >= threshold:
                    break
                qa, ra = q_min[qi]["theta"], r_min[ri]["theta"]
                ad = abs(qa - ra) % 180
                if ad <= angle_tol or 180 - ad <= angle_tol:
                    candidates.append((float(d[ri]), qi, int(ri)))
                    break

        # Un solo riferimento per minuzia e una sola query per riferimento.
        candidates.sort(key=lambda x: x[0])
        used_q, used_r, cq, cr = set(), set(), [], []
        for _, qi, ri in candidates:
            if qi in used_q or ri in used_r:
                continue
            used_q.add(qi); used_r.add(ri)
            cq.append(qi); cr.append(ri)
        return cq, cr

    def _ransac_affine(self, q_pts, r_pts, cq, cr):
        src = q_pts[cq].reshape(-1, 1, 2)
        dst = r_pts[cr].reshape(-1, 1, 2)
        try:
            _, mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.minutiae_dist_tolerance,
                maxIters=3000, confidence=0.995,
            )
        except cv2.error:
            return []
        if mask is None:
            return []
        return [(cq[i], cr[i]) for i, v in enumerate(mask.ravel()) if v]
