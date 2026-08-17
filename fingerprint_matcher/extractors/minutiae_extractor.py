"""
extractors/minutiae_extractor.py
=================================
Estrattore di minuzie dattiloscopiche.

Strategia in cascata:
  1. NBIS mindtct  – se installato (sudo apt install nbis)
  2. OpenCV puro   – fallback sempre disponibile

FIX v2.1 – riduzione falsi positivi
-------------------------------------
Il crossing-number grezzo produce 700-1800 minuzie (per lo più rumore).
Un dito reale ne ha 30-80. Con troppe minuzie su entrambe le immagini,
RANSAC trova sempre un allineamento "perfetto" anche tra impronte diverse.

Soluzione: dopo il crossing-number, si calcola un punteggio di qualità
locale (varianza dell'immagine preprocessata intorno al punto) e si
mantengono solo le top-K (Config.minutiae_max_count, default 120).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseExtractor, FeatureSet
from ..config import Config


class MinutiaeExtractor(BaseExtractor):

    def __init__(self, config: Config):
        super().__init__(config)
        self._nbis_ok = shutil.which("mindtct") is not None
        backend = "NBIS mindtct" if self._nbis_ok else "OpenCV (fallback)"
        print(f"  [MinutiaeExtractor] backend: {backend}")

    @property
    def name(self) -> str:
        return "minutiae"

    def is_available(self) -> bool:
        return True

    def extract(self, img: np.ndarray) -> FeatureSet:
        img_r, scale = self._maybe_resize(img)

        if self._nbis_ok:
            fs = self._extract_nbis(img_r, scale)
            if fs.is_valid():
                return fs

        return self._extract_opencv(img_r)

    # ─── NBIS ────────────────────────────────────────────────────────

    def _extract_nbis(self, img: np.ndarray, scale: float) -> FeatureSet:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "fp.png")
            cv2.imwrite(img_path, img)
            prefix = os.path.join(tmpdir, "out")
            try:
                subprocess.run(["mindtct", img_path, prefix],
                               capture_output=True, timeout=30, check=True)
            except Exception:
                return FeatureSet(source="nbis_failed")

            xyt_path = prefix + ".xyt"
            if not os.path.exists(xyt_path):
                return FeatureSet(source="nbis_failed")

            minutiae = self._parse_xyt(xyt_path)

        # NBIS restituisce coordinate nell'immagine già ridimensionata
        minutiae = self._filter_by_quality(minutiae, img)
        return self._to_featureset(minutiae, source="nbis")

    def _parse_xyt(self, path: str) -> List[Dict]:
        minutiae = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x    = int(float(parts[0]))
                    y    = int(float(parts[1]))
                    theta = float(parts[2])
                    qual  = int(parts[3]) if len(parts) > 3 else 50
                    minutiae.append({"x": x, "y": y, "theta": theta,
                                     "type": "unknown", "quality": qual})
        return minutiae

    # ─── OpenCV puro ─────────────────────────────────────────────────

    def _extract_opencv(self, img: np.ndarray) -> FeatureSet:
        """
        Pipeline:
          1. Binarizza
          2. Scheletrizza (Zhang-Suen)
          3. Crossing-number → minuzie grezze
          4. Filtra spurie (bordo + prossimità)
          5. FIX: filtra per qualità locale → top-K
        """
        binary   = self._binarize(img)
        skeleton = self._skeletonize(binary)
        minutiae = self._crossing_number(skeleton)
        minutiae = self._filter_spurious(minutiae, skeleton)
        minutiae = self._filter_by_quality(minutiae, img)   # ← chiave del fix
        return self._to_featureset(minutiae, source="opencv")

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        binary  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2,
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        return binary

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        try:
            return cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except AttributeError:
            return self._morphological_thin(binary)

    def _morphological_thin(self, binary: np.ndarray) -> np.ndarray:
        skel = np.zeros_like(binary)
        el   = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img  = binary.copy()
        for _ in range(200):
            eroded = cv2.erode(img, el)
            opened = cv2.dilate(eroded, el)
            skel   = cv2.bitwise_or(skel, cv2.subtract(img, opened))
            img    = eroded
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _crossing_number(self, skeleton: np.ndarray) -> List[Dict]:
        """
        Calcolo vettorizzato del crossing number.
        CN=1 → terminazione, CN=3 → biforcazione.
        """
        sk = (skeleton > 0).astype(np.uint8)

        p = [sk[:-2, 1:-1], sk[:-2, 2:],  sk[1:-1, 2:], sk[2:, 2:],
             sk[2:, 1:-1],  sk[2:, :-2],   sk[1:-1, :-2], sk[:-2, :-2]]

        cn = np.zeros(p[0].shape, dtype=np.uint8)
        for i in range(8):
            cn += np.abs(p[i].astype(int) - p[(i + 1) % 8].astype(int)).astype(np.uint8)
        cn = cn // 2

        center = sk[1:-1, 1:-1]

        endings      = np.argwhere((cn == 1) & (center == 1)) + 1
        bifurcations = np.argwhere((cn == 3) & (center == 1)) + 1

        minutiae: List[Dict] = []
        for y, x in endings:
            minutiae.append({
                "x": int(x), "y": int(y), "type": "ending",
                "theta": self._local_orientation(sk, int(x), int(y)),
                "quality": 0.0,
            })
        for y, x in bifurcations:
            minutiae.append({
                "x": int(x), "y": int(y), "type": "bifurcation",
                "theta": self._local_orientation(sk, int(x), int(y)),
                "quality": 0.0,
            })
        return minutiae

    def _local_orientation(self, sk, x, y, radius=8):
        h, w = sk.shape
        patch = sk[max(0,y-radius):min(h,y+radius),
                   max(0,x-radius):min(w,x+radius)].astype(np.float32)
        if patch.size == 0:
            return 0.0
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.degrees(np.arctan2(gy.sum(), gx.sum())) % 180)

    def _filter_spurious(self, minutiae: List[Dict],
                          skeleton: np.ndarray) -> List[Dict]:
        """Rimuove minuzie al bordo e troppo vicine tra loro."""
        h, w = skeleton.shape
        m    = self.cfg.minutiae_border_margin
        dmin = self.cfg.minutiae_min_distance

        clean = [p for p in minutiae
                 if m < p["x"] < w - m and m < p["y"] < h - m]

        kept: List[Dict] = []
        for p in clean:
            if not any(
                (p["x"] - k["x"])**2 + (p["y"] - k["y"])**2 < dmin**2
                for k in kept
            ):
                kept.append(p)
        return kept

    def _filter_by_quality(self, minutiae: List[Dict],
                            img: np.ndarray) -> List[Dict]:
        """
        FIX CENTRALE: assegna un punteggio di qualità a ogni minuzia
        in base alla varianza locale dell'immagine preprocessata,
        poi mantiene solo le top minutiae_max_count.

        La varianza locale è un proxy della chiarezza delle creste:
        alta varianza → creste nette → minuzia affidabile.
        Bassa varianza → zona omogenea o rumore → minuzia spuria.
        """
        max_k = self.cfg.minutiae_max_count
        if not minutiae:
            return minutiae

        h, w = img.shape[:2]
        r    = self.cfg.minutiae_quality_radius

        for m in minutiae:
            x, y = m["x"], m["y"]
            patch = img[max(0, y-r):min(h, y+r),
                        max(0, x-r):min(w, x+r)]
            m["quality"] = float(np.var(patch.astype(float))) if patch.size > 4 else 0.0

        # Ordina per qualità decrescente
        minutiae.sort(key=lambda m: -m["quality"])

        # Le biforcazioni sono più affidabili delle terminazioni:
        # le terminazioni sono spesso spuri da rumore o artifacts.
        # Assicuriamo un mix equilibrato.
        bifurcs  = [m for m in minutiae if m["type"] == "bifurcation"][:max_k // 2]
        endings  = [m for m in minutiae if m["type"] == "ending"][:max_k // 2]
        combined = bifurcs + endings
        combined.sort(key=lambda m: -m["quality"])
        return combined[:max_k]

    # ─── Utility ────────────────────────────────────────────────────

    def _maybe_resize(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        max_dim = self.cfg.minutiae_max_dim
        h, w    = img.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return img, 1.0
        scale = max_dim / longest
        return cv2.resize(img, (int(w*scale), int(h*scale)),
                          interpolation=cv2.INTER_AREA), scale

    def _to_featureset(self, minutiae: List[Dict], source: str) -> FeatureSet:
        if not minutiae:
            return FeatureSet(source=source)
        kp = [
            cv2.KeyPoint(float(m["x"]), float(m["y"]),
                         size=6.0, angle=float(m["theta"]),
                         response=float(m.get("quality", 50)))
            for m in minutiae
        ]
        return FeatureSet(
            keypoints=kp, descriptors=None,
            metadata={"minutiae": minutiae, "source": source},
            source=source,
        )
