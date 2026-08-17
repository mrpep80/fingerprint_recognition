from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseExtractor, FeatureSet
from ..engines.nbis import get_nbis_runtime


class NBISExtractor(BaseExtractor):
    """MINDTCT adapter using the portable desktop NBIS runtime."""

    name = "nbis"
    needs_raw_input = True

    def __init__(self, config):
        super().__init__(config)
        self._source_path: Optional[str] = None
        self._tmp_roots: list[str] = []
        self.runtime = get_nbis_runtime()
        self.mindtct = os.environ.get("NBIS_MINDTCT")
        if not self.mindtct and self.runtime.available:
            self.mindtct = str(self.runtime.mindtct)
        if not self.mindtct:
            self.mindtct = shutil.which("mindtct") or shutil.which("mindtct.exe")

    def set_source_path(self, path: Optional[str]) -> None:
        self._source_path = path

    def is_available(self) -> bool:
        return bool(self.mindtct)

    def _make_compatible_input(self, img: np.ndarray, source_path: Optional[str]):
        if source_path and Path(source_path).is_file() and Path(source_path).suffix.lower() in {".jpg", ".jpeg", ".wsq"}:
            return source_path, None
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray, dtype=np.uint8)
        fd, temp_input = tempfile.mkstemp(prefix="nbis_fp_", suffix=".jpg")
        os.close(fd)
        if not cv2.imwrite(temp_input, gray, [cv2.IMWRITE_JPEG_QUALITY, 100]):
            try:
                os.unlink(temp_input)
            except OSError:
                pass
            raise RuntimeError("impossibile scrivere input JPEG temporaneo per NBIS")
        return temp_input, temp_input

    def extract(self, img: np.ndarray) -> FeatureSet:
        if not self.mindtct:
            return FeatureSet(metadata={"source": "nbis", "error": "NBIS non disponibile"}, source=self.name)
        temp_input = None
        try:
            input_path, temp_input = self._make_compatible_input(img, self._source_path)
        except Exception as exc:
            return FeatureSet(metadata={"source": "nbis", "error": str(exc)}, source=self.name)
        root_dir = tempfile.mkdtemp(prefix="nbis_mindtct_")
        root = os.path.join(root_dir, "fp")
        self._tmp_roots.append(root_dir)
        try:
            proc = subprocess.run([self.mindtct, "-b", "-m1", input_path, root],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=30)
            xyt = root + ".xyt"
            if proc.returncode != 0 or not os.path.isfile(xyt):
                return FeatureSet(metadata={"source": "nbis", "error": (proc.stderr or proc.stdout).strip()}, source=self.name)
            minutiae = []
            with open(xyt, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, theta, quality = map(float, parts[:4])
                        minutiae.append((x, y, theta, quality))
                    except ValueError:
                        continue
            keypoints = [cv2.KeyPoint(float(x), float(y), 1.0, float(theta), float(q))
                         for x, y, theta, q in minutiae]
            return FeatureSet(keypoints=keypoints, descriptors=None,
                              metadata={"source": "nbis", "xyt_path": xyt,
                                        "minutiae": minutiae,
                                        "mindtct_stderr": proc.stderr.strip()},
                              source=self.name)
        finally:
            if temp_input:
                try:
                    os.unlink(temp_input)
                except OSError:
                    pass

    def __del__(self):
        for root_dir in getattr(self, "_tmp_roots", []):
            shutil.rmtree(root_dir, ignore_errors=True)
