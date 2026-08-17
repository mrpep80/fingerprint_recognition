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


class NBISExtractor(BaseExtractor):
    """MINDTCT adapter.

    MINDTCT is kept independent from the OpenCV preprocessing pipeline: NBIS
    receives the original fingerprint image when a source path is available.
    For card ROIs (which have no source file) a temporary 8-bit grayscale PNG
    is created.
    """

    name = "nbis"
    needs_raw_input = True

    def __init__(self, config):
        super().__init__(config)
        self._source_path: Optional[str] = None
        self._tmp_roots: list[str] = []
        self.mindtct = (
            os.environ.get("NBIS_MINDTCT")
            or shutil.which("mindtct")
            or shutil.which("mindtct.exe")
        )

    def set_source_path(self, path: Optional[str]) -> None:
        self._source_path = path

    def is_available(self) -> bool:
        return bool(self.mindtct)

    def extract(self, img: np.ndarray) -> FeatureSet:
        if not self.mindtct:
            return FeatureSet(metadata={"source": "nbis", "error": "mindtct non trovato"}, source=self.name)

        input_path = self._source_path
        temp_input = None
        if not input_path or not Path(input_path).is_file():
            fd, temp_input = tempfile.mkstemp(prefix="nbis_fp_", suffix=".png")
            os.close(fd)
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.asarray(gray, dtype=np.uint8)
            if not cv2.imwrite(temp_input, gray):
                return FeatureSet(metadata={"source": "nbis", "error": "impossibile scrivere input temporaneo"}, source=self.name)
            input_path = temp_input

        root_dir = tempfile.mkdtemp(prefix="nbis_mindtct_")
        root = os.path.join(root_dir, "fp")
        self._tmp_roots.append(root_dir)

        try:
            cmd = [self.mindtct, "-b", "-m1", input_path, root]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
            return FeatureSet(
                keypoints=keypoints,
                descriptors=None,
                metadata={
                    "source": "nbis",
                    "xyt_path": xyt,
                    "minutiae": minutiae,
                    "mindtct_stderr": proc.stderr.strip(),
                },
                source=self.name,
            )
        finally:
            if temp_input:
                try:
                    os.unlink(temp_input)
                except OSError:
                    pass

    def __del__(self):
        for root_dir in getattr(self, "_tmp_roots", []):
            shutil.rmtree(root_dir, ignore_errors=True)
