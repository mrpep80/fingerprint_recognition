"""
preprocessing/hong_enhancer.py
================================
Fingerprint enhancement using Hong-Jain (1998) approach.

Libreria: fingerprint-enhancer (pip install fingerprint-enhancer)

Il metodo Hong-Jain è lo standard de facto per il pre-processing di impronte.
Implementa:
  1. Ridge segmentation   – rimuove le regioni di sfondo a bassa qualità
  2. Orientation field    – stima il campo di orientazione locale (gradient)
  3. Frequency estimation – stima la frequenza locale delle creste
  4. Gabor filtering      – applica filtri Gabor adattativi (orientazione+freq)
  5. Binarization         – soglia il risultato Gabor

Output: mappa binaria delle creste (True=cresta, False=valle/sfondo)

Citato in:
  BoLiu & Barczak (2009): "Image quality is the key step in preprocessing"
  Sahu & Shukla (2013): Fingerprint enhancement before SURF/SIFT extraction
"""

from __future__ import annotations
import numpy as np
import cv2
from ..config import Config


class HongEnhancer:
    """
    Wrapper per la libreria fingerprint-enhancer.
    Restituisce una mappa binaria (uint8: 0 o 255) delle creste.
    """

    def __init__(self, config: Config):
        self.cfg          = config
        self._available   = None     # lazy: valutato solo al primo accesso
        self._enhance_fn  = None

    @property
    def available(self) -> bool:
        """Controlla la disponibilità al primo accesso (import lazy)."""
        if self._available is None:
            self._available = self._check()
        return self._available

    def _check(self) -> bool:
        try:
            from fingerprint_enhancer.fingerprint_image_enhancer import enhance_fingerprint
            self._enhance_fn = enhance_fingerprint
            return True
        except ImportError:
            return False

    def enhance(self, gray: np.ndarray) -> np.ndarray:
        """
        Applica il pipeline Hong-Jain all'immagine.

        Args:
            gray : immagine in scala di grigi (uint8)

        Returns:
            binary ridge map uint8 in {0, 255}
            255 = cresta, 0 = valle/sfondo
        """
        if not self._available:
            return gray

        try:
            result = self._enhance_fn(
                gray,
                ridge_segment_blksze   = self.cfg.hong_block_size,
                ridge_segment_thresh   = self.cfg.hong_seg_thresh,
                gradient_sigma         = self.cfg.hong_grad_sigma,
                block_sigma            = self.cfg.hong_block_sigma,
                orient_smooth_sigma    = self.cfg.hong_orient_sigma,
                ridge_freq_blksze      = self.cfg.hong_freq_blksize,
                ridge_freq_windsze     = self.cfg.hong_freq_windsize,
                min_wave_length        = self.cfg.hong_min_wavelength,
                max_wave_length        = self.cfg.hong_max_wavelength,
                angle_inc              = self.cfg.hong_angle_inc,
                ridge_filter_thresh    = self.cfg.hong_filter_thresh,
            )
            # result è booleano → converti in uint8
            return result.astype(np.uint8) * 255
        except Exception:
            return gray
