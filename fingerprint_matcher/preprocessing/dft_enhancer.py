"""
preprocessing/dft_enhancer.py
==============================
DFT-based fingerprint image enhancement.

Basato su:
  Bhowmik et al., "Fingerprint Image Enhancement And Its Feature
  Extraction For Recognition", IJSTR Vol.1 Issue 5, 2012.

Formula (eq. 2.1):
    i_enh(x,y) = F^-1 [ F(u,v) × |F(u,v)|^k ]

Effetto:
  - Amplifica le frequenze dominanti (creste) più del rumore
  - k=0.3  → miglioramento leggero, ideale per immagini già buone
  - k=0.4  → bilanciato (valore del paper)
  - k=0.5+ → aggressivo, può saturare
  - k=0.0  → nessun effetto (immagine originale)

Utilizzo tipico nel pipeline completo (Paper 1):
    Raw → DFT (k=0.3) → Histogram Equalization → fingerprint_enhancer → feature
"""

import numpy as np
import cv2
from ..config import Config


class DFTEnhancer:
    """
    Miglioramento dell'impronta tramite filtro nel dominio della frequenza.
    Implementa il metodo di Bhowmik et al. (IJSTR 2012).
    """

    def __init__(self, config: Config):
        self.cfg = config

    def enhance(self, gray: np.ndarray, k: float = None) -> np.ndarray:
        """
        Applica il filtro DFT per evidenziare le creste.

        Args:
            gray : immagine in scala di grigi (uint8)
            k    : esponente dello spettro di potenza (None = usa cfg.dft_k)

        Returns:
            immagine filtrata, normalizzata in [0, 255], dtype uint8
        """
        if k is None:
            k = self.cfg.dft_k

        if k <= 0.0:
            return gray   # nessun effetto

        f    = gray.astype(np.float32)
        dft  = np.fft.fft2(f)
        mag  = np.abs(dft)

        # i_enh = F^-1 [ F(u,v) × |F(u,v)|^k ]
        enh  = dft * (mag ** k)
        out  = np.real(np.fft.ifft2(enh))

        return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
