"""
preprocessing/image_processor.py
=================================
Pipeline di pre-elaborazione delle immagini di impronta digitale.

Passaggi (nell'ordine):
  0. Resize     – porta l'immagine entro max_working_dim (gestisce qualsiasi MP)
  1. Grayscale  – conversione in scala di grigi
  2. CLAHE      – equalizzazione adattiva del contrasto locale
  3. Gabor      – evidenzia le creste alle diverse orientazioni (opzionale)
  4. CLAHE      – seconda passata per normalizzare il risultato Gabor
  5. Blur       – riduzione del rumore residuo

Il resize al passo 0 è trasparente: viene applicato prima di qualsiasi
elaborazione in modo che tutti i componenti successivi (SIFT, ORB, Minutiae)
ricevano sempre immagini di dimensioni gestibili, indipendentemente dalla
risoluzione originale della fotocamera.
"""

import cv2
import numpy as np
from ..config import Config


class ImageProcessor:
    """Pre-elaborazione delle immagini di impronta digitale."""

    def __init__(self, config: Config):
        self.cfg = config
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._gabor_kernels = self._build_gabor_kernels()

    # ── Metodi pubblici ───────────────────────────────────────────────

    def preprocess(self, img: np.ndarray, use_gabor: bool = True) -> np.ndarray:
        """
        Applica l'intera pipeline di pre-elaborazione.

        Funziona con qualsiasi dimensione di input: immagini da fotocamere
        da 1MP a 100MP vengono gestite automaticamente grazie al resize
        iniziale controllato da Config.max_working_dim.

        Args:
            img:       immagine in ingresso (BGR o grigio, qualsiasi risoluzione)
            use_gabor: applica i filtri Gabor (più accurato, ~3x più lento)

        Returns:
            immagine pre-elaborata in scala di grigi, bounded da max_working_dim
        """
        gray = self.to_gray(img)
        gray = self.resize_to_working(gray)   # passo 0: porta a dimensione gestibile
        enh  = self.apply_clahe(gray)
        if use_gabor:
            enh = self.apply_clahe(self.gabor_enhance(enh))
        return cv2.GaussianBlur(enh, (3, 3), 0)

    def to_gray(self, img: np.ndarray) -> np.ndarray:
        """Converte in scala di grigi. No-op se già grigio."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    def resize_to_working(self, gray: np.ndarray) -> np.ndarray:
        """
        Ridimensiona l'immagine al massimo max_working_dim px sul lato maggiore,
        preservando l'aspect ratio. No-op se già entro i limiti o max_working_dim=0.

        Questo è il punto centrale che rende il sistema indipendente dalla
        risoluzione della fotocamera: qualsiasi immagine viene portata a una
        dimensione in cui tutti gli algoritmi lavorano in modo efficiente.
        """
        max_dim = self.cfg.max_working_dim
        if max_dim <= 0:
            return gray   # resize disabilitato

        h, w    = gray.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return gray   # già entro i limiti

        scale = max_dim / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Equalizzazione adattiva del contrasto (CLAHE)."""
        return self._clahe.apply(gray)

    def gabor_enhance(self, gray: np.ndarray) -> np.ndarray:
        """
        Filtraggio Gabor multi-orientazione per evidenziare le creste.
        Applica GABOR_N filtri e restituisce il massimo delle risposte.
        """
        responses = [
            cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, k)
            for k in self._gabor_kernels
        ]
        enhanced = np.max(np.stack(responses, axis=-1), axis=-1)
        return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Metodi privati ────────────────────────────────────────────────

    def _build_gabor_kernels(self):
        """Pre-costruisce i kernel Gabor per tutte le orientazioni."""
        kernels = []
        for i in range(self.cfg.gabor_n_orientations):
            theta = i * np.pi / self.cfg.gabor_n_orientations
            k = cv2.getGaborKernel(
                (self.cfg.gabor_kernel_size, self.cfg.gabor_kernel_size),
                sigma=self.cfg.gabor_sigma,
                theta=theta,
                lambd=self.cfg.gabor_lambda,
                gamma=self.cfg.gabor_gamma,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernels.append(k)
        return kernels
