"""
preprocessing/card_segmenter.py
================================
Segmentazione intelligente delle immagini di riferimento.

Gestisce DUE tipi di immagini:
  1. Scheda dattiloscopica COMPLETA (es. modulo AFIS italiano)
     → estrae 10 impronte singole dalla griglia 5×2
     → per ogni modello: 10 confronti con il frammento

  2. Impronta SINGOLA (già ritagliata)
     → restituisce lista vuota (l'engine usa l'immagine intera)

Rilevamento automatico del tipo:
  - Scheda: formato ritratto, altezza > 1500px, aspect ratio < 0.85
  - Singola: tutto il resto

Strategia di estrazione per schede:
  A) Grid-based (principale): divide la zona impronte in griglia 5×2
     basandosi sui GAP bianchi identificati nel profilo di varianza verticale
  B) Blob-based (fallback): blob detection morfologica con parametri ridotti

Torna le ROI ordinate per posizione: [P.Dx, I.Dx, M.Dx, A.Dx, Mi.Dx,
                                       P.Sx, I.Sx, M.Sx, A.Sx, Mi.Sx]
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple
from ..config import Config


class CardSegmenter:
    """Segmenta schede dattiloscopiche ed estrae le singole impronte."""

    # Nomi delle 10 posizioni (usati per il debug)
    FINGER_NAMES = [
        "Pollice Dx",  "Indice Dx",  "Medio Dx",  "Anulare Dx",  "Mignolo Dx",
        "Pollice Sx",  "Indice Sx",  "Medio Sx",  "Anulare Sx",  "Mignolo Sx",
    ]

    def __init__(self, config: Config):
        self.cfg = config

    # ── API pubblica ───────────────────────────────────────────────────

    def is_card(self, img: np.ndarray) -> bool:
        """
        Determina se l'immagine è una scheda completa (vs impronta singola).
        Una scheda ha formato ritratto, è grande e ha aspect ratio < 0.85.
        """
        h, w = img.shape[:2]
        aspect = w / max(h, 1)
        return aspect < 0.85 and h > 1500 and w > 1000

    def extract_rois(self, card: np.ndarray) -> List[np.ndarray]:
        """
        Estrae le ROI dall'immagine.

        - Se è una scheda dattiloscopica: restituisce fino a 10 impronte ordinate
        - Se è un'impronta singola: restituisce lista vuota
          (l'engine userà l'intera immagine)

        Returns:
            lista di array numpy in scala di grigi, una per impronta
        """
        if not self.is_card(card):
            return []   # impronta singola → gestita dall'engine senza ROI

        gray = self._to_gray(card)
        h, w = gray.shape

        # Strategia A: grid-based (affidabile per schede standard)
        rois = self._extract_grid(gray)
        if len(rois) >= 8:
            return rois

        # Strategia B: blob-based (fallback)
        rois = self._extract_blobs(gray)
        return rois

    # ── Strategia A: Grid-based ────────────────────────────────────────

    def _extract_grid(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Estrae le 10 impronte usando la struttura a griglia della scheda.

        Algoritmo:
          1. Calcola la varianza locale per ogni riga orizzontale
          2. Trova i GAP (bande a bassa varianza = spazio bianco tra sezioni)
          3. Identifica le due bande principali con le impronte per rotazione
          4. Divide ciascuna banda in 5 colonne uguali → 10 celle
          5. Verifica che ogni cella contenga davvero un'impronta (varianza > soglia)
        """
        h, w = gray.shape

        # Le rolled impressions si trovano nella fascia centrale della scheda.
        # Limita la ricerca a questa zona per escludere header e plain impressions.
        # Proporzioni empiriche per scheda dattiloscopica italiana standard:
        #   header:            y = 0.00h – 0.18h
        #   rolled row 1:      y = 0.18h – 0.44h
        #   rolled row 2:      y = 0.44h – 0.70h
        #   plain impressions: y = 0.70h – 1.00h
        y_search_start = int(h * 0.15)
        y_search_end   = int(h * 0.72)   # escludi plain impressions
        search_region  = gray[y_search_start:y_search_end, :]

        # Calcola varianza locale per striscia orizzontale (finestra 40px)
        sr_h = search_region.shape[0]
        var_profile = np.array([
            float(search_region[max(0, y-20):min(sr_h, y+20), :].std())
            for y in range(sr_h)
        ])

        # Trova le BANDE ATTIVE (varianza > soglia = contenuto ricco)
        med_var = np.median(var_profile)
        active_thresh = max(med_var * 0.58, 42.0)
        is_active = var_profile > active_thresh

        # Raggruppa righe attive in bande continue
        bands = self._find_bands(is_active, min_height=180, min_gap=60)

        if len(bands) < 2:
            return []

        # Prendi le prime due bande nella zona di ricerca
        # (ordinate per posizione verticale)
        bands_sorted = sorted(bands, key=lambda b: b[0])
        # Scegli le 2 bande con la varianza media più alta (= le impronte)
        def band_score(b):
            return var_profile[b[0]:b[1]].mean() * (b[1]-b[0])
        fp_bands = sorted(bands_sorted, key=band_score, reverse=True)[:2]
        fp_bands = sorted(fp_bands)   # ri-ordina per posizione y

        if len(fp_bands) < 2:
            return []

        # Converti le coordinate dalla search_region alle coordinate originali
        row1 = (fp_bands[0][0] + y_search_start, fp_bands[0][1] + y_search_start)
        row2 = (fp_bands[1][0] + y_search_start, fp_bands[1][1] + y_search_start)

        # Estrai 5 celle per riga (divisione uniforme in 5 colonne)
        rois: List[np.ndarray] = []
        pad = self.cfg.roi_padding

        for y1, y2 in [row1, row2]:
            col_w = w // 5
            for col in range(5):
                x1 = max(0, col * col_w - pad // 2)
                x2 = min(w, (col + 1) * col_w + pad // 2)
                y1c = max(0, y1 - pad)
                y2c = min(h, y2 + pad)
                cell = gray[y1c:y2c, x1:x2]

                # Verifica: la cella contiene un'impronta? (varianza locale > soglia)
                if cell.size > 0 and cell.std() > 25.0:
                    rois.append(cell)
                else:
                    # Aggiungi comunque per mantenere l'allineamento 10 dita
                    rois.append(cell if cell.size > 0 else gray[0:10, 0:10])

        return rois

    def _find_bands(self, is_active: np.ndarray,
                    min_height: int = 200,
                    min_gap: int = 80) -> List[Tuple[int, int]]:
        """
        Trova bande contigue di righe attive, filtrando gap troppo piccoli.
        """
        h = len(is_active)
        bands: List[Tuple[int, int]] = []
        in_band = False
        start = 0
        gap_start = 0

        for y in range(h):
            if is_active[y] and not in_band:
                # Controlla se il gap precedente era abbastanza lungo
                if bands and (y - gap_start) < min_gap:
                    # Gap troppo corto → estendi la banda precedente
                    start_prev, _ = bands.pop()
                    start = start_prev
                else:
                    start = y
                in_band = True
            elif not is_active[y] and in_band:
                in_band = False
                gap_start = y
                if y - start >= min_height:
                    bands.append((start, y))

        if in_band and h - start >= min_height:
            bands.append((start, h))

        return bands

    # ── Strategia B: Blob-based (fallback) ───────────────────────────

    def _extract_blobs(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Fallback: trova blob scuri (impronte) con morfologia ridotta.
        Funziona anche su schede non standard.
        """
        h, w = gray.shape
        total = h * w

        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Morfologia RIDOTTA per non fondere le impronte adiacenti
        k_small = np.ones((3, 3), np.uint8)
        k_med   = np.ones((5, 5), np.uint8)
        closed  = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_small, iterations=2)
        dilated = cv2.dilate(closed, k_med, iterations=1)

        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        rois: List[np.ndarray] = []
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if not (total * 0.001 < area < total * 0.05):
                continue
            x, y, bw2, bh2 = cv2.boundingRect(cnt)
            asp = bw2 / max(bh2, 1)
            if not (0.25 < asp < 3.5):
                continue
            p = self.cfg.roi_padding
            roi = gray[max(0,y-p):min(h,y+bh2+p),
                       max(0,x-p):min(w,x+bw2+p)]
            if roi.shape[0] >= self.cfg.roi_min_side and \
               roi.shape[1] >= self.cfg.roi_min_side:
                rois.append(roi)
            if len(rois) >= 15:
                break

        return rois

    # ── Utility ───────────────────────────────────────────────────────

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
