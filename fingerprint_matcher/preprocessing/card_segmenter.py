"""
preprocessing/card_segmenter.py
================================
Estrazione delle 10 impronte da schede dattiloscopiche standard.

Approccio scale-invariant
--------------------------
NON usa coordinate pixel assolute o proporzioni fisse.
Invece, analizza la struttura visiva reale dell'immagine scansionata.

Tecnica chiave: run-length orizzontale
  Per ogni riga dell'immagine, misura la lunghezza massima dei pixel scuri
  contigui (max_run). Questo distingue:

  - LINEE DELLA GRIGLIA:  max_run > 20% larghezza (linea nera lunga)
  - CONTENUTO IMPRONTA:   dark_pct > 5%,  max_run < 20%  (creste curve)
  - TESTO/ETICHETTE:      max_run moderato in poche righe concentrate
  - SPAZIO VUOTO:         dark_pct ≈ 0%,  max_run ≈ 0

Pipeline di estrazione
----------------------
  1. Scansiona il profilo run-length verticale dell'intera immagine
  2. Identifica le LINEE STRUTTURALI (max_run > MIN_LINE_RATIO * width)
  3. Tra le linee strutturali, trova le ZONE DI CONTENUTO (fingerprint)
  4. La sezione "ROLLED IMPRESSIONS" = area con due zone di contenuto
     separate da una linea strutturale o da una zona vuota
  5. Divide quella sezione in 5 colonne uniformi
  6. Estrae 10 ROI: 5 per la mano destra + 5 per la mano sinistra

Funziona indipendentemente da:
  - Risoluzione scanner (200/300/400/600 DPI)
  - Posizione della scheda nell'area di scansione
  - Formato/dimensione del foglio
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple

from ..config import Config


class CardSegmenter:

    FINGER_NAMES = [
        "Pollice Dx",  "Indice Dx",  "Medio Dx",  "Anulare Dx",  "Mignolo Dx",
        "Pollice Sx",  "Indice Sx",  "Medio Sx",  "Anulare Sx",  "Mignolo Sx",
    ]

    MIN_VALID_ROIS = 8

    # Una riga con max_run > MIN_LINE_RATIO * width è una "linea strutturale"
    MIN_LINE_RATIO = 0.20   # 20% della larghezza

    # Soglie per distinguere contenuto da spazio vuoto
    CONTENT_DARK_PCT  = 0.05   # >5% pixel scuri = riga con contenuto
    CONTENT_MIN_ROWS  = 80     # un blocco deve essere alto almeno 80px

    def __init__(self, config: Config):
        self.cfg = config

    # ── API pubblica ──────────────────────────────────────────────────

    def is_card(self, img: np.ndarray) -> bool:
        h, w = img.shape[:2]
        return w / max(h, 1) < 0.85 and h > 1500 and w > 1000

    def extract_rois(self, card: np.ndarray) -> List[np.ndarray]:
        """
        Estrae le 10 ROI delle impronte dalla scheda.

        Strategia:
          A) Analisi run-length (principale, scale-invariant)
          B) Per-column density (fallback se A fallisce)
        """
        if not self.is_card(card):
            return []

        gray = self._to_gray(card)

        rois = self._extract_via_runlength(gray)
        valid = [r for r in rois if r.shape[0] > 60 and r.shape[1] > 60
                 and r.std() > 20]
        if len(valid) >= self.MIN_VALID_ROIS:
            return rois

        return self._extract_fallback(gray)

    # ─────────────────────────────────────────────────────────────────
    #  STRATEGIA A: Run-length analysis
    # ─────────────────────────────────────────────────────────────────

    def _extract_via_runlength(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Usa il profilo run-length per trovare la sezione delle impronte
        e le due sotto-righe (mano destra e sinistra).
        """
        h, w = gray.shape
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Calcola per ogni riga: max_run e dark_pct
        max_runs  = np.zeros(h, dtype=float)
        dark_pcts = np.zeros(h, dtype=float)
        for y in range(h):
            row          = bw[y, :]
            dark_pcts[y] = row.mean() / 255.0
            max_runs[y]  = self._max_run_length(row) / w

        # Trova le linee strutturali (max_run > soglia)
        struct_lines = self._find_structural_lines(max_runs)

        # Trova i blocchi di contenuto (fingerprint)
        content_blocks = self._find_content_blocks(dark_pcts, max_runs, h)

        # Identifica la sezione rolled impressions
        section = self._find_rolled_section(struct_lines, content_blocks, h)
        if section is None:
            return []

        y_top, y_bot, row1_blocks, row2_blocks = section

        # Trova i confini delle due righe di impronte
        row1_range = self._row_range(y_top, y_bot, row1_blocks, row2_blocks,
                                      struct_lines, dark_pcts, which=1)
        row2_range = self._row_range(y_top, y_bot, row1_blocks, row2_blocks,
                                      struct_lines, dark_pcts, which=2)

        if row1_range is None or row2_range is None:
            return []

        # Dividi in 5 colonne e estrai 10 celle
        col_dividers = self._column_dividers(gray, y_top, y_bot, w)
        rois: List[np.ndarray] = []

        for y1, y2 in [row1_range, row2_range]:
            for i in range(5):
                x1 = col_dividers[i]
                x2 = col_dividers[i + 1]
                cell = gray[y1:y2, x1:x2]
                # Salta l'etichetta in cima alla cella
                fp_start = self._find_fp_start(cell)
                roi = gray[y1 + fp_start:y2, x1:x2]
                rois.append(roi if roi.size > 0 else cell)

        return rois

    # ── Profilo run-length ────────────────────────────────────────────

    @staticmethod
    def _max_run_length(row: np.ndarray) -> int:
        """Lunghezza massima di pixel contigui scuri (>0) in una riga."""
        max_r, cur = 0, 0
        for px in row:
            if px > 0:
                cur += 1
                if cur > max_r:
                    max_r = cur
            else:
                cur = 0
        return max_r

    def _find_structural_lines(self, max_runs: np.ndarray) -> List[int]:
        """
        Trova le righe che sono linee strutturali della griglia
        (max_run > MIN_LINE_RATIO).
        Raggruppa righe adiacenti nello stesso "cluster".
        """
        is_line = max_runs > self.MIN_LINE_RATIO
        lines: List[int] = []
        in_line, start = False, 0

        for y, v in enumerate(is_line):
            if v and not in_line:
                in_line, start = True, y
            elif not v and in_line:
                in_line = False
                center = (start + y) // 2
                lines.append(center)

        if in_line:
            lines.append((start + len(is_line)) // 2)

        return lines

    def _find_content_blocks(
        self,
        dark_pcts: np.ndarray,
        max_runs:  np.ndarray,
        h:         int,
    ) -> List[Tuple[int, int]]:
        """
        Trova i blocchi di righe con contenuto reale:
        dark_pct > soglia E max_run < MIN_LINE_RATIO (= non è una linea).
        I blocchi "fingerprint" hanno contenuto distribuito, non righe lunghe.
        """
        is_content = (dark_pcts > self.CONTENT_DARK_PCT) & \
                     (max_runs  < self.MIN_LINE_RATIO)

        blocks: List[Tuple[int, int]] = []
        in_block, start, gap = False, 0, 0

        for y in range(h):
            if is_content[y]:
                if not in_block:
                    in_block, start, gap = True, y, 0
                else:
                    gap = 0
            else:
                if in_block:
                    gap += 1
                    if gap > 60:
                        in_block = False
                        if y - start - gap >= self.CONTENT_MIN_ROWS:
                            blocks.append((start, y - gap))

        if in_block and h - start >= self.CONTENT_MIN_ROWS:
            blocks.append((start, h))

        return blocks

    # ── Sezione rolled impressions ───────────────────────────────────

    def _find_rolled_section(
        self,
        struct_lines:   List[int],
        content_blocks: List[Tuple[int, int]],
        h:              int,
    ) -> Optional[Tuple[int, int, List, List]]:
        """
        Identifica la sezione delle rolled impressions come l'area che:
          - Contiene ESATTAMENTE DUE blocchi di contenuto separati
          - I due blocchi = mano destra e mano sinistra

        Strategia:
          Cerca la coppia (b1, b2) di content_blocks più grandi e
          verticalmente separati (tra loro c'è uno spazio o una linea).
          L'area va dal bordo superiore di b1 al bordo inferiore di b2.
        """
        if len(content_blocks) < 2:
            return None

        # Ordina per dimensione (i due blocchi più grandi)
        sorted_by_size = sorted(content_blocks,
                                key=lambda b: b[1]-b[0], reverse=True)
        top2 = sorted(sorted_by_size[:2], key=lambda b: b[0])
        b1, b2 = top2[0], top2[1]

        # Verifica che siano separati (b1 termina prima che b2 inizi)
        if b1[1] >= b2[0]:
            return None

        # I confini della sezione: espandi un po' sopra b1 e sotto b2
        # per includere le etichette
        pad = 20
        y_top = max(0, b1[0] - 200)   # 200px sopra la prima impronta = c'è lo spazio etichette
        y_bot = min(h, b2[1] + pad)

        # Raffina y_top cercando la prima linea strutturale sopra b1
        lines_above_b1 = [l for l in struct_lines if l < b1[0]]
        if lines_above_b1:
            # La linea strutturale più vicina a b1 (dall'alto)
            closest = max(lines_above_b1)
            y_top = closest

        # Raffina y_bot cercando la prima linea strutturale sotto b2
        lines_below_b2 = [l for l in struct_lines if l > b2[1]]
        if lines_below_b2:
            closest = min(lines_below_b2)
            y_bot = closest

        return (y_top, y_bot, [b1], [b2])

    def _row_range(
        self,
        y_top:        int,
        y_bot:        int,
        row1_blocks:  List,
        row2_blocks:  List,
        struct_lines: List[int],
        dark_pcts:    np.ndarray,
        which:        int,   # 1 = mano destra, 2 = mano sinistra
    ) -> Optional[Tuple[int, int]]:
        """
        Determina i confini verticali di una riga di impronte.

        La riga include l'etichetta sopra (es. "1. Pollice sinistro")
        E l'impronta stessa sotto.

        Per riga 1: da y_top alla linea che divide le due righe
        Per riga 2: dalla linea divisoria a y_bot
        """
        b1 = row1_blocks[0]
        b2 = row2_blocks[0]

        # La linea divisoria = punto medio tra fine di b1 e inizio di b2
        gap_start = b1[1]
        gap_end   = b2[0]
        mid_y     = (gap_start + gap_end) // 2

        # Raffina cercando una linea strutturale nel gap
        lines_in_gap = [l for l in struct_lines if gap_start <= l <= gap_end]
        if lines_in_gap:
            mid_y = min(lines_in_gap, key=lambda l: abs(l - mid_y))

        # Oppure: cerca il punto di minimo dark_pct nel gap
        if gap_end > gap_start and not lines_in_gap:
            gap_profile = dark_pcts[gap_start:gap_end]
            if len(gap_profile) > 0:
                mid_idx = int(np.argmin(gap_profile))
                mid_y   = gap_start + mid_idx

        if which == 1:
            return (y_top, mid_y)
        else:
            return (mid_y, y_bot)

    # ── Divisori di colonna ──────────────────────────────────────────

    def _column_dividers(
        self,
        gray:  np.ndarray,
        y_top: int,
        y_bot: int,
        w:     int,
    ) -> List[int]:
        """
        Trova i 6 punti x che delimitano le 5 colonne.

        Prova a rilevare le linee verticali nell'area delle impronte.
        Se ne trova 4 interne ben spaziate → le usa.
        Altrimenti → divisione uniforme in 5 parti.
        """
        h = gray.shape[0]
        zone = gray[y_top:y_bot, :]
        _, bw = cv2.threshold(zone, 200, 255, cv2.THRESH_BINARY_INV)

        # Kernel verticale proporzionale alla zona
        min_v = max(10, (y_bot - y_top) // 6)
        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v))
        v_det  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kern)
        v_proj = v_det.sum(axis=0).astype(float)

        if v_proj.max() > 0:
            v_proj /= v_proj.max()

        # Cluster linee verticali
        v_thresh = 0.15
        v_cands  = np.where(v_proj > v_thresh)[0]
        v_groups: List[int] = []
        if len(v_cands):
            gs, prev = int(v_cands[0]), int(v_cands[0])
            for c in v_cands[1:]:
                c = int(c)
                if c - prev > 15:
                    v_groups.append((gs + prev) // 2)
                    gs = c
                prev = c
            v_groups.append((gs + prev) // 2)

        # Mantieni solo i divisori interni (non troppo vicini ai bordi)
        internal = [x for x in v_groups if w * 0.05 < x < w * 0.95]
        exp = w / 5

        # Cerca i 4 più vicini alle posizioni attese
        chosen: List[int] = []
        for i in range(1, 5):
            target = int(w * i / 5)
            cands  = [x for x in internal if abs(x - target) < exp * 0.35]
            if cands:
                chosen.append(min(cands, key=lambda x: abs(x - target)))

        if len(chosen) == 4:
            return [0] + sorted(chosen) + [w]

        # Fallback: divisione uniforme
        cw = w // 5
        return [cw * i for i in range(6)]

    # ── Start impronta (salta etichetta) ─────────────────────────────

    def _find_fp_start(self, cell: np.ndarray) -> int:
        """
        Trova dove inizia l'impronta vera dentro la cella (salta l'etichetta).
        L'impronta = contenuto STABILE e DISTRIBUITO (creste su area larga).
        L'etichetta = contenuto CONCENTRATO su poche righe.
        """
        if cell.size == 0 or cell.shape[0] < 40:
            return 0

        ch = cell.shape[0]
        gray = cell if cell.ndim == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        dark = (gray < 100).mean(axis=1)

        # Media mobile per lisciare
        k = max(3, ch // 25)
        smooth = np.convolve(dark, np.ones(k)/k, mode='same')

        # Soglia: 12% del valore massimo della cella
        thresh = max(0.08, smooth.max() * 0.12)

        # Cerca il primo punto con contenuto stabile (almeno k righe)
        for y in range(int(ch * 0.05), int(ch * 0.80)):
            window = smooth[y:min(y+k, ch)]
            if len(window) > 0 and window.min() > thresh:
                return max(0, y - 2)

        return 0

    # ─────────────────────────────────────────────────────────────────
    #  STRATEGIA B: Fallback per-column density
    # ─────────────────────────────────────────────────────────────────

    def _extract_fallback(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Fallback: analisi densità per colonna.
        Divide il card in 5 colonne uniformi e trova 2 blocchi per colonna.
        """
        h, w = gray.shape
        col_w = w // 5
        col_dividers = [col_w * i for i in range(6)]

        right_hand: List[np.ndarray] = []
        left_hand:  List[np.ndarray] = []

        for i in range(5):
            x1, x2 = col_dividers[i], col_dividers[i+1]
            col = gray[:, x1:x2]
            dark = (col < 100).mean(axis=1)
            k = max(3, h // 30)
            smooth = np.convolve(dark, np.ones(k)/k, mode='same')
            is_act = smooth > 0.10

            blocks: List[Tuple[int, int]] = []
            in_b, start, gap = False, 0, 0
            for y, v in enumerate(is_act):
                if v:
                    if not in_b: in_b, start, gap = True, y, 0
                    else: gap = 0
                else:
                    if in_b:
                        gap += 1
                        if gap > 50:
                            in_b = False
                            if y - start - gap >= 80:
                                blocks.append((start, y - gap))
            if in_b and h - start >= 80:
                blocks.append((start, h))

            blocks.sort(key=lambda b: b[1]-b[0], reverse=True)
            top2 = sorted(blocks[:2])

            pad = self.cfg.roi_padding

            def crop(b):
                return gray[max(0, b[0]-pad):min(h, b[1]+pad),
                            max(0, x1-pad):min(w, x2+pad)]

            if len(top2) >= 2:
                right_hand.append(crop(top2[0]))
                left_hand.append(crop(top2[1]))
            elif len(top2) == 1:
                mid = h // 2
                right_hand.append(crop(top2[0]))
                left_hand.append(crop((mid, h)))
            else:
                right_hand.append(gray[h//4:h//2, x1:x2])
                left_hand.append(gray[h//2:h*3//4, x1:x2])

        return right_hand + left_hand

    # ── Utility ──────────────────────────────────────────────────────

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
