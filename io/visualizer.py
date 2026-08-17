"""
io/visualizer.py
================
Genera un'immagine di visualizzazione che affianca frammento e riferimento
con i match disegnati.

Supporta due modalità:
  - Descrittori (SIFT/ORB/AKAZE) : usa cv2.drawMatches con DMatch objects
  - Minuzie                       : disegna linee tra le coppie (q_idx, r_idx)
                                    e sovrappone i punti delle minuzie

L'immagine viene salvata come JPG nella cartella di output.
"""

from __future__ import annotations
import os
from typing import List, Tuple, Union

import cv2
import numpy as np

from ..config import Config
from ..extractors.base import FeatureSet
from ..matchers.base   import MatchScore
from ..pipeline.verifier import VerificationResult
from ..preprocessing.image_processor import ImageProcessor


class MatchVisualizer:
    """Genera immagini di visualizzazione per i match trovati."""

    MATCH_COLOR    = (0, 220, 100)   # verde per gli inlier
    POINT_COLOR    = (0, 140, 255)   # arancione per le minuzie
    TEXT_COLOR     = (0, 210, 255)   # giallo per il testo
    SINGLE_COLOR   = (100, 100, 100) # grigio per punti senza match

    def __init__(self, config: Config):
        self.cfg  = config
        self.prep = ImageProcessor(config)

    def save_match_image(
        self,
        frag_img:    np.ndarray,
        ref_img:     np.ndarray,
        frag_feat:   FeatureSet,
        ref_feat:    FeatureSet,
        verif:       VerificationResult,
        match_score: MatchScore,
        label:       str,
        out_path:    str,
        use_gabor:   bool = True,
    ) -> None:
        """
        Salva un'immagine affiancata con i match evidenziati.

        Args:
            frag_img    : immagine originale del frammento
            ref_img     : immagine originale del riferimento
            frag_feat   : FeatureSet del frammento
            ref_feat    : FeatureSet del riferimento
            verif       : VerificationResult con gli inlier
            match_score : MatchScore con i raw matches
            label       : testo informativo da sovrapporre
            out_path    : percorso del file di output
            use_gabor   : se preprocessare con Gabor per la visualizzazione
        """
        fp = self.prep.preprocess(frag_img, use_gabor)
        rp = self.prep.preprocess(ref_img,  use_gabor)

        # Scegli la modalità in base al tipo di feature
        if frag_feat.descriptors is not None:
            vis = self._draw_descriptor_matches(
                fp, frag_feat, rp, ref_feat, verif.inlier_matches
            )
        else:
            vis = self._draw_minutiae_matches(
                fp, frag_feat, rp, ref_feat, verif.inlier_matches
            )

        # Sovrascrivi l'etichetta
        cv2.putText(vis, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.TEXT_COLOR, 2)

        cv2.imwrite(out_path, vis)
        print(f"[VIS] Salvata: {out_path}")

    # ── Descriptor matching (SIFT/ORB/AKAZE) ─────────────────────────

    def _draw_descriptor_matches(
        self,
        fp: np.ndarray, frag_feat: FeatureSet,
        rp: np.ndarray, ref_feat:  FeatureSet,
        inlier_matches: list,
    ) -> np.ndarray:
        """Usa cv2.drawMatches per i DMatch object."""
        vis = cv2.drawMatches(
            fp, frag_feat.keypoints,
            rp, ref_feat.keypoints,
            inlier_matches[:60], None,
            matchColor=self.MATCH_COLOR,
            singlePointColor=self.SINGLE_COLOR,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return vis

    # ── Minutiae matching ─────────────────────────────────────────────

    def _draw_minutiae_matches(
        self,
        fp: np.ndarray, frag_feat: FeatureSet,
        rp: np.ndarray, ref_feat:  FeatureSet,
        inlier_pairs: list,  # list of (q_idx, r_idx) tuples
    ) -> np.ndarray:
        """
        Disegna i punti delle minuzie e le linee di match.
        Le immagini sono affiancate orizzontalmente.
        """
        h1, w1 = fp.shape[:2]
        h2, w2 = rp.shape[:2]
        h_out  = max(h1, h2)

        # Canvas affiancato in BGR
        canvas = np.zeros((h_out, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1]     = cv2.cvtColor(fp, cv2.COLOR_GRAY2BGR)
        canvas[:h2, w1:w1+w2] = cv2.cvtColor(rp, cv2.COLOR_GRAY2BGR)

        kp_q = frag_feat.keypoints or []
        kp_r = ref_feat.keypoints  or []

        # Disegna tutti i keypoints
        for kp in kp_q:
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(canvas, (cx, cy), 3, self.SINGLE_COLOR, -1)

        for kp in kp_r:
            cx, cy = int(kp.pt[0] + w1), int(kp.pt[1])
            cv2.circle(canvas, (cx, cy), 3, self.SINGLE_COLOR, -1)

        # Disegna le linee di match per gli inlier
        for pair in inlier_pairs[:60]:
            if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                continue
            qi, ri = int(pair[0]), int(pair[1])
            if qi >= len(kp_q) or ri >= len(kp_r):
                continue
            pt1 = (int(kp_q[qi].pt[0]),       int(kp_q[qi].pt[1]))
            pt2 = (int(kp_r[ri].pt[0]) + w1,  int(kp_r[ri].pt[1]))
            cv2.line(canvas, pt1, pt2, self.MATCH_COLOR, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 4, self.POINT_COLOR, -1)
            cv2.circle(canvas, pt2, 4, self.POINT_COLOR, -1)

        return canvas
