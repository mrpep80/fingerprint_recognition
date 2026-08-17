"""
pipeline/engine.py
==================
Orchestratore principale del sistema di confronto impronte.

L'engine assembla la pipeline completa:
    ImageProcessor  → pre-elabora le immagini
    CardSegmenter   → estrae ROI dalla scheda
    BaseExtractor   → estrae feature dal frammento (una volta sola)
    BaseExtractor   → estrae feature da ogni riferimento/ROI
    BaseMatcher     → matcha feature query ↔ reference
    RANSACVerifier  → verifica geometrica
    MatchScorer     → calcola score normalizzato

Per i metodi ensemble, esegue più pipeline in parallelo e combina i punteggi.

Metodi supportati
-----------------
  sift      – SIFT + FLANN + RANSAC omografia
  orb       – ORB  + BFMatcher + RANSAC omografia
  akaze     – AKAZE + BFMatcher + RANSAC omografia
  minutiae  – MinutiaeExtractor + MinutiaeMatcher + RANSAC affine
  ensemble  – SIFT×0.50 + ORB×0.15 + minutiae×0.35
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from ..preprocessing.image_processor import ImageProcessor
from ..preprocessing.card_segmenter  import CardSegmenter
from ..extractors                    import get_extractor
from ..extractors.base               import FeatureSet
from ..matchers                      import get_matcher
from ..matchers.base                 import MatchScore
from ..pipeline.verifier             import RANSACVerifier, VerificationResult
from ..pipeline.scorer               import MatchScorer


# ── Struttura del risultato per ogni scheda ───────────────────────────────────
@dataclass
class MatchResult:
    filename:       str
    score:          float    # [0..100]
    n_inliers:      int
    n_good_matches: int
    n_features:     int      # feature del frammento
    method:         str
    minutiae_source: str = "n/a"  # 'nbis' | 'opencv' | 'n/a'

    def __lt__(self, other):
        return self.score < other.score


# ── Pipeline per un singolo metodo ───────────────────────────────────────────
@dataclass
class _Pipeline:
    extractor: object
    matcher:   object
    weight:    float


class FingerprintEngine:
    """Motore principale di confronto impronte digitali."""

    # Composizione dei metodi ensemble: (method_name, weight)
    # Composizione ensemble aggiornata con enhanced_sift (DFT→HE→Hong-Jain→SIFT)
    # Pesi basati su: SURF EER=2% vs SIFT 10% (Sahu 2013) + test cross-domain
    ENSEMBLE_COMPOSITION = [
        ("enhanced_sift", 0.45),   # cross-domain best: DFT+Hong-Jain+SIFT
        ("sift",          0.25),   # standard SIFT per immagini buone
        ("minutiae",      0.20),   # minuzie per immagini di qualità
        ("orb",           0.10),   # ORB per velocità
    ]

    def __init__(
        self,
        config:      Config,
        method:      str  = "sift",
        use_gabor:   bool = True,
        try_rois:    bool = True,
        n_workers:   int  = 1,
    ):
        self.cfg        = config
        self.method     = method
        self.use_gabor  = use_gabor
        self.try_rois   = try_rois
        self.n_workers  = n_workers

        # Componenti condivisi
        self.preprocessor = ImageProcessor(config)
        self.segmenter    = CardSegmenter(config)
        self.verifier     = RANSACVerifier(config)
        self.scorer       = MatchScorer(config)

        # Pipeline (una o più in caso di ensemble)
        self._pipelines: List[_Pipeline] = self._build_pipelines(method)

        # Cache feature del frammento (pre-computata una volta sola)
        self._frag_features: Dict[str, FeatureSet] = {}

    # ── API pubblica ──────────────────────────────────────────────────

    def prepare_fragment(self, frag_img: np.ndarray,
                         frag_path: Optional[str] = None) -> Dict[str, int]:
        """
        Pre-processa il frammento ed estrae le feature.
        Va chiamato una volta sola prima di run_search().

        Args:
            frag_img  : immagine del frammento
            frag_path : percorso originale (usato per il multiprocessing)

        Returns:
            dizionario { method_name: n_keypoints }
        """
        self._frag_path = frag_path   # salvato per ProcessPoolExecutor

        frag_proc = self.preprocessor.preprocess(frag_img, self.use_gabor)
        frag_gray = self.preprocessor.to_gray(frag_img)
        frag_gray = self.preprocessor.resize_to_working(frag_gray)

        stats = {}
        for pl in self._pipelines:
            inp = frag_gray if pl.extractor.needs_raw_input else frag_proc
            fs  = pl.extractor.extract(inp)
            self._frag_features[pl.extractor.name] = fs
            stats[pl.extractor.name] = fs.n_features
        return stats

    def run_search(
        self,
        ref_folder: Path,
        top_n:      int = 10,
    ) -> List[MatchResult]:
        """
        Confronta il frammento con tutte le schede nella cartella.

        Args:
            ref_folder : Path della cartella con le schede
            top_n      : numero di risultati da restituire ordinati

        Returns:
            lista di MatchResult ordinata per score decrescente
        """
        ref_files = [
            p for p in sorted(ref_folder.iterdir())
            if p.suffix in self.cfg.supported_extensions
        ]
        if not ref_files:
            raise FileNotFoundError(f"Nessuna immagine in {ref_folder}")

        results: List[MatchResult] = []

        if self.n_workers > 1:
            results = self._run_parallel(ref_files)
        else:
            for idx, p in enumerate(ref_files, 1):
                print(f"  [{idx:>3}/{len(ref_files)}] {p.name:<45}", end="\r")
                results.append(self._process_reference(p))

        print()
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n] if top_n else results

    def score_against(self, ref_img: np.ndarray) -> Tuple[float, int, int]:
        """
        Confronta il frammento con una singola immagine.

        Returns:
            (score, n_inliers, n_good_matches)
        """
        return self._compare_card(ref_img)

    # ── Metodi privati ────────────────────────────────────────────────

    def _build_pipelines(self, method: str) -> List[_Pipeline]:
        """Costruisce la lista di pipeline in base al metodo scelto."""
        if method == "ensemble":
            pls = []
            for m, w in self.ENSEMBLE_COMPOSITION:
                pls.append(_Pipeline(
                    extractor=get_extractor(m, self.cfg),
                    matcher=get_matcher(m, self.cfg),
                    weight=w,
                ))
            return pls
        else:
            return [_Pipeline(
                extractor=get_extractor(method, self.cfg),
                matcher=get_matcher(method, self.cfg),
                weight=1.0,
            )]

    def _run_parallel(self, ref_files: List[Path]) -> List[MatchResult]:
        """Esecuzione parallela su più thread."""
        results = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futs = {ex.submit(self._process_reference, p): p for p in ref_files}
            done = 0
            for fut in as_completed(futs):
                done += 1
                print(f"  [{done:>3}/{len(ref_files)}]", end="\r")
                try:
                    results.append(fut.result())
                except Exception as e:
                    p = futs[fut]
                    print(f"\n  [WARN] {p.name}: {e}")
                    results.append(MatchResult(p.name, 0.0, 0, 0, 0, self.method))
        return results

    def _process_reference(self, ref_path: Path) -> MatchResult:
        """Carica una scheda e la confronta con il frammento."""
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            return MatchResult(ref_path.name, 0.0, 0, 0, 0, self.method)

        score, inl, good = self._compare_card(ref_img)
        n_feat   = max(
            (fs.n_features for fs in self._frag_features.values()), default=0
        )
        # Recupera source minuzie se presente
        min_src = "n/a"
        if "minutiae" in self._frag_features:
            min_src = self._frag_features["minutiae"].metadata.get("source", "n/a")

        return MatchResult(
            filename=ref_path.name,
            score=score,
            n_inliers=inl,
            n_good_matches=good,
            n_features=n_feat,
            method=self.method,
            minutiae_source=min_src,
        )

    def _compare_card(
        self, card_img: np.ndarray
    ) -> Tuple[float, int, int]:
        """
        Confronta il frammento con una scheda (intera + ROI se richiesto).
        Restituisce il miglior risultato trovato.
        """
        best = (0.0, 0, 0)

        # 1. Scheda intera
        r = self._compare_image(card_img)
        if r[0] > best[0]:
            best = r

        # 2. Singole ROI
        if self.try_rois:
            for roi in self.segmenter.extract_rois(card_img):
                r = self._compare_image(roi)
                if r[0] > best[0]:
                    best = r

        return best

    def _compare_image(
        self, ref_img: np.ndarray
    ) -> Tuple[float, int, int]:
        """
        Esegue tutte le pipeline sull'immagine e combina i punteggi.
        Rispetta needs_raw_input: passa l'immagine grezza agli estrattori
        che gestiscono internamente la propria pipeline (es. EnhancedSIFT).
        """
        ref_proc = self.preprocessor.preprocess(ref_img, self.use_gabor)
        # Versione grezza per estrattori con pipeline propria
        ref_gray = self.preprocessor.to_gray(ref_img)
        ref_gray = self.preprocessor.resize_to_working(ref_gray)

        total_score = 0.0
        best_inl    = 0
        best_good   = 0

        for pl in self._pipelines:
            frag_fs = self._frag_features.get(pl.extractor.name)
            if frag_fs is None or not frag_fs.is_valid():
                continue

            inp    = ref_gray if pl.extractor.needs_raw_input else ref_proc
            ref_fs = pl.extractor.extract(inp)
            ms      = pl.matcher.match(frag_fs, ref_fs)
            vr      = self.verifier.verify(frag_fs, ref_fs, ms)
            s       = self.scorer.score(vr.n_inliers, ms.n_good,
                                         frag_fs.n_features,
                                         ms.n_features_ref)

            total_score += s * pl.weight
            if vr.n_inliers > best_inl:
                best_inl  = vr.n_inliers
                best_good = ms.n_good

        return total_score, best_inl, best_good
