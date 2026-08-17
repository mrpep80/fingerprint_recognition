"""
parallel_worker.py
==================
Funzioni worker a livello di MODULO per il multiprocessing.

Devono essere a livello di modulo (non lambda/nested) per essere picklabili
da multiprocessing. Python non può serializzare funzioni annidate.

Flusso:
  1. ProcessPoolExecutor chiama _init_worker() UNA VOLTA per processo worker
     → carica il frammento e pre-calcola le feature (costoso: fatto 1 volta)
  2. ProcessPoolExecutor chiama _compare_ref() per ogni immagine di riferimento
     → usa le feature pre-calcolate, processa solo il riferimento (veloce)

Vantaggio rispetto a ThreadPoolExecutor:
  - ThreadPoolExecutor: il GIL di Python limita il parallelismo CPU-bound
  - ProcessPoolExecutor: processi separati, nessun GIL, vero parallelismo
  - Su 4 core: ~4x speedup per enhanced_sift (che è CPU-bound)
"""

import cv2
import os
import sys
from pathlib import Path

# Stato locale al processo worker (ogni processo ha la propria copia)
_worker_engine = None
_worker_minutiae_source = "n/a"


def _init_worker(frag_path: str, method: str, use_gabor: bool,
                 try_rois: bool, max_dim: int) -> None:
    """
    Inizializzatore del worker: chiamato UNA SOLA VOLTA per processo.

    Pre-calcola le feature del frammento in modo che non vengano
    ricalcolate per ogni immagine di riferimento.

    Args:
        frag_path : percorso del frammento (stringa, picklable)
        method    : metodo di matching
        use_gabor : usa Gabor preprocessing
        try_rois  : estrai ROI dalle schede
        max_dim   : risoluzione massima di lavoro
    """
    global _worker_engine, _worker_minutiae_source

    # Setup path Python nel processo figlio
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        from fingerprint_matcher.config import Config
        from fingerprint_matcher.pipeline.engine import FingerprintEngine

        cfg = Config()
        cfg.max_working_dim = max_dim

        frag_img = cv2.imread(frag_path)
        if frag_img is None:
            return

        _worker_engine = FingerprintEngine(
            config=cfg,
            method=method,
            use_gabor=use_gabor,
            try_rois=try_rois,
            n_workers=1,
        )
        _worker_engine.prepare_fragment(frag_img)

        # Recupera sorgente minuzie per il report
        if "minutiae" in _worker_engine._frag_features:
            fs = _worker_engine._frag_features["minutiae"]
            _worker_minutiae_source = fs.metadata.get("source", "opencv")

    except Exception as e:
        # In caso di errore, il worker restituirà score 0 per tutte le immagini
        _worker_engine = None


def _compare_ref(ref_path: str) -> tuple:
    """
    Confronta il frammento (pre-calcolato) con un'immagine di riferimento.

    Chiamata molte volte per processo, una per immagine di riferimento.
    Le feature del frammento sono già in _worker_engine._frag_features.

    Args:
        ref_path : percorso dell'immagine di riferimento

    Returns:
        (filename, score, n_inliers, n_good_matches, minutiae_source)
    """
    global _worker_engine, _worker_minutiae_source

    fname = Path(ref_path).name

    if _worker_engine is None:
        return (fname, 0.0, 0, 0, "n/a")

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        return (fname, 0.0, 0, 0, "n/a")

    try:
        score, inl, good = _worker_engine._compare_card(ref_img)
    except Exception:
        return (fname, 0.0, 0, 0, "n/a")

    return (fname, score, inl, good, _worker_minutiae_source)
