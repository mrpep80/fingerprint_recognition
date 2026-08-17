#!/usr/bin/env python3
"""
main.py  –  Fingerprint Fragment Matcher v2.2
==============================================
Confronta un frammento con una cartella di schede.

Utilizzo
--------
    python main.py <frammento> <cartella> [opzioni]

Metodi disponibili
------------------
    sift      – SIFT + FLANN + RANSAC (robusto, lento)
    orb       – ORB  + BFMatcher (veloce)
    akaze     – AKAZE (buono su immagini rumorose)
    minutiae      – Minuzie + RANSAC affine (forense, richiede immagini nitide)
    enhanced_sift – DFT→HE→Hong-Jain→SIFT (OTTIMO per latente↔inchiostro)
    fusion    – TUTTI i metodi in parallelo → CombMNZ (CONSIGLIATO)

Il metodo 'fusion' esegue tutti i metodi e combina i punteggi con
l'algoritmo CombMNZ (standard in biometria multi-modale). Non richiede
di sapere a priori quale metodo funziona meglio sull'immagine in input.

Esempi
------
    # Consigliato: lascia decidere al sistema
    python main.py frammento.jpg ./schede/ --method fusion --top 10

    # Veloce (solo SIFT)
    python main.py frammento.jpg ./schede/ --method sift --top 10

    # Con visualizzazione per il miglior match
    python main.py frammento.jpg ./schede/ --method fusion --top 10 \\
        --visualize --output ./risultati/
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_matcher.config          import Config
from fingerprint_matcher.pipeline.engine import FingerprintEngine, MatchResult
from fingerprint_matcher.pipeline.fusion import ScoreFusion, FusedResult
from fingerprint_matcher.io.loader       import ImageLoader
from fingerprint_matcher.io.reporter     import ResultReporter
from fingerprint_matcher.io.visualizer   import MatchVisualizer
from fingerprint_matcher.extractors      import get_extractor
from fingerprint_matcher.matchers        import get_matcher
from fingerprint_matcher.pipeline.verifier import RANSACVerifier

# Tutti i metodi supportabili (usati per --method singolo)
ALL_SINGLE_METHODS = ["sift", "orb", "akaze", "minutiae", "enhanced_sift"]

# Metodi attivi in modalità fusion: solo quelli con peso > 0
# Modifica FUSION_WEIGHTS per abilitare/disabilitare metodi
SINGLE_METHODS = ALL_SINGLE_METHODS  # sovrascritta dinamicamente in run_fusion
ALL_METHODS    = ["fusion"] + SINGLE_METHODS

# Pesi di default per CombMNZ
# Basati su letteratura e test empirici:
#   SIFT     → più robusto, descrittori ad alta dimensionalità
#   MINUTIAE → più forense ma dipende dalla qualità dell'immagine
#   ORB/AKAZE→ complementari, veloci
# Pesi fusion base (validi indipendentemente dalla build OpenCV)
# enhanced_sift implementa DFT(Bhowmik2012)→HE→HongJain(1998)→SIFT
# NOTA SUI PESI:
# enhanced_sift (DFT→HE→Hong-Jain→SIFT su mappe binarie) migliora il
# cross-domain matching MA produce FALSI POSITIVI perché le mappe binarie
# di creste di dita diverse hanno struttura locale simile (linee parallele),
# e SIFT non discrimina abbastanza.
# → Peso 0: disabilitato nella fusion (usabile come metodo singolo)
#
# minutiae è il più affidabile per il matching forense:
# le posizioni geometriche delle minuzie sono univoche per ogni dito.
# → Peso 2.0: metodo primario
#
# sift e orb contribuiscono come verifica complementare
_BASE_WEIGHTS = {
    "enhanced_sift": 0.0,   # DISABILITATO in fusion: falsi positivi su binary maps
    "sift":          1.2,   # standard SIFT, complementare
    "minutiae":      2.0,   # PRIMARIO: più discriminativo, meno falsi positivi
    "orb":           0.8,   # veloce, complementare
    "akaze":         0.9,   # buono su immagini rumorose (se disponibile)
}

def _build_fusion_weights() -> dict:
    """
    Controlla la disponibilità dei metodi con un check LEGGERO:
    non istanzia estrattori, usa solo attribute check e try-import.
    Questo evita la lunga pausa iniziale dovuta all'import di scipy
    tramite fingerprint_enhancer.
    """
    import cv2
    weights = dict(_BASE_WEIGHTS)

    # AKAZE: basta controllare se l'attributo esiste in cv2
    if not hasattr(cv2, "AKAZE_create"):
        weights["akaze"] = 0.0

    # enhanced_sift: controlla se fingerprint_enhancer è installato
    try:
        import fingerprint_enhancer  # noqa: F401
    except ImportError:
        weights["enhanced_sift"] = 0.0

    return weights

# NOTA: FUSION_WEIGHTS viene calcolato DENTRO main() dopo il banner,
# non qui a livello di modulo, per non bloccare l'avvio del programma.
FUSION_WEIGHTS: dict = {}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fingerprint Fragment Matcher v2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("fragment")
    ap.add_argument("folder")
    ap.add_argument("--method", choices=ALL_METHODS, default="fusion",
                    help="Metodo di matching (default: fusion = tutti i metodi)")
    ap.add_argument("--top",     type=int,   default=10)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--output",  default=".")
    ap.add_argument("--no-gabor",   dest="gabor",   action="store_false", default=True)
    ap.add_argument("--no-regions", dest="regions", action="store_false", default=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-dim", dest="max_dim", type=int, default=1500)
    ap.add_argument("--weights", type=str, default=None,
                    help="Pesi per la fusione es. 'sift:2,minutiae:1.5,orb:0.5,akaze:0.5'")
    return ap.parse_args()


def parse_weights(weights_str: str) -> dict:
    """Parsa la stringa dei pesi es. 'sift:2,minutiae:1.5'"""
    if not weights_str:
        return FUSION_WEIGHTS
    w = {}
    for part in weights_str.split(","):
        k, v = part.strip().split(":")
        w[k.strip()] = float(v.strip())
    return w


# ── Banner ───────────────────────────────────────────────────────────

def print_banner(args: argparse.Namespace) -> None:
    method_label = (
        "FUSION (multi-metodo, calcolo pesi in corso...)"
        if args.method == "fusion" else args.method.upper()
    )
    print("\n" + "═" * 70)
    print("  FINGERPRINT FRAGMENT MATCHER  v2.2")
    print(f"  Metodo  : {method_label}")
    print(f"  Gabor   : {'sì' if args.gabor else 'no'}")
    print(f"  ROI     : {'sì' if args.regions else 'no'}")
    print(f"  max-dim : {args.max_dim}px")
    print(f"  Workers : {args.workers}")
    print("═" * 70)


# ── Fusion mode ──────────────────────────────────────────────────────

def run_fusion(frag_img, ref_folder: Path, args, cfg: Config,
               reporter: ResultReporter) -> list:
    """
    Esegue tutti i metodi con peso > 0 in parallelo e li fonde con CombMNZ.
    I metodi con peso = 0 in FUSION_WEIGHTS vengono saltati automaticamente.
    """
    weights = parse_weights(args.weights)

    # Filtra: esegui solo i metodi con peso > 0
    active_methods = [m for m in ALL_SINGLE_METHODS if weights.get(m, 0) > 0]

    print(f"\n  Metodi attivi ({len(active_methods)}): " +
          ", ".join(f"{m} (×{weights[m]})" for m in active_methods))
    print()

    all_results: dict = {}   # { method: [MatchResult] }
    ref_files = sorted(
        p for p in ref_folder.iterdir()
        if p.suffix in cfg.supported_extensions
    )
    if not ref_files:
        sys.exit(f"[ERRORE] Nessuna immagine in: {ref_folder}")

    print(f"  {len(ref_files)} schede da confrontare con {len(active_methods)} metodi attivi")

    def run_one_method(method: str):
        engine = FingerprintEngine(
            config=cfg, method=method,
            use_gabor=args.gabor, try_rois=args.regions,
            n_workers=1,
        )
        stats = engine.prepare_fragment(frag_img, frag_path=str(Path(args.fragment).resolve()))
        n_feat = list(stats.values())[0]
        print(f"    {method.upper():10s}: {n_feat} feature estratte")
        results = engine.run_search(ref_folder, top_n=0)  # top_n=0 → tutti
        return method, results

    t0 = time.time()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(active_methods))) as ex:
            futs = {ex.submit(run_one_method, m): m for m in active_methods}
            for fut in as_completed(futs):
                method, results = fut.result()
                all_results[method] = results
    else:
        for idx, method in enumerate(active_methods, 1):
            print(f"\n  [{idx}/{len(active_methods)}]"
                  f" Esecuzione metodo {method.upper()}...")
            method, results = run_one_method(method)
            all_results[method] = results

    elapsed = time.time() - t0
    print(f"\n  Tutti i metodi completati in {elapsed:.1f}s")

    # Fusione CombMNZ
    fusion  = ScoreFusion(threshold=3.0)
    fused   = fusion.fuse(all_results, weights=weights)

    reporter.print_fusion_table(fused, args.top)
    reporter.save_fusion_csv(fused, args.output)
    reporter.print_fusion_summary(fused)

    return fused, all_results


# ── Single method mode ───────────────────────────────────────────────

def run_single(frag_img, ref_folder: Path, args, cfg: Config,
               reporter: ResultReporter) -> list:
    engine = FingerprintEngine(
        config=cfg, method=args.method,
        use_gabor=args.gabor, try_rois=args.regions,
        n_workers=args.workers,
    )
    stats = engine.prepare_fragment(frag_img)
    for m, n in stats.items():
        print(f"      {m.upper():10s}: {n} feature estratte")

    ref_files = list(ref_folder.iterdir())
    print(f"\n[3/4] {len([p for p in ref_files if p.suffix in cfg.supported_extensions])} "
          f"schede in: {ref_folder}")
    print(f"\n[4/4] Confronto in corso...")

    t0      = time.time()
    results = engine.run_search(ref_folder, top_n=args.top)
    elapsed = time.time() - t0
    print(f"  Completato in {elapsed:.1f}s")

    reporter.print_table(results, args.top)
    reporter.save_csv(results, args.output)
    reporter.print_summary(results)

    return results, {args.method: results}


# ── Visualizzazione ──────────────────────────────────────────────────

def run_visualizer(best_filename: str, best_method: str,
                   frag_img, ref_folder: Path,
                   output_dir: str, cfg: Config, use_gabor: bool):
    ref_path = ref_folder / best_filename
    ref_img  = cv2.imread(str(ref_path))
    if ref_img is None:
        print(f"[WARN] Impossibile caricare {ref_path}")
        return

    from fingerprint_matcher.preprocessing.image_processor import ImageProcessor
    prep = ImageProcessor(cfg)
    fp   = prep.preprocess(frag_img, use_gabor)
    rp   = prep.preprocess(ref_img,  use_gabor)

    m   = best_method if best_method in SINGLE_METHODS else "sift"
    ext = get_extractor(m, cfg)
    mtc = get_matcher(m, cfg)
    ver = RANSACVerifier(cfg)

    fs_f = ext.extract(fp);  fs_r = ext.extract(rp)
    ms   = mtc.match(fs_f, fs_r)
    vr   = ver.verify(fs_f, fs_r, ms)

    label    = f"Score: {vr.n_inliers} inliers | {m.upper()} | {best_filename}"
    vis_path = os.path.join(output_dir, f"vis_{best_filename}")

    MatchVisualizer(cfg).save_match_image(
        frag_img, ref_img, fs_f, fs_r, vr, ms,
        label, vis_path, use_gabor=use_gabor,
    )


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    print_banner(args)

    # Calcola i pesi DOPO il banner (può richiedere qualche secondo per
    # il check di fingerprint_enhancer/scipy al primo avvio)
    global FUSION_WEIGHTS
    FUSION_WEIGHTS = _build_fusion_weights()
    if args.method == "fusion":
        active = [(m, w) for m, w in FUSION_WEIGHTS.items() if w > 0]
        print(f"  Metodi attivi  : " + 
              "  ".join(f"{m}(×{w:.1f})" for m, w in active))
        disabled = [m for m, w in FUSION_WEIGHTS.items() if w == 0]
        if disabled:
            print(f"  Disabilitati   : {', '.join(disabled)} (non disponibili)")
        print()

    cfg              = Config()
    cfg.max_working_dim = args.max_dim
    loader   = ImageLoader(cfg)
    reporter = ResultReporter()

    # 1. Carica frammento
    print(f"\n[1/4] Caricamento frammento: {args.fragment}")
    try:
        frag_img = loader.load_fragment(args.fragment)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERRORE] {e}")
    print(f"      Dimensioni: {frag_img.shape[1]}×{frag_img.shape[0]} px")

    ref_folder = Path(args.folder)
    if not ref_folder.exists():
        sys.exit(f"[ERRORE] Cartella non trovata: {args.folder}")

    # 2. Esegui
    print(f"\n[2/4] Estrazione feature dal frammento...")
    if args.method == "fusion":
        active = [m for m in ALL_SINGLE_METHODS if FUSION_WEIGHTS.get(m, 0) > 0]
        print(f"\n[3/4] Avvio fusione ({len(active)} metodi attivi, "
              f"{len(ALL_SINGLE_METHODS)-len(active)} disabilitati)...")
        fused, all_results = run_fusion(frag_img, ref_folder, args, cfg, reporter)
        best_filename = fused[0].filename if fused else None
        best_method   = fused[0].best_method if fused else "sift"
    else:
        results, all_results = run_single(frag_img, ref_folder, args, cfg, reporter)
        best_filename = results[0].filename if results else None
        best_method   = args.method

    # 3. Visualizzazione
    if args.visualize and best_filename:
        run_visualizer(best_filename, best_method,
                       frag_img, ref_folder,
                       args.output, cfg, args.gabor)


if __name__ == "__main__":
    main()
