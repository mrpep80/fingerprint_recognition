#!/usr/bin/env python3
"""Fingerprint Fragment Matcher.

Fusion consigliata: MINUTIAE custom + NBIS/BOZORTH3 + SIFT + ORB.
Enhanced SIFT resta disponibile come metodo singolo ma non entra nella fusion
principale, perché sui test correnti tende a produrre falsi positivi.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

#sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fingerprint_matcher.config import Config
from fingerprint_matcher.pipeline.engine import FingerprintEngine
from fingerprint_matcher.pipeline.fusion import ScoreFusion
from fingerprint_matcher.io.loader import ImageLoader
from fingerprint_matcher.io.reporter import ResultReporter


FUSION_BASE_WEIGHTS = {
    "minutiae": 1.8,
    "nbis": 2.2,
    "sift": 1.0,
    "orb": 0.6,
}
SINGLE_METHODS = ["sift", "orb", "akaze", "minutiae", "enhanced_sift", "nbis"]


def nbis_available() -> bool:
    mindtct = os.environ.get("NBIS_MINDTCT") or shutil.which("mindtct") or shutil.which("mindtct.exe")
    bozorth = os.environ.get("NBIS_BOZORTH3") or shutil.which("bozorth3") or shutil.which("bozorth3.exe")
    return bool(mindtct and bozorth)


def build_weights() -> dict:
    w = dict(FUSION_BASE_WEIGHTS)
    if not nbis_available():
        w["nbis"] = 0.0
    return w


def parse_args():
    ap = argparse.ArgumentParser(description="Fingerprint Fragment Matcher")
    ap.add_argument("fragment")
    ap.add_argument("folder")
    ap.add_argument("--method", choices=["fusion"] + SINGLE_METHODS, default="fusion")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-gabor", dest="gabor", action="store_false", default=True)
    ap.add_argument("--no-regions", dest="regions", action="store_false", default=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-dim", dest="max_dim", type=int, default=1500)
    ap.add_argument("--weights", default=None,
                    help="es. minutiae:1.5,nbis:2,sift:1,orb:0.5")
    return ap.parse_args()


def parse_weights(value: str | None, defaults: dict) -> dict:
    if not value:
        return defaults
    out = dict(defaults)
    for item in value.split(","):
        name, weight = item.split(":", 1)
        out[name.strip()] = float(weight)
    return out


def banner(args, weights):
    print("\n" + "═" * 78)
    print("  FINGERPRINT FRAGMENT MATCHER  v5")
    print(f"  Metodo  : {args.method.upper()}")
    print(f"  Gabor   : {'sì' if args.gabor else 'no'}")
    print(f"  ROI     : {'sì' if args.regions else 'no'}")
    print(f"  max-dim : {args.max_dim}px")
    print(f"  Workers : {args.workers}")
    if args.method == "fusion":
        active = [f"{m}(×{w:g})" for m, w in weights.items() if w > 0]
        print("  Fusion  : " + ", ".join(active))
        print(f"  NBIS    : {'disponibile' if weights.get('nbis', 0) > 0 else 'non disponibile'}")
    print("═" * 78)


def run_one_method(method, frag_img, fragment_path, ref_folder, args, cfg):
    engine = FingerprintEngine(
        config=cfg, method=method, use_gabor=args.gabor,
        try_rois=args.regions, n_workers=1,
    )
    stats = engine.prepare_fragment(frag_img, frag_path=fragment_path)
    n = stats.get(method, 0)
    print(f"    {method.upper():10s}: {n} feature/minuzie estratte")
    results = engine.run_search(ref_folder, top_n=0)
    return method, results


def run_fusion(frag_img, fragment_path, ref_folder, args, cfg, reporter, weights):
    methods = [m for m, w in weights.items() if w > 0]
    print(f"\n  Metodi attivi ({len(methods)}): " + ", ".join(f"{m}(×{weights[m]:g})" for m in methods))
    print(f"  Schede da confrontare: {len([p for p in ref_folder.iterdir() if p.suffix in cfg.supported_extensions])}")

    all_results = {}
    t0 = time.time()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(methods))) as pool:
            futures = [pool.submit(run_one_method, m, frag_img, fragment_path, ref_folder, args, cfg) for m in methods]
            for fut in as_completed(futures):
                m, results = fut.result()
                all_results[m] = results
    else:
        for i, m in enumerate(methods, 1):
            print(f"\n  [{i}/{len(methods)}] Esecuzione metodo {m.upper()}...")
            name, results = run_one_method(m, frag_img, fragment_path, ref_folder, args, cfg)
            all_results[name] = results

    print(f"\n  Tutti i metodi completati in {time.time() - t0:.1f}s")
    fused = ScoreFusion(support_threshold=5.0).fuse(all_results, weights=weights)
    reporter.print_fusion_table(fused, args.top)
    reporter.save_fusion_csv(fused, args.output if hasattr(args, "output") else ".")
    reporter.print_fusion_summary(fused)
    return fused


def main():
    args = parse_args()
    args.output = "."
    weights = parse_weights(args.weights, build_weights())
    banner(args, weights)

    if args.method == "nbis" and not nbis_available():
        sys.exit("[ERRORE] NBIS non disponibile: installare mindtct e bozorth3 oppure impostare NBIS_MINDTCT/NBIS_BOZORTH3.")

    cfg = Config()
    cfg.max_working_dim = args.max_dim
    loader = ImageLoader(cfg)
    reporter = ResultReporter()

    print(f"\n[1/4] Caricamento frammento: {args.fragment}")
    frag_img = loader.load_fragment(args.fragment)
    print(f"      Dimensioni: {frag_img.shape[1]}×{frag_img.shape[0]} px")

    ref_folder = Path(args.folder)
    if not ref_folder.exists():
        sys.exit(f"[ERRORE] Cartella non trovata: {ref_folder}")

    print("\n[2/4] Preparazione matching...")
    if args.method == "fusion":
        fused = run_fusion(frag_img, str(Path(args.fragment).resolve()), ref_folder,
                           args, cfg, reporter, weights)
        if fused:
            b = fused[0]
            print(f"\n  Miglior match : {b.filename}")
            print(f"  Score fusion : {b.combined_score:.2f}%")
            print(f"  Metodi attivi: {b.n_active}/{len(b.per_method)}")
            print(f"  Metodo forte : {b.best_method.upper()}")
    else:
        method, results = run_one_method(args.method, frag_img,
                                         str(Path(args.fragment).resolve()),
                                         ref_folder, args, cfg)
        print(f"\n[CSV] Risultati metodo {method.upper()}")
        reporter.print_table(results, args.top)
        reporter.save_csv(results, ".")
        reporter.print_summary(results)


if __name__ == "__main__":
    main()
