"""
io/reporter.py – v2.2
======================
Formattazione e salvataggio dei risultati.
Supporta sia MatchResult (singolo metodo) che FusedResult (fusione multi-metodo).
"""

import csv
import os
from typing import List, Union

from ..pipeline.engine import MatchResult
from ..pipeline.fusion import FusedResult


class ResultReporter:
    """Stampa e salva i risultati del confronto."""

    SEP  = "═" * 90
    SEP2 = "─" * 90

    # ── MatchResult (singolo metodo) ──────────────────────────────────

    def print_table(self, results: List[MatchResult], top_n: int = 10) -> None:
        print(f"\n{self.SEP}")
        print(f"  {'RK':<4} {'FILE':<38} {'SCORE':>7}  "
              f"{'INLIERS':>8}  {'MATCHES':>8}  {'MINUZIE':>8}")
        print(self.SEP)
        for i, r in enumerate(results[:top_n], 1):
            bar = "▓" * int(r.score / 5) + "░" * (20 - int(r.score / 5))
            print(f"  {i:<4} {r.filename:<38} {r.score:>6.2f}%  "
                  f"{r.n_inliers:>8}  {r.n_good_matches:>8}  {r.minutiae_source:>8}")
            print(f"       [{bar}]")
        print(self.SEP)

    def print_summary(self, results: List[MatchResult]) -> None:
        if not results:
            print("[NESSUN RISULTATO]"); return
        b = results[0]
        print(f"\n{self.SEP2}")
        print(f"  Miglior match    : {b.filename}")
        print(f"  Score            : {b.score:.2f}%")
        print(f"  Inliers RANSAC   : {b.n_inliers}")
        print(f"  Feature frammento: {b.n_features}")
        print(f"  Metodo           : {b.method.upper()}")
        print(f"{self.SEP2}\n")

    def save_csv(self, results: List[MatchResult], output_dir: str,
                 filename: str = "fingerprint_matches.csv") -> str:
        path = os.path.join(output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank","filename","score_%","inliers",
                         "good_matches","n_features","method","minutiae_source"])
            for i, r in enumerate(results, 1):
                w.writerow([i, r.filename, f"{r.score:.4f}",
                             r.n_inliers, r.n_good_matches,
                             r.n_features, r.method, r.minutiae_source])
        print(f"[CSV] Salvato: {path}")
        return path

    # ── FusedResult (fusione multi-metodo) ───────────────────────────

    def print_fusion_table(self, results: List[FusedResult],
                           top_n: int = 10) -> None:
        """Tabella dettagliata per i risultati della fusione multi-metodo."""
        if not results:
            return

        methods = sorted(results[0].per_method.keys())

        # Header dinamico in base ai metodi disponibili
        method_cols = "  ".join(f"{m.upper():>8}" for m in methods)
        print(f"\n{self.SEP}")
        print(f"  {'RK':<4} {'FILE':<35} {'FUSED':>7}  {'ATTIVI':>6}  {method_cols}")
        print(self.SEP)

        for i, r in enumerate(results[:top_n], 1):
            bar   = "▓" * int(r.combined_score / 5) + "░" * (20 - int(r.combined_score / 5))
            mscores = "  ".join(
                f"{r.per_method.get(m, 0):>7.1f}%" for m in methods
            )
            star  = "★" if r.n_active == len(methods) else " "
            print(f"  {i:<4} {r.filename:<35} {r.combined_score:>6.1f}%{star} "
                  f"{r.n_active:>4}/{len(methods)}  {mscores}")
            print(f"       [{bar}]  (migliore: {r.best_method.upper()})")
        print(self.SEP)
        print(f"  ★ = match confermato da tutti i {len(methods)} metodi")

    def print_fusion_summary(self, results: List[FusedResult]) -> None:
        if not results:
            return
        b = results[0]
        methods = sorted(b.per_method.keys())
        print(f"\n{self.SEP2}")
        print(f"  Miglior match    : {b.filename}")
        print(f"  Score fusione    : {b.combined_score:.2f}%")
        print(f"  Metodi attivi    : {b.n_active}/{len(methods)}")
        for m in methods:
            s   = b.per_method.get(m, 0)
            inl = b.inliers.get(m, 0)
            bar = "▓" * int(s / 5) + "░" * (20 - int(s / 5))
            print(f"    {m.upper():10s}: {s:>6.1f}%  [{bar}]  inliers={inl}")
        print(f"  Metodo vincente  : {b.best_method.upper()}")
        print(f"{self.SEP2}\n")

    def save_fusion_csv(self, results: List[FusedResult], output_dir: str,
                        filename: str = "fingerprint_matches.csv") -> str:
        if not results:
            return ""
        methods = sorted(results[0].per_method.keys())
        path    = os.path.join(output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = (["rank","filename","fused_score","n_active_methods"] +
                      [f"score_{m}" for m in methods] +
                      [f"inliers_{m}" for m in methods] +
                      ["best_method"])
            w.writerow(header)
            for i, r in enumerate(results, 1):
                row = ([i, r.filename, f"{r.combined_score:.4f}", r.n_active] +
                       [f"{r.per_method.get(m,0):.4f}" for m in methods] +
                       [r.inliers.get(m, 0) for m in methods] +
                       [r.best_method])
                w.writerow(row)
        print(f"[CSV] Salvato: {path}")
        return path
