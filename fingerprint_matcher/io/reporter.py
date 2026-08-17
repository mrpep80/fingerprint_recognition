import csv
import os
from typing import List
from ..pipeline.engine import MatchResult
from ..pipeline.fusion import FusedResult

class ResultReporter:
    SEP = "═" * 110
    SEP2 = "─" * 110

    def print_table(self, results: List[MatchResult], top_n: int = 10) -> None:
        print(f"\n{self.SEP}")
        print(f"  {'RK':<4} {'FILE':<35} {'SCORE':>8} {'INLIERS':>8} {'MATCH':>7} {'DITO':<18}")
        print(self.SEP)
        for i, r in enumerate(results[:top_n], 1):
            print(f"  {i:<4} {r.filename:<35} {r.score:>7.2f}% {r.n_inliers:>8} {r.n_good_matches:>7} {r.matched_finger:<18}")
        print(self.SEP)

    def print_summary(self, results: List[MatchResult]) -> None:
        if not results:
            print("[NESSUN RISULTATO]"); return
        b = results[0]
        print(f"\n{self.SEP2}\n  Miglior match    : {b.filename}\n  Score            : {b.score:.2f}%\n  Inliers RANSAC   : {b.n_inliers}\n  Dito/ROI         : {b.matched_finger}\n  Metodo           : {b.method.upper()}\n{self.SEP2}\n")

    def save_csv(self, results: List[MatchResult], output_dir: str, filename: str = "fingerprint_matches.csv") -> str:
        path = os.path.join(output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank","filename","score_%","inliers","good_matches","n_features","method","minutiae_source","matched_finger"])
            for i, r in enumerate(results, 1):
                w.writerow([i,r.filename,f"{r.score:.4f}",r.n_inliers,r.n_good_matches,r.n_features,r.method,r.minutiae_source,r.matched_finger])
        print(f"[CSV] Salvato: {path}")
        return path

    def print_fusion_table(self, results: List[FusedResult], top_n: int = 10) -> None:
        if not results: return
        methods = sorted(results[0].per_method)
        cols = "  ".join(f"{m.upper():>10}" for m in methods)
        print(f"\n{self.SEP}\n  {'RK':<4} {'FILE':<30} {'FUSED':>8} {'ATTIVI':>8} {'DITO':<18} {cols}\n{self.SEP}")
        for i, r in enumerate(results[:top_n], 1):
            scores = "  ".join(f"{r.per_method.get(m,0):>9.1f}%" for m in methods)
            print(f"  {i:<4} {r.filename:<30} {r.combined_score:>7.2f}% {r.n_active:>3}/{len(methods):<3} {r.matched_finger:<18} {scores}")
        print(self.SEP)

    def print_fusion_summary(self, results: List[FusedResult]) -> None:
        if not results: return
        b = results[0]; methods = sorted(b.per_method)
        print(f"\n{self.SEP2}\n  Miglior match    : {b.filename}\n  Score fusione    : {b.combined_score:.2f}%\n  Consenso         : {b.n_active}/{len(methods)} metodi\n  Dito migliore    : {b.matched_finger}")
        for m in methods:
            print(f"    {m.upper():12s}: {b.per_method.get(m,0):6.2f}%  rank={b.ranks.get(m,'-')}  inliers={b.inliers.get(m,0)}")
        print(f"{self.SEP2}\n")

    def save_fusion_csv(self, results: List[FusedResult], output_dir: str, filename: str = "fingerprint_matches.csv") -> str:
        if not results: return ""
        methods = sorted(results[0].per_method); path = os.path.join(output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank","filename","fused_score","n_active_methods","consensus","matched_finger"] + [f"score_{m}" for m in methods] + [f"rank_{m}" for m in methods] + [f"inliers_{m}" for m in methods] + ["best_method"])
            for i, r in enumerate(results, 1):
                w.writerow([i,r.filename,f"{r.combined_score:.4f}",r.n_active,f"{r.consensus:.4f}",r.matched_finger] + [f"{r.per_method.get(m,0):.4f}" for m in methods] + [r.ranks.get(m,0) for m in methods] + [r.inliers.get(m,0) for m in methods] + [r.best_method])
        print(f"[CSV] Salvato: {path}")
        return path
