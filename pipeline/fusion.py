"""
pipeline/fusion.py
==================
Fusione dei risultati di più metodi di matching (score-level fusion).

Algoritmo: CombMNZ normalizzato
--------------------------------
CombMNZ è uno standard in biometria multi-modale (Ho et al., 1994).

  1. Per ogni metodo M:
       score_norm[i] = (score[i] - min_M) / (max_M - min_M + ε)
  2. Per ogni immagine i:
       combMNZ[i] = Σ(score_norm[i][m]) × n_metodi_attivi[i]

Il fattore n_metodi_attivi premia le immagini identificate da più metodi,
rendendo il risultato più robusto rispetto a qualsiasi metodo singolo.

Vantaggi rispetto all'ensemble weighted-average:
  - Non richiede pesi pre-impostati
  - Robusto alle scale diverse tra i metodi
  - Trasparente: mostra il contributo di ogni metodo
  - Automaticamente adattivo alla qualità del database
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FusedResult:
    """Risultato della fusione multi-metodo per una singola scheda."""
    filename:        str
    combined_score:  float              # CombMNZ normalizzato [0..100]
    per_method:      Dict[str, float]   # score di ogni metodo [0..100]
    best_method:     str                # metodo con score più alto
    n_active:        int                # quanti metodi hanno trovato un match
    inliers:         Dict[str, int]     # inliers per metodo

    def __lt__(self, other):
        return self.combined_score < other.combined_score


class ScoreFusion:
    """
    Fonde i risultati di N metodi di matching in un ranking unico.

    Uso:
        fusion = ScoreFusion(threshold=5.0)
        fused  = fusion.fuse(results_by_method)
    """

    def __init__(self, threshold: float = 5.0):
        """
        Args:
            threshold: score minimo (0-100) perché un metodo
                       sia considerato "attivo" per un'immagine.
                       Sotto questa soglia il metodo non contribuisce al CombMNZ.
        """
        self.threshold = threshold

    def fuse(
        self,
        results_by_method: Dict[str, List],   # { method_name: [MatchResult] }
        weights: Optional[Dict[str, float]] = None,
    ) -> List[FusedResult]:
        """
        Esegue la fusione CombMNZ e restituisce la lista ordinata.

        Args:
            results_by_method : dizionario { nome_metodo: lista MatchResult }
            weights           : pesi opzionali { nome_metodo: peso }.
                                Se None, tutti i metodi hanno peso uguale.
        """
        if not results_by_method:
            return []

        methods   = list(results_by_method.keys())
        n_methods = len(methods)

        # Pesi di default: uniformi
        if weights is None:
            weights = {m: 1.0 for m in methods}
        # Normalizza i pesi in modo che sommino a n_methods
        total_w = sum(weights.values())
        weights = {m: w * n_methods / total_w for m, w in weights.items()}

        # Indice filename → risultati per metodo
        scores_raw:   Dict[str, Dict[str, float]] = {}   # fname → {method → score}
        inliers_raw:  Dict[str, Dict[str, int]]   = {}   # fname → {method → inliers}

        for method, results in results_by_method.items():
            for r in results:
                if r.filename not in scores_raw:
                    scores_raw[r.filename]  = {}
                    inliers_raw[r.filename] = {}
                scores_raw[r.filename][method]  = r.score
                inliers_raw[r.filename][method] = r.n_inliers

        all_files = list(scores_raw.keys())

        # Min-max normalization per metodo
        method_scores_all: Dict[str, List[float]] = {
            m: [scores_raw[f].get(m, 0.0) for f in all_files]
            for m in methods
        }
        norms: Dict[str, Dict[str, float]] = {}
        for m in methods:
            vals   = method_scores_all[m]
            lo, hi = min(vals), max(vals)
            spread = hi - lo if hi > lo else 1.0
            norms[m] = {
                f: (scores_raw[f].get(m, 0.0) - lo) / spread
                for f in all_files
            }

        # CombMNZ per ogni file
        fused: List[FusedResult] = []
        for fname in all_files:
            per_method = {m: scores_raw[fname].get(m, 0.0) for m in methods}
            n_active   = sum(
                1 for m in methods
                if per_method.get(m, 0.0) >= self.threshold
            )

            combmnz = sum(
                norms[m][fname] * weights.get(m, 1.0)
                for m in methods
                if per_method.get(m, 0.0) >= self.threshold
            ) * max(n_active, 1)

            # Normalizza CombMNZ in [0,100]
            combined = float(np.clip(combmnz / n_methods * 100, 0, 100))

            best_m = max(per_method, key=lambda m: per_method[m]) if per_method else "n/a"

            fused.append(FusedResult(
                filename=fname,
                combined_score=combined,
                per_method=per_method,
                best_method=best_m,
                n_active=n_active,
                inliers={m: inliers_raw[fname].get(m, 0) for m in methods},
            ))

        fused.sort(key=lambda r: r.combined_score, reverse=True)
        return fused
