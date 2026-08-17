from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from ..preprocessing.image_processor import ImageProcessor
from ..preprocessing.card_segmenter import CardSegmenter
from ..extractors import get_extractor
from ..extractors.base import FeatureSet
from ..matchers import get_matcher
from ..pipeline.verifier import RANSACVerifier
from ..pipeline.scorer import MatchScorer


@dataclass
class MatchResult:
    filename: str
    score: float
    n_inliers: int
    n_good_matches: int
    n_features: int
    method: str
    minutiae_source: str = "n/a"
    matched_finger: str = "n/a"
    n_rois: int = 0
    geometry_quality: float = 0.0
    native_score: float = 0.0

    def __lt__(self, other):
        return self.score < other.score


@dataclass
class _Pipeline:
    extractor: object
    matcher: object
    weight: float


class FingerprintEngine:
    """Motore di identificazione.

    Una scheda non viene mai confrontata come immagine intera: viene prima
    segmentata e ogni ROI viene confrontata separatamente. Il punteggio della
    scheda e' quello della migliore ROI valida.
    """

    def __init__(self, config: Config, method: str = "sift",
                 use_gabor: bool = True, try_rois: bool = True,
                 n_workers: int = 1):
        self.cfg = config
        self.method = method
        self.use_gabor = use_gabor
        self.try_rois = try_rois
        self.n_workers = n_workers
        self.preprocessor = ImageProcessor(config)
        self.segmenter = CardSegmenter(config)
        self.verifier = RANSACVerifier(config)
        self.scorer = MatchScorer(config)
        self._pipelines = self._build_pipelines(method)
        self._frag_features: Dict[str, FeatureSet] = {}
        self._frag_path: Optional[str] = None

    def prepare_fragment(self, frag_img: np.ndarray,
                         frag_path: Optional[str] = None) -> Dict[str, int]:
        self._frag_path = frag_path
        frag_proc = self.preprocessor.preprocess(frag_img, self.use_gabor)
        frag_gray = self.preprocessor.resize_to_working(self.preprocessor.to_gray(frag_img))
        stats = {}
        for pl in self._pipelines:
            # NBIS/MINDTCT should see the original fingerprint image, not the
            # Gabor/CLAHE representation used by the OpenCV descriptors.
            if hasattr(pl.extractor, "set_source_path"):
                pl.extractor.set_source_path(frag_path)
            inp = frag_gray if pl.extractor.needs_raw_input else frag_proc
            fs = pl.extractor.extract(inp)
            self._frag_features[pl.extractor.name] = fs
            stats[pl.extractor.name] = fs.n_features
        return stats

    def run_search(self, ref_folder: Path, top_n: int = 10) -> List[MatchResult]:
        ref_files = [p for p in sorted(ref_folder.iterdir())
                     if p.suffix in self.cfg.supported_extensions]
        if not ref_files:
            raise FileNotFoundError(f"Nessuna immagine in {ref_folder}")
        results: List[MatchResult] = []
        if self.n_workers > 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                futs = {ex.submit(self._process_reference, p): p for p in ref_files}
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        print(f"\n[WARN] {p.name}: {exc}")
                        results.append(MatchResult(p.name, 0, 0, 0, 0, self.method))
        else:
            for idx, p in enumerate(ref_files, 1):
                print(f"  [{idx:>3}/{len(ref_files)}] {p.name:<45}", end="\r")
                results.append(self._process_reference(p))
        print()
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n] if top_n else results

    def score_against(self, ref_img: np.ndarray) -> Tuple[float, int, int]:
        r = self._compare_reference(ref_img)
        return r.score, r.n_inliers, r.n_good_matches

    def _build_pipelines(self, method: str) -> List[_Pipeline]:
        return [_Pipeline(get_extractor(method, self.cfg), get_matcher(method, self.cfg), 1.0)]

    def _process_reference(self, ref_path: Path) -> MatchResult:
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            return MatchResult(ref_path.name, 0, 0, 0, 0, self.method)
        return self._compare_reference(ref_img, ref_path.name, str(ref_path))

    def _compare_reference(self, ref_img: np.ndarray, filename: str = "reference",
                           ref_path: Optional[str] = None) -> MatchResult:
        # Scheda: solo le 10 ROI. Impronta singola: confronto diretto.
        if self.try_rois and self.segmenter.is_card(ref_img):
            rois = self.segmenter.extract_rois(ref_img)
            best: Optional[MatchResult] = None
            for i, roi in enumerate(rois[:10]):
                r = self._compare_image(roi, None)
                r.matched_finger = (self.segmenter.FINGER_NAMES[i]
                                    if i < len(self.segmenter.FINGER_NAMES) else f"ROI {i+1}")
                r.n_rois = len(rois)
                if best is None or r.score > best.score:
                    best = r
            if best is not None:
                best.filename = filename
                return best
        r = self._compare_image(ref_img, ref_path)
        r.filename = filename
        return r

    def _compare_image(self, ref_img: np.ndarray,
                       ref_path: Optional[str] = None) -> MatchResult:
        ref_proc = self.preprocessor.preprocess(ref_img, self.use_gabor)
        ref_gray = self.preprocessor.resize_to_working(self.preprocessor.to_gray(ref_img))
        total = 0.0
        best_inl = 0
        best_good = 0
        best_gq = 0.0
        best_native = 0.0
        min_src = "n/a"
        nfeat = max((fs.n_features for fs in self._frag_features.values()), default=0)

        for pl in self._pipelines:
            frag_fs = self._frag_features.get(pl.extractor.name)
            if frag_fs is None or not frag_fs.is_valid():
                continue
            if hasattr(pl.extractor, "set_source_path"):
                # For a direct gallery fingerprint NBIS can consume the actual
                # file. For a card ROI there is deliberately no source path.
                pl.extractor.set_source_path(ref_path)
            inp = ref_gray if pl.extractor.needs_raw_input else ref_proc
            ref_fs = pl.extractor.extract(inp)

            if pl.extractor.name == "nbis":
                ms = pl.matcher.match(frag_fs, ref_fs)
                score = self.scorer.score(None, ms, frag_fs.n_features,
                                          ref_fs.n_features, method="nbis")
                best_native = ms.native_score
                best_inl = 0
                best_good = ms.n_good
                best_gq = 1.0 if ms.is_verified else 0.0
            else:
                ms = pl.matcher.match(frag_fs, ref_fs)
                vr = self.verifier.verify(frag_fs, ref_fs, ms)
                score = self.scorer.score(vr, ms, frag_fs.n_features,
                                          ref_fs.n_features, method=pl.extractor.name)
                if vr.n_inliers > best_inl:
                    best_inl = vr.n_inliers
                    best_good = ms.n_good
                    best_gq = vr.geometry_quality

            total += score * pl.weight
            if pl.extractor.name == "minutiae":
                min_src = frag_fs.metadata.get("source", "n/a")

        return MatchResult("reference", float(np.clip(total, 0, 100)),
                           best_inl, best_good, nfeat, self.method,
                           min_src, geometry_quality=best_gq,
                           native_score=best_native)
