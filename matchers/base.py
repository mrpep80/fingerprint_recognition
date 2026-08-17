"""
matchers/base.py
================
Contratto comune per tutti i matcher e struttura dati MatchScore.

v2.1: aggiunto n_features_ref a MatchScore per scoring bidirezionale.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

from ..extractors.base import FeatureSet


@dataclass
class MatchScore:
    raw_matches:      List[Any]
    n_features_query: int
    is_verified:      bool = False
    n_features_ref:   int  = 0    # FIX v2.1: usato dallo scorer per copertura bidirezionale

    @property
    def n_good(self) -> int:
        return len(self.raw_matches)


class BaseMatcher(ABC):

    def __init__(self, config):
        self.cfg = config

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def match(self, query: FeatureSet, reference: FeatureSet) -> MatchScore:
        pass
