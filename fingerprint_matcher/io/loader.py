"""
io/loader.py
============
Caricamento delle immagini di input (frammento e schede di riferimento).
"""

from pathlib import Path
import cv2
import numpy as np
from ..config import Config


class ImageLoader:
    """Carica immagini dal filesystem con validazione."""

    def __init__(self, config: Config):
        self.cfg = config

    def load_fragment(self, path: str | Path) -> np.ndarray:
        """
        Carica l'immagine del frammento.

        Raises:
            FileNotFoundError  se il file non esiste
            ValueError         se l'immagine non è leggibile
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Frammento non trovato: {path}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Impossibile decodificare l'immagine: {path}")
        return img

    def list_references(self, folder: str | Path) -> list[Path]:
        """
        Elenca tutti i file immagine nella cartella di riferimento.

        Returns:
            lista di Path ordinata alfabeticamente

        Raises:
            FileNotFoundError se la cartella non esiste o è vuota
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Cartella non trovata: {folder}")

        files = sorted(
            p for p in folder.iterdir()
            if p.suffix in self.cfg.supported_extensions
        )
        if not files:
            raise FileNotFoundError(f"Nessuna immagine in: {folder}")
        return files

    def load(self, path: Path) -> np.ndarray | None:
        """Carica una singola immagine; restituisce None in caso di errore."""
        img = cv2.imread(str(path))
        return img
