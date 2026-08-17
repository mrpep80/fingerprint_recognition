from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NBISRuntime:
    """Location and capabilities of the bundled/installed NBIS runtime."""

    root: Optional[Path]
    mindtct: Optional[Path]
    bozorth3: Optional[Path]
    source: str  # bundled, external, or unavailable

    @property
    def available(self) -> bool:
        return self.mindtct is not None and self.bozorth3 is not None


def _platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        machine = "arm64" if machine in {"arm64", "aarch64"} else "x64"
        return f"darwin-{machine}"
    if system == "linux":
        machine = "arm64" if machine in {"arm64", "aarch64"} else "x64"
        return f"linux-{machine}"
    if system == "windows":
        machine = "arm64" if machine in {"arm64", "aarch64"} else "x64"
        return f"windows-{machine}"
    return f"{system}-{machine}"


def _exe(name: str) -> str:
    return name + ".exe" if platform.system().lower() == "windows" else name


def _valid_pair(root: Path) -> Optional[NBISRuntime]:
    bin_dir = root / "bin"
    m = bin_dir / _exe("mindtct")
    b = bin_dir / _exe("bozorth3")
    if m.is_file() and b.is_file():
        return NBISRuntime(root=root, mindtct=m, bozorth3=b, source="bundled")
    return None


def _candidate_roots() -> list[Path]:
    key = _platform_key()
    roots: list[Path] = []

    # PyInstaller one-folder/one-file extraction directory.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "engines" / "nbis" / "runtimes" / key)
        roots.append(Path(meipass) / "nbis" / key)

    # Source checkout / packaged application resources.
    here = Path(__file__).resolve().parent
    roots.append(here / "runtimes" / key)
    roots.append(here / "bin" / key)

    # Explicit application override, useful for support/debugging.
    env_root = os.environ.get("FINGERPRINT_NBIS_HOME")
    if env_root:
        roots.insert(0, Path(env_root).expanduser())

    return roots


def get_nbis_runtime() -> NBISRuntime:
    """Resolve bundled NBIS first, then an externally installed NBIS.

    The desktop application therefore works without Homebrew/CMake/NBIS being
    installed. A system NBIS is only a development/fallback option.
    """
    for root in _candidate_roots():
        runtime = _valid_pair(root)
        if runtime:
            return runtime

    mindtct = shutil.which("mindtct") or shutil.which("mindtct.exe")
    bozorth3 = shutil.which("bozorth3") or shutil.which("bozorth3.exe")
    if mindtct and bozorth3:
        return NBISRuntime(
            root=Path(mindtct).resolve().parent.parent,
            mindtct=Path(mindtct),
            bozorth3=Path(bozorth3),
            source="external",
        )

    return NBISRuntime(root=None, mindtct=None, bozorth3=None, source="unavailable")


def version(runtime: Optional[NBISRuntime] = None) -> Optional[str]:
    runtime = runtime or get_nbis_runtime()
    if not runtime.available:
        return None
    try:
        p = subprocess.run([str(runtime.bozorth3)], capture_output=True,
                           text=True, timeout=5)
        text = (p.stdout or p.stderr or "").strip().splitlines()
        return text[0] if text else None
    except (OSError, subprocess.SubprocessError):
        return None
