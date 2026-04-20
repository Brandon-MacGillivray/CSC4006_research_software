"""Pytest configuration for repository-local imports.

This file ensures that the ``src/`` tree is importable when tests are run from
the repository root.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
