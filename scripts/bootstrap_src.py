"""Add the local ``src/`` directory to ``sys.path`` for script entry points.

This helper keeps the command-line scripts runnable from the repository root
without requiring an installed package build.
"""

from pathlib import Path
import sys


def bootstrap_src_path():
    """Add the local src directory to sys.path for script execution."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
