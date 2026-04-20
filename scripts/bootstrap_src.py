from pathlib import Path
import sys


def bootstrap_src_path():
    """Add the local src directory to sys.path for script execution."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
