from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SRC_PATH_STR = str(SRC_PATH)

if SRC_PATH_STR not in sys.path:
    sys.path.insert(0, SRC_PATH_STR)
