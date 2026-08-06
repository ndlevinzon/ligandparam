import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "python" / "lib"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
