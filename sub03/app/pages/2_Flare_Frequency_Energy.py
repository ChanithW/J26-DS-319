import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make sub03/ importable

from subtopic3.ui import render  # noqa: E402

render()
