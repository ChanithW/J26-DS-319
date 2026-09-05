import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make sub03/ importable

from stubs.mock_next_stages import render_habitability  # noqa: E402

render_habitability()
