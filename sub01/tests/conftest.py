"""Shared test setup: make `import flare_cnn` (and tests/fakes.py) work without installing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
