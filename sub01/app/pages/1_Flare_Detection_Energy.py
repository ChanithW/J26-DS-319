import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make flare_cnn importable

from flare_cnn.ui import render  # noqa: E402

render(next_page="pages/2_MOCK_Next_Stage.py")
