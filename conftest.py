"""Ensures `import src...` resolves when pytest is run from any directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
