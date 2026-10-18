# -*- coding: utf-8 -*-
"""Make `scripts/` and the repo root importable from the examples.

`python examples/foo.py` puts `examples/` on `sys.path[0]`, not the repo root,
so an example cannot `import cuped` without this. Mirrors `tests/conftest.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"


def add_scripts_to_path() -> Path:
    """Put the repo root and `scripts/` on sys.path; return the repo root."""
    for path in (str(ROOT), str(SCRIPTS)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return ROOT
