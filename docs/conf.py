from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "ChillProp"
author = "ChillProp contributors"
copyright = "2026, ChillProp contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_class_signature = "mixed"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_mock_imports = ["equinox", "jax", "jax.numpy", "numpy"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "ChillProp documentation"

nitpicky = False

os.environ.setdefault("PYTHONPATH", str(SRC))
