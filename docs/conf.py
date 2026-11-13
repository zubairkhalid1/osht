"""Sphinx configuration for the osht documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import osht  # noqa: E402

project = "osht"
author = "Zubair Khalid"
copyright = "2026, Zubair Khalid"
release = osht.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False}
numpydoc_show_class_members = False
numpydoc_xref_param_type = False
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
if os.environ.get("OSHT_DOCS_OFFLINE"):  # build without network access
    intersphinx_mapping = {}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_title = f"osht {release}"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/zubairkhalid1/osht/",
    "source_branch": "main",
    "source_directory": "docs/",
}
