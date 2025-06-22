"""Config file for sphinx docs."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from datetime import datetime

project = "gymnasium-hardmaze"
copyright = f"{datetime.now().year}, Teaspoon AI"
author = "Stefano Palmieri"
release = importlib.metadata.version(project)

# Add src/ to path so autodoc finds the package that *uv pip -e .*
# just installed in editable mode.
sys.path.insert(0, os.path.abspath("../src"))

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_github_changelog",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # match other Farama projects
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

html_theme = "furo"
html_title = "HardMaze Documentation"
html_static_path = ["_static"]
html_favicon = "_static/img/favicon.svg"
html_theme_options = {
    "light_logo": "img/hardmaze.svg",
    "dark_logo": "img/hardmaze-white.svg",
    "image": "img/hardmaze-github.png",
    "description": "Gymnasium HardMaze is a reimplementation of the 'hardmaze' \
    environment used in several neuroevolution research papers.",
    "gtag": "",
    "versioning": True,
    "source_branch": "master",
    "source_directory": "docs/",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Generate Changelog -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
