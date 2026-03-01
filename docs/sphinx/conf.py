"""Sphinx configuration for Double ML Time Series documentation."""

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add repo root to sys.path so autodoc can import src.*
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information ------------------------------------------------------
project = "Double ML Time Series"
copyright = "2026, Brandon Behring"
author = "Brandon Behring"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon (docstring parsing) ---------------------------------------------
# Project uses mixed Google-style (25 files) and NumPy-style (5 files)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
# Use :ivar: for Attributes sections — prevents duplicate object descriptions
# when autodoc also documents dataclass fields as class attributes
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Autodoc configuration ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
# "signature" keeps types in the function/class signature only,
# avoiding collision with Napoleon's Attributes section
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"

# Mock imports for heavy/optional dependencies not needed at doc build time
autodoc_mock_imports = [
    "fredapi",
    "econml",
    "dowhy",
    "causalml",
    "doubleml",
    "xgboost",
    "lightgbm",
    "linearmodels",
    "plotly",
    "ipywidgets",
    "papermill",
    "seaborn",
]

# -- Autosummary --------------------------------------------------------------
autosummary_generate = True

# -- Intersphinx (cross-reference external projects) --------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MathJax configuration ----------------------------------------------------
# Macros matching the book's LaTeX preamble for consistency
mathjax3_config = {
    "tex": {
        "macros": {
            "E": [r"\mathbb{E}\left[#1\right]", 1],
            "Var": [r"\text{Var}\left(#1\right)", 1],
            "Cov": [r"\text{Cov}\left(#1, #2\right)", 2],
            "Prob": [r"\mathbb{P}\left(#1\right)", 1],
            "R": r"\mathbb{R}",
            "N": r"\mathcal{N}",
            "indep": r"\perp\!\!\!\perp",
            "plim": r"\overset{p}{\to}",
            "dto": r"\overset{d}{\to}",
            "iid": r"\overset{\text{iid}}{\sim}",
            "ols": r"\hat{\beta}_{\text{OLS}}",
        }
    }
}

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Double ML Time Series"

html_theme_options = {
    "repository_url": "https://github.com/bbehring/double_ml_time_series",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

# -- Copybutton configuration -------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Suppress known harmless warnings -----------------------------------------
suppress_warnings = [
    "autodoc.import",
    "sphinx_autodoc_typehints.forward_reference",
]
