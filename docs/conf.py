# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dig-A-Plan-Optimization"
copyright = "2025, HEIG-VD (Institute of Energies)"
author = "Institute of Energies"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_size",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
    "pydata_sphinx_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosectionlabel_prefix_document = True


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = [
    "custom.css",  # Include the custom CSS file
]

sphinx_rtd_size_width = "75%"

mathjax3_config = {
    "chtml": {"displayAlign": "left"},
    "tex": {
        "tags": "ams",  # Use AMS-style tagging for equations
        "tagSide": "right",  # Align equation labels to the left
        "tagIndent": "0em",  # Optional: adjust the indentation
        "useLabelIds": True,  # Add labels to elements
    },
}

html_sidebars = {
    "**": [
        # "globaltoc.html",  # This shows the full table of contents
    ]
}
html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,  # How many levels to show in right TOC
    "navigation_depth": 4,  # How deep the left navigation goes
    "pygment_light_style": "default",  # Code highlighting style for light mode
}

import re
from pathlib import Path


def fix_readme_links():
    readme = Path("../README.md").read_text()
    # Adjust relative links
    readme = re.sub(r"\]\(docs/", r"](", readme)
    Path("intro.md").write_text(readme)


fix_readme_links()
