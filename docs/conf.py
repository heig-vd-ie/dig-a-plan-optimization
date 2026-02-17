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

html_theme = "furo"
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

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#336790",
        "color-brand-content": "#336790",
    },
    # Optional: customize sidebar and navigation
    "sidebar_hide_name": False,  # Show project name in sidebar
    "navigation_with_keys": True,  # Enable keyboard navigation
}
html_theme_options["footer_icons"] = [
    {
        "name": "GitHub",
        "url": "https://github.com/heig-vd-ie/dig-a-plan-optimization",
        "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
        "class": "",
    },
]
import re
from pathlib import Path


def fix_readme_links():
    readme = Path("../README.md").read_text()
    # Adjust relative links
    readme = re.sub(r"\]\(docs/", r"](", readme)
    Path("intro.md").write_text(readme)


fix_readme_links()
