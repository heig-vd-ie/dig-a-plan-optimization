# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dig-A-Plan-Optimization"
copyright = "2025, Mokhles-Gerami Adeleh, Tomasini Luca, Bozorg Mokhtar"
author = "Mokhles-Gerami Adeleh, Tomasini Luca, Bozorg Mokhtar"

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
    "sphinx_rtd_theme",
]


autosectionlabel_prefix_document = True


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
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
