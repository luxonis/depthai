# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'DepthAI SDK Docs'
html_show_copyright=False
author = 'Luxonis'

# The full version, including alpha/beta/rc tags
release = '1.13.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",  # https://github.com/sphinx-doc/sphinx/issues/7697 wait for this and implement
    "sphinx_rtd_theme",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autodocsumm',
    'sphinx_tabs.tabs'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "collapse_navigation" : False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = '_static/images/favicon.png'
html_css_files = [
    'css/index.css',
    'https://docs.luxonis.com/en/latest/_static/css/navbar.css',
]
html_js_files = [
    'https://docs.luxonis.com/en/latest/_static/js/navbar.js',
]
html_context = {
    'meta_robots': '<meta name="robots" content="noindex, nofollow" />',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'depthai': ('https://docs.luxonis.com/projects/api/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
