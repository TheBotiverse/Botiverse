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

import os, sys
sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, 'd:/programing/programs/anaconda/envs/pytorch/lib/site-packages/gensim')
# sys.path.insert(0, os.path.abspath('../botiverse/TODS'))

project = 'Botiverse'
copyright = '2023, Botiverse Org.'
author = 'Botiverse Org.'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


autodoc_mock_imports = ['PyAudio']

# autodoc_mock_imports = [
# 'PyAudio',
# 'benepar',
# 'cvxopt',
# 'Flask',
# 'gdown',
# 'gensim',
# 'matplotlib',
# 'multiprocess',
# 'nltk',
# 'numpy',
# 'pandas',
# 'scikit_learn',
# 'scipy',
# 'setuptools',
# 'spacy',
# 'tokenizers',
# 'torch',
# 'tqdm',
# 'transformers',
# 'pyngrok',
# 'waveglowpkg',
# 'sentence-transformers',
# 'torchaudio',
# 'soundfile',
# 'playsound',
# 'pydub',
# 'gtts',
# 'librosa',
# 'audiomentations']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'

# autodoc_mock_imports = ['gtts', 'playsound', 'gensim', 'waveglow', 'pydub', 'gdown', 'librosa', 'soundfile', '_ufuncs']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# to create the html files locally run the following command in the docs folder
# .\docs\make.bat html

autoclass_content = 'both'
