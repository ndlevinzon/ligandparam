# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
import sys
import ase
import ffpopt

project = 'FFPOPT'
copyright = '2025, Tim Giese, Zeke Piskulich, York Group'
author = 'Tim Giese, Zeke Piskulich, York Group'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
]



templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme' 
html_static_path = ['_static']

# Add custom CSS file for responsive design
html_css_files = [
    'responsive.css',
]

# -- Cross-references --------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'ase':    ('https://wiki.fysik.dtu.dk/ase', None),
}

napoleon_preprocess_types = True
napoleon_type_aliases = {
    'numpy.array': ':class:`numpy.ndarray`',
    'Constraint': ':class:`ffpopt.Constraints.Constraint`',
    'ListOfStruct': ':class:`ffpopt.Struct.ListOfStruct`',
}

# Docstrings in this project use numpydoc-style type fields like
# ``numpy.array, shape=(nat, 3), optional`` which napoleon splits on commas.
# Most pieces aren't real classes, so silence those nitpicky warnings rather
# than rewrite every docstring. parmed has no public objects.inv.
nitpick_ignore_regex = [
    ('py:class', r'optional'),
    ('py:class', r'shape=.*'),
    ('py:class', r'default=.*'),
    ('py:class', r'\d+'),
    ('py:class', r'nat'),
    ('py:class', r'tuples'),
    ('py:class', r'Amber.*'),
    ('py:class', r'parmed\..*'),
    ('py:class', r'parmed .*'),
    ('py:class', r'ResidueTemplate'),
    # Unqualified GeomOpt is a stale ref from the JSON refactor.
    ('py:class', r'GeomOpt'),
    ('py:class', r'Constraints'),
    ('py:class', r'ffpopt\.GeomOpt\.GeomOpt'),
    ('py:class', r'ffpopt\.Constraint\.Constraint'),
    ('py:class', r'Dihedral'),
    ('py:class', r'DihedralType'),
    ('py:class', r'Constraint objects'),
    ('py:class', r'a list'),
]