SHELL = /bin/sh

SPHINXBUILD = sphinx-build
SPHINXOPTS =
SPHINXBUILDOPTS = -b html
SPHINXBUILDDIR = _build/html
SPHINXDOCTESTOPTS = -b doctest

.PHONY: help clean html doctest

help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "  html      to build the HTML documentation"
	@echo "  clean     to remove the build directory"

clean:
	rm -rf $(SPHINXBUILDDIR)

html:
	$(SPHINXBUILD) $(SPHINXOPTS) $(SPHINXBUILDOPTS) . $(SPHINXBUILDDIR)

doctest:
	$(SPHINXBUILD) $(SPHINXOPTS) $(SPHINXDOCTESTOPTS) . $(SPHINXBUILDDIR)