# Makefile for the SoftwareX submission.
#
#   make            -> build pyfieldml-softwarex.pdf
#   make figures    -> regenerate the six PNG figures from the source notebooks
#   make clean      -> remove LaTeX auxiliary files (keeps the PDF)
#   make distclean  -> also remove the PDF
#
# Requires: pdflatex, bibtex (TeX Live 2020+), and a working pyfieldml
# environment (uv run) for the figures/ target.

PAPER   := pyfieldml-softwarex
TEX     := $(PAPER).tex
BIB     := paper.bib
PDF     := $(PAPER).pdf
AUX_EXT := aux bbl blg log out toc brf fls fdb_latexmk synctex.gz

.PHONY: all pdf figures clean distclean

all: pdf

pdf: $(PDF)

$(PDF): $(TEX) $(BIB) $(wildcard figures/*.png)
	pdflatex -interaction=nonstopmode $(PAPER)
	bibtex   $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)

figures:
	cd ../.. && uv run python paper/softwarex/figures/generate.py

clean:
	$(foreach ext,$(AUX_EXT),rm -f $(PAPER).$(ext);)

distclean: clean
	rm -f $(PDF)
