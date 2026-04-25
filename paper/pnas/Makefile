.PHONY: all clean

all: main.pdf

main.pdf: main.tex refs.bib
	@kpsewhich pnas-new.cls >/dev/null || { \
		echo "Missing pnas-new.cls. Install the official PNAS LaTeX template or place pnas-new.cls in paper/pnas/."; \
		exit 1; \
	}
	@kpsewhich pnas-new.bst >/dev/null || { \
		echo "Missing pnas-new.bst. Install the official PNAS LaTeX template or place pnas-new.bst in paper/pnas/."; \
		exit 1; \
	}
	@kpsewhich algorithm.sty >/dev/null || { \
		echo "Missing algorithm.sty. Install the LaTeX algorithms package required by pnas-new.cls."; \
		exit 1; \
	}
	@kpsewhich algpseudocode.sty >/dev/null || { \
		echo "Missing algpseudocode.sty. Install the LaTeX algorithmicx package required by pnas-new.cls."; \
		exit 1; \
	}
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc main.pdf
