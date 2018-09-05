CWD := $(dir $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

all: pdflatex

pdflatex:
	mkdir -p build/pdflatex/
	latexmk -pdf -shell-escape -output-directory=build/pdflatex/ -synctex=1 -file-line-error futureinternet-307052.tex
	ln -sf build/pdflatex/futureinternet-307052.pdf StegNet-FutureInternet.pdf

clean:
	rm -f build/*/*.log
	rm -f build/*/*.aux
	rm -f build/*/*.dvi
	rm -f build/*/*.lof
	rm -f build/*/*.lot
	rm -f build/*/*.toc
	rm -f build/*/*.bbl
	rm -f build/*/*.blg
	rm -f build/*/*.bcf
	rm -f build/*/*.xml

clean-pdf:
	rm -f build/*/*.pdf
	
clean-all: clean clean-pdf
	rm StegNet-FutureInternet.pdf
	rm -rf build/

