SHELL := /bin/bash

PRODUCTS = sl_slides.pdf

%.pdf : %.tex
	latexmk --pdf $*.tex

all : $(PRODUCTS)
	@ls -al $(PRODUCTS)

clean :
	rm -f sl_*.{aux,fdb_latexmk,fls,log,nav,out,pdf,snm,tex~,toc}
