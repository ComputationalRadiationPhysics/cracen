OBJECTS = LevenbergMarquardt.tex Speichertransfer.tex Host.tex

Pflichtenheft.pdf: $(OBJECTS) Pflichtenheft.tex
	pdflatex Pflichtenheft.tex
	pdflatex Pflichtenheft.tex
	
%.tex: %.md
	pandoc -o $@ $<
	
.PHONY: clean
clean:
	@rm -f Speichertransfer.tex Host.tex
	@rm -f Pflichtenheft.aux Pflichtenheft.out
	@rm -f Pflichtenheft.pdf
	@rm -f *~
