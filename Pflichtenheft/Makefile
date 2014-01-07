OBJECTS = Levenberg_Marquardt.tex Speichertransfer.tex Host.tex

Pflichtenheft.pdf: $(OBJECTS) Pflichtenheft.tex
	pdflatex Pflichtenheft.tex
	
%.tex: %.md
	pandoc -o $@ $<
	
.PHONY: clean
clean:
	@rm -f $(OBJECTS)
	@rm -f Pflichtenheft.aux Pflichtenheft.out
	@rm -f Pflichtenheft.pdf
	@rm -f *~
