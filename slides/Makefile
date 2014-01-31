slides: Slides.tex
	pdflatex Slides.tex
	pdflatex Slides.tex

.PHONY: clean	
clean:
	@rm -f *.aux *.log Slides.pdf
	
show: slides
	evince Slides.pdf&
