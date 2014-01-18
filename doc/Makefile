.PHONY: doc
doc:
	@doxygen Doxygen.conf
	@$(MAKE) -C doc/latex all
	@mv doc/latex/refman.pdf doc/Code_Documentation.pdf
	@$(MAKE) -C doc/latex clean
	@echo "Complete."
