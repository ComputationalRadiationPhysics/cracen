.PHONY: doc
doc:
	@doxygen Doxygen.ini
	@$(MAKE) -C latex all
	@mv latex/refman.pdf Code_Documentation.pdf
	@$(MAKE) -C latex clean
	@echo "Complete."
