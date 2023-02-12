# Makefile for venv-torch

VENV_NAME := .venv

.PHONY: all clean realclean test testclean

all: $(VENV_NAME)
	@echo "\nTo start, please run\nsource setup.sh\n"

$(VENV_NAME):
	bash setup.sh

clean: testclean
	find . -type f -name '*.py[co]' -exec rm -fv {} +
	find . -type d -name __pycache__  -exec rm -rfv {} +

realclean: clean
	find . -maxdepth 1 -type d -name $(VENV_NAME) -exec rm -rfv {} +

test:
	cd tests && pytest && cd ..

testclean:
	find tests -type d -name .pytest_cache -exec rm -rfv {} +
