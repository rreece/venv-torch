# Makefile for venv-torch

VENV_NAME := .venv

.PHONY: all install clean realclean test testclean lint

all: lint install

install: $(VENV_NAME)
	@echo "\nTo start, please run\nsource setup.sh\n"

$(VENV_NAME):
	bash setup.sh

clean: testclean
	find python -type f -name '*.py[co]' -exec rm -fv {} +
	find python -type d -name __pycache__  -exec rm -rfv {} +

realclean: clean
	find . -maxdepth 1 -type d -name $(VENV_NAME) -exec rm -rfv {} +

test:
	cd tests && pytest && cd ..

testclean:
	find tests -type f -name '*.py[co]' -exec rm -fv {} +
	find tests -type d -name __pycache__  -exec rm -rfv {} +
	find tests -type d -name .pytest_cache -exec rm -rfv {} +

lint:
	flake8 python tests --count --select=E9,F63,F7,F82 --show-source --statistics
